from os import mkdir
from os.path import join, exists
from random import shuffle

from numpy import int64, floor
from pandas import read_csv, to_datetime
from torch import nn, tensor, eq, max, save, device, LongTensor

from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss, Module, Linear

from torch.nn.functional import softmax, sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Константы
INPUT_DIR = 'inputs'
BATCH_SIZE = 16


# Класс преобразования
class cbwd:
    dictionary = {
        'cv': 0,
        'NE': 1,
        'NW': 2,
        'SE': 3,
        'SW': 4
    }

    @classmethod
    def encode(cls, value):
        return cls.dictionary[value]

    @classmethod
    def decode(cls, value):
        return list(cls.dictionary.keys())[value]


# Модели
class LinearPredictor(Module):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        self.predictor = Linear(10, 1)

    def forward(self, x):
        x = self.predictor(x)
        return x


class LogisticPredictor(Module):
    def __init__(self):
        super(LogisticPredictor, self).__init__()
        self.predictor = Linear(10, 5)

    def forward(self, x):
        x = self.predictor(x)
        # Softmax зашито в функцию ошибок, будем использовать CrossEntropy
        return x


# Класс датасета
class ShanghaiDataset(Dataset):
    # В соответствии со спецификацией реализовываем методы init, getitem и len

    def __init__(self, filepath, label):
        # Читаем датасет
        dataset = read_csv(filepath)

        # Очевидно, что год вообще не имеет значения, а отдельно месяц и день также неинтересны.
        # Информация о сезоне уже содержится в месяце.

        # Преобразуем год, месяц и день во временную метку,
        # исключим год и отнормируем

        timestamps = to_datetime(dataset[['year', 'month', 'day']]).values.view(float) // 10 ** 9
        SECONDS_IN_YEAR = int(60 * 60 * 24 * 365.25)
        timestamps %= SECONDS_IN_YEAR
        timestamps = timestamps / SECONDS_IN_YEAR
        dataset['timestamps'] = timestamps

        # Выкидываем год, месяц, день и время года, а также номер
        dataset.drop(columns=['No', 'year', 'month', 'day', 'season'], inplace=True)

        # PM где-то NA, а где-то нет. При этом в описании есть информация только про PM в общем -- объединим эти столбцы.
        dataset['PM_Jingan'] /= dataset['PM_Jingan'].max()
        dataset['PM_US Post'] /= dataset['PM_US Post'].max()
        dataset['PM_Xuhui'] /= dataset['PM_Xuhui'].max()

        dataset['PM'] = dataset[['PM_Jingan', 'PM_US Post', 'PM_Xuhui']].mean(axis=1)

        # Выкидываем старые
        dataset.drop(['PM_Jingan', 'PM_US Post', 'PM_Xuhui'], axis=1, inplace=True)

        print(dataset.shape)

        # Выкидываем NaN-ы
        dataset.dropna(inplace=True)

        # Преобразовываем cbwd
        dataset['cbwd'] = dataset['cbwd'].apply(cbwd.encode)

        # Кажется, готово
        print(dataset.head())

        self._dataframe = dataset

        # Ставим на первое место интересующий нас столбец
        label_column = self._dataframe.pop(label)
        self._dataframe.insert(0, label, label_column)

    def __getitem__(self, index):
        row = self._dataframe.iloc[index].to_numpy()
        # label и features
        return tensor(row[1:]).float(), tensor(row[0]).float()

    def __len__(self):
        return len(self._dataframe)


# Функция обучения
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device=device('cpu'), task_num=0):
    for epoch in range(epochs):
        # Для каждой единицы разбиения в случае тренировочной выборки
        # используем optimizer, в случае валидационной просто считаем loss
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            # print('target:', targets)

            # CrossEntropy кушает LongTensor,
            # а для MSE нужно преобразовать (11) в (11, 1)
            if task_num:
                targets = targets.type(LongTensor)
            else:
                targets = targets.unsqueeze(1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            if task_num:
                targets = targets.type(LongTensor)
            else:
                targets = targets.unsqueeze(1)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
        valid_loss /= len(val_loader.dataset)

        # На всякий случай сохраняем веса
        if not exists('weights'):
            mkdir('weights')

        save(model.state_dict(), f'weights/{epoch}_ep.pth')

        print(f'Epoch: {epoch}, training Loss: {training_loss:.2f}, validation Loss: {valid_loss:.2f}')


if __name__ == '__main__':
    dataset_filename = input('Input CSV filename: ')
    dataset_filepath = join(INPUT_DIR, dataset_filename)

    # 0 -- 1-е задание, 1 -- 2-е.
    task_num = int(input('Choose task: 0 or 1: '))

    dataset = ShanghaiDataset(dataset_filepath, label=('cbwd' if task_num else 'PRES'))

    # Делим на тренировочную и валидационную выборки
    validation_split = 0.15
    indices = list(range(len(dataset)))
    shuffle(indices)
    split = int(floor(validation_split * len(dataset)))
    train_indices, validation_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                   sampler=valid_sampler)

    # Выбираем нужную модель и функцию ошибок
    predictor = LogisticPredictor() if task_num else LinearPredictor()
    loss_function = CrossEntropyLoss() if task_num else MSELoss()
    # Будем считать на видеокарте
    torch_device = device('cuda')
    predictor.to(device=torch_device)

    # learning rate -- скорость спуска
    # Адам -- всё равно лучше, чем руками подбирать
    optimizer = Adam(predictor.parameters(), lr=0.001)
    train(model=predictor, optimizer=optimizer, loss_fn=loss_function,
          train_loader=train_loader, val_loader=validation_loader,
          epochs=30, device=torch_device, task_num=task_num)
