from os import mkdir
from treelib import Tree
from copy import deepcopy
from random import shuffle

from os.path import join, exists

from pandas import read_csv
from torch import tensor, save, device, LongTensor

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Module, Linear, ParameterDict

from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader

# Константы
INPUT_DIR = 'inputs'
BATCH_SIZE = 16


# Вывод как в JavaScript
class console:
    log = print


# Marie Laforet -- Ivan, Boris et Moi, 1967
# https://en.wikipedia.org/wiki/Ivan,_Boris_et_moi
NAMES = ['Anton', 'Ivan', 'Boris', 'Marie', 'Rebecca', 'Pola', 'Yohanna', 'Sacha',
         'Sonia', 'David', 'Dimitri', 'Yanni', 'Natacha']

# Treasure Island
NAMES += ['Jim Hawkins', 'Captain Smollett', 'Billy Bones', 'Doctor Livesey',
          'Squire Trelawney', 'Captain Flint', 'Blind Pew', 'Benn Gunn']

# The Expanse
NAMES += ['Camina Drummer', 'Chrisjen Avasarala', 'Amos Burton', 'Bobbie Draper',
          'Klaes Ashford', 'Alex Kamal', 'Josephus Miller', 'Elvi Okoye']


# Модели
class DecisionTree(Module):
    def __init__(self, depth, names):
        super(DecisionTree, self).__init__()

        self.depth = depth

        self.names = deepcopy(names)
        shuffle(self.names)
        self.name_index = 0

        self.tree = Tree()
        self.tree.create_node('root', 'root', data=Linear(4, 1))
        self._build_branch(parent_node='root', branch_depth=self.depth - 1)
        print('Built tree:')
        self.tree.show()

        for node in self.tree.expand_tree(mode=Tree.DEPTH):
            self.register_module(node, self.tree.get_node(node).data)

    def _build_branch(self, parent_node, branch_depth):
        if branch_depth == 0:
            return
        left_node_name = self.get_name()
        self.tree.create_node(left_node_name, left_node_name, parent=parent_node, data=Linear(4, 1))
        self._build_branch(parent_node=left_node_name, branch_depth=branch_depth - 1)

        right_node_name = self.get_name()
        self.tree.create_node(right_node_name, right_node_name, parent=parent_node, data=Linear(4, 1))
        self._build_branch(parent_node=right_node_name, branch_depth=branch_depth - 1)

    def get_name(self):
        if self.name_index == len(self.names):
            raise IndexError(f'Not enough names to build the tree with depth {self.depth}')
        self.name_index += 1
        return self.names[self.name_index - 1]

    def get_full_node(self, name) -> (str, Module, list):
        node = self.tree.get_node(name)
        node_name = node.identifier
        node_module = node.data
        node_children = self.tree.is_branch(name)
        return node_name, node_module, node_children

    def forward(self, x):
        name = 'root'
        while True:
            node, module, children = self.get_full_node(name=name)
            if children:
                z = relu(module(x))
                name = children[0] if z == 0 else children[1]
            else:
                # Как и в прошлой задачке, не используем softmax
                # благодаря CrossEntropyLoss
                x = module(x)


# Класс датасета
class ODDataset(Dataset):
    # В соответствии со спецификацией реализовываем методы init, getitem и len

    def __init__(self, filepath, label):
        # Читаем датасет
        dataset = read_csv(filepath)

        # Очевидно, что год вообще не имеет значения, а отдельно месяц и день также неинтересны.
        # Информация о сезоне уже содержится в месяце.

        # Преобразуем год, месяц и день во временную метку,
        # исключим год и отнормируем

        # Выкидываем год, месяц, день и время года, а также номер
        dataset.drop(columns=['id', 'date'], inplace=True)

        print(dataset.shape)

        # Выкидываем NaN-ы
        dataset.dropna(inplace=True)

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
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device=device('cuda')):
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

            # CrossEntropy кушает LongTensor
            targets = targets.type(LongTensor)
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
            targets = targets.type(LongTensor)
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
    train_filename, val_filename = 'datatest.txt', 'datatraining.txt'
    # train_filename, val_filename = input('Input CSV filenames: ').split()
    train_filepath = join(INPUT_DIR, train_filename)
    val_filepath = join(INPUT_DIR, val_filename)

    train_dataset = ODDataset(train_filepath, label=('Occupancy'))
    val_dataset = ODDataset(val_filepath, label=('Occupancy'))

    # Делим на тренировочную и валидационную выборки

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Выбираем нужную модель и функцию ошибок
    model = DecisionTree(depth=4, names=NAMES)
    loss_function = CrossEntropyLoss()
    # Будем считать на видеокарте
    torch_device = device('cuda')
    model.to(device=torch_device)

    # learning rate -- скорость спуска
    # Адам -- всё равно лучше, чем руками подбирать
    optimizer = Adam(model.parameters(), lr=0.001)
    train(model=model, optimizer=optimizer, loss_fn=loss_function,
          train_loader=train_loader, val_loader=validation_loader,
          epochs=30, device=torch_device)
