from requests import get
from gzip import decompress
from numpy import zeros, argmin, amax, unravel_index
from math import log
from json import dumps
from numpy.linalg import norm


BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/'
WORDS_FILE_NAME = 'vocab.kos.txt'
ARCHIVE_NAME = 'docword.kos.txt.gz'
TEXTS_AMOUNT = 3430
WORDS_AMOUNT = 6906

if __name__ == '__main__':
    # раз на датасеты такая хорошая ссылка, не будем закачивать файлик локально
    data = [[int(item) for item in line.split()] for line in decompress(get(BASE_URL + ARCHIVE_NAME).content).decode('utf-8').splitlines(False)[3:]]
    texts_to_words, words_to_texts = dict(), dict()
    for text, word, count in data:
        if text not in texts_to_words:
            texts_to_words[text] = []
        if word not in words_to_texts:
            words_to_texts[word] = []
        texts_to_words[text].append((word, count))
        words_to_texts[word].append(text)

    #  Чтобы наиболее корректно подсчитать метрику, вычислим коэффициенты tf-idf
    tf = zeros((TEXTS_AMOUNT, WORDS_AMOUNT))
    idf = zeros((WORDS_AMOUNT,))

    for idx in range(WORDS_AMOUNT):
        idf[idx] = log(TEXTS_AMOUNT / len(words_to_texts[idx + 1]))

    for text_idx in range(TEXTS_AMOUNT):
        text_data = texts_to_words[text_idx + 1]
        max_occurences = max(count for word, count in text_data)
        for word, count in text_data:
            tf[text_idx, word - 1] = count / max_occurences

    # Строим датасет в виде "текст: ['tf-idf слова 1', 'tf-idf слова 2', ...]"
    dataset = tf.copy()
    for text_idx in range(TEXTS_AMOUNT):
        dataset[text_idx, :] *= idf

    # print(dataset)

    # Упорядочиваем расстояние между векторами по убыванию
    # Само расстояние храним в виде [(расстояние1_2, текст 1, текст 2), ...]
    ordered_distances = list()

    for row_idx in range(TEXTS_AMOUNT):
        for col_idx in range(row_idx + 1, TEXTS_AMOUNT):
            distance = norm(dataset[row_idx] - dataset[col_idx])
            ordered_distances.append((distance, row_idx, col_idx))

    ordered_distances.sort(key=lambda item: item[0], reverse=True)


    # Собственно, сам алгоритм

    # edges -- представление графа
    edges = list()
    # количество вершин, остающихся изолированными
    isolated = set(range(TEXTS_AMOUNT))

    min_distance, min_left, min_right = ordered_distances.pop()
    edges.append((min_distance, min_left, min_right))
    isolated.remove(min_left)
    isolated.remove(min_right)

    # Можно пойти выпить чаю
    iterations = 0
    while isolated:
        iterations += 1
        if iterations % 200 == 0:
            print(f'{len(isolated)} texts left')
        for distances_idx in range(len(ordered_distances) - 1, 0, -1):
            distance, left, right = ordered_distances[distances_idx]
            if left in isolated != right in isolated:
                if left not in isolated:
                    left, right = right, left
                edges.append((distance, left, right))
                isolated.remove(left)
                ordered_distances.pop(distances_idx)
                break

    # Сортируем узлы
    edges.sort(key=lambda item: item[0], reverse=True)
    with open('edges.json', 'w') as fd:
        fd.write(dumps(edges, indent=4))

    clusters_amount = int(input('Please input the amount of clusters:'))
    edges = edges[:-clusters_amount]

    # Готово!
    print(edges)







