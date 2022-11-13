# Копипастим код из первого пункта

from requests import get
from gzip import decompress
from numpy import zeros, argmin, amax, unravel_index
from math import log
from sklearn.cluster import KMeans, AgglomerativeClustering
from json import dumps
from numpy.linalg import norm


BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/'
WORDS_FILE_NAME = 'vocab.kos.txt'
ARCHIVE_NAME = 'docword.kos.txt.gz'
TEXTS_AMOUNT = 3430
WORDS_AMOUNT = 6906

if __name__ == '__main__':
    data = [[int(item) for item in line.split()] for line in decompress(get(BASE_URL + ARCHIVE_NAME).content).decode('utf-8').splitlines(False)[3:]]
    texts_to_words, words_to_texts = dict(), dict()
    for text, word, count in data:
        if text not in texts_to_words:
            texts_to_words[text] = []
        if word not in words_to_texts:
            words_to_texts[word] = []
        texts_to_words[text].append((word, count))
        words_to_texts[word].append(text)

    tf = zeros((TEXTS_AMOUNT, WORDS_AMOUNT))
    idf = zeros((WORDS_AMOUNT,))

    for idx in range(WORDS_AMOUNT):
        idf[idx] = log(TEXTS_AMOUNT / len(words_to_texts[idx + 1]))

    for text_idx in range(TEXTS_AMOUNT):
        text_data = texts_to_words[text_idx + 1]
        max_occurences = max(count for word, count in text_data)
        for word, count in text_data:
            tf[text_idx, word - 1] = count / max_occurences

    dataset = tf.copy()
    for text_idx in range(TEXTS_AMOUNT):
        dataset[text_idx, :] *= idf

    for clustering_class in KMeans, AgglomerativeClustering:
        clustering_implementation = clustering_class()
        clustering_implementation.fit(dataset)
        labels_to_size = dict()
        for label in clustering_implementation.labels_:
            if label not in labels_to_size:
                labels_to_size[label] = 0
            labels_to_size[label] += 1
        with open(f'{type(clustering_implementation)}.json', 'w') as fd:
            fd.write(dumps(labels_to_size))