import os
from gensim.models import KeyedVectors


def load_data(path_to_data):
    xs = []
    ys = []
    with open(path_to_data) as raw_data:
        for line in raw_data:
            y, x = line.strip().split(',')
            xs.append(x)
            ys.append(y)
    return xs, ys


def assign_id_to_document(xs, word2index):
    ids = []
    for x in xs:
        words = x.split(' ')
        cur_ids = []
        for word in words:
            id = word2index.get(word, word2index['<UNK>'])
            cur_ids.append(id)
        ids.append(cur_ids)
    return ids


def main():
    # load data
    DATA_DIR = '/baobab/kiyomaru/2018-shinjin/jumanpp.midasi'
    PATH_TO_TRAIN = os.path.join(DATA_DIR, 'train.csv')
    PATH_TO_TEST = os.path.join(DATA_DIR, 'test.csv')
    PATH_TO_WE = '/share/data/word2vec/2016.08.02/w2v.midasi.256.100K.bin'
    train_x, train_y = load_data(PATH_TO_TRAIN)
    test_x, test_y = load_data(PATH_TO_TEST)
    word_vectors = KeyedVectors.load_word2vec_format(PATH_TO_WE, binary=True)
    word2index = {}
    for index, word in enumerate(word_vectors.index2word):
        word2index[word] = index

    # convert document to ids
    train_ids = assign_id_to_document(train_x, word2index)
    # define a model

    # get results

    return 0


if __name__ == '__main__':
    main()
