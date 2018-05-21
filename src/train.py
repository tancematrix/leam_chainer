import os
from gensim.models import KeyedVectors


def load_data(path_to_data):
    data = []
    with open(path_to_data) as raw_data:
        for line in raw_data:
            data.append((line.strip().split(',')))
    return data


def main():
    # load data
    DATA_DIR = '/baobab/kiyomaru/2018-shinjin/jumanpp.midasi'
    PATH_TO_TRAIN = os.path.join(DATA_DIR, 'train.csv')
    PATH_TO_TEST = os.path.join(DATA_DIR, 'test.csv')
    PATH_TO_WE = '/share/data/word2vec/2016.08.02/w2v.midasi.256.100K.bin'
    train = load_data(PATH_TO_TRAIN)
    test = load_data(PATH_TO_TEST)
    word_vectors = KeyedVectors.load_word2vec_format(PATH_TO_WE, binary=True)
    for index, word in enumerate(word_vectors.index2word):
        word2index[word] = index
    from IPython import embed;
    embed()

    # convert document to ids

    # define a model

    # get results

    return 0


if __name__ == '__main__':
    main()
