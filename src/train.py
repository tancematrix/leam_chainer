import os
from gensim.models import KeyedVectors
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

PAD = -1


class MLP(chainer.Chain):

    def __init__(self, n_vocab, n_embed, n_units, W=None):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=W)
            self.l1 = L.Linear(n_embed, n_units)
            self.l2 = L.Linear(n_units, 4)

    def __call__(self, x):
        e = self.embed(x)
        h1 = F.average(e, axis=1)
        h2 = F.relu(self.l1(h1))
        return self.l2(h2)


def load_data(path_to_data):
    xs = []
    ys = []
    with open(path_to_data) as raw_data:
        for line in raw_data:
            y, x = line.strip().split(',')
            xs.append(x)
            ys.append(int(y))
    return xs, ys


def assign_id_to_document(xs, word2index, max_length=100):
    ids = []
    for x in xs:
        words = x.split(' ')
        cur_ids = [PAD] * max_length
        for index, word in enumerate(words[:max_length]):
            id = word2index.get(word, word2index['<UNK>'])
            cur_ids[index] = id
        ids.append(cur_ids)
    return np.array(ids, np.int32)


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
    test_ids = assign_id_to_document(test_x, word2index)

    # define a model
    train = chainer.datasets.TupleDataset(train_ids, train_y)
    test = chainer.datasets.TupleDataset(test_ids, test_y)

    model = MLP(
        n_vocab=len(word2index),
        n_embed=128,
        n_units=64
    )
    model = L.Classifier(model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    batchsize = 64
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    gpu = -1
    epoch = 10
    out = 'result'
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()

    return 0


if __name__ == '__main__':
    main()
