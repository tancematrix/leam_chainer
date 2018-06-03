import argparse

import os
from gensim.models import KeyedVectors
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Parameter
from chainer import training
from chainer.training import extensions
import numpy as np

PAD = -1
VALIDATION_SIZE = 500


class MLP(chainer.Chain):

    def __init__(self, n_vocab, n_embed, n_units, n_class, W=None):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=W, ignore_label=PAD)
            self.l1 = L.Linear(n_embed, n_units)
            self.l2 = L.Linear(n_units, 4)

            # parameters to compute attention score
            self.lw = L.Linear(None, 1)
            self.lb = L.Bias(axis=1, shape=(n_class,))
            # c:label embedding parameter
            self.c = Parameter(
                initializer=self.xp.random.randn(n_embed, n_class).astype(self.xp.float32)
            )

    def __call__(self, x):
        e = self.embed(x)
        # attention scare
        batch_size, sentence_len, n_embed = e.shape
        n_embed, n_class = self.c.shape
        # g = g / g_norm : normalize

        c = F.broadcast_to(self.c, (batch_size, n_embed, n_class))
        g = F.matmul(e, c)
        g = F.rollaxis(g, axis=2, start=1)

        # padding

        window_size = 0
        mat_pad = np.array([[[PAD] * window_size] * n_class] * batch_size, dtype=np.float32)
        g = F.concat((mat_pad, g, mat_pad), axis=2)

        # g = [g]*sentence_len
        g_l = []
        for i in range(sentence_len):
            g_i = F.get_item(g, (Ellipsis, slice(i, i + 2 * window_size + 1)))
            g_l.append(g_i)

        g = F.stack(g_l, axis=1)
        # g.shape = (batch_size, sentence_len, n_class,  window_size, n_class)

        g = F.rollaxis(g, axis=3, start=1)
        g = F.reshape(g, (batch_size * sentence_len * n_class, window_size * 2 + 1))
        u_w = F.relu(self.lw(g))
        # g.shape = (batch_size*sentence_len*n_class, 1)
        u_w = F.reshape(u_w, (batch_size * sentence_len, n_class))

        u_b = self.lb(u_w)
        # g.shape = (batch_size*sentence_len, n_class)

        u = F.max(u_b, axis=1)
        u = F.reshape(u, (batch_size, sentence_len))
        beta = F.softmax(u)
        beta = F.expand_dims(beta, axis=1)
        h1 = F.squeeze(F.matmul(beta, e)) / self.xp.sum(x != PAD, axis=1)[:, None]
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
    # keyboard arguments
    parser = argparse.ArgumentParser(description='Chainer example: WordClassification')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=64,
                        help='Number of units')
    args = parser.parse_args()

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

    # validation
    train_ids, valid_ids = train_ids[VALIDATION_SIZE:], train_ids[:VALIDATION_SIZE]
    train_y, valid_y = train_y[VALIDATION_SIZE:], train_y[:VALIDATION_SIZE]

    # define a model
    train = chainer.datasets.TupleDataset(train_ids, train_y)
    test = chainer.datasets.TupleDataset(test_ids, test_y)
    valid = chainer.datasets.TupleDataset(valid_ids, valid_y)

    model = MLP(
        n_vocab=len(word2index),
        n_embed=word_vectors.vector_size,
        n_units=args.unit,
        n_class=4,
        W=word_vectors.vectors
    )

    model.embed.disable_update()

    model = L.Classifier(model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))

    trainer.extend(
        extensions.snapshot_object(model, 'best_model'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss')
    )

    trainer.run()

    return 0


if __name__ == '__main__':
    main()
