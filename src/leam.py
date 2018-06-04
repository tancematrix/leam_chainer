import argparse

import os
from gensim.models import KeyedVectors
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Parameter
from chainer import training
from chainer.training import extensions
from chainer import reporter
import numpy as np

PAD = -1
VALIDATION_SIZE = 500


class MLP(chainer.Chain):

    def __init__(self, n_vocab, n_embed, n_units, n_class, n_window, W=None):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=W, ignore_label=PAD)
            self.l1 = L.Linear(n_embed, n_units)
            self.l2 = L.Linear(n_units, 4)

            # parameters to compute attention score
            self.l3 = L.Linear(n_window * 2 + 1, 1)
            # c:label embedding parameter
            self.c = Parameter(
                initializer=self.xp.random.randn(n_class, n_embed).astype(self.xp.float32)
            )
        self.n_window = n_window
        self.n_class = n_class

    def __call__(self, x):
        batch_size, sentence_len = x.shape
        n_class, n_embed = self.c.shape

        e = self.embed(x)
        ep = self.pad_sequence(e)

        # Eq. (2)
        c = F.broadcast_to(self.c, (batch_size, n_class, n_embed))
        g = F.matmul(ep, c, transb=True)
        norm_ep = self.xp.expand_dims(self.xp.linalg.norm(ep.data, axis=2), axis=2)
        norm_c = self.xp.expand_dims(self.xp.linalg.norm(c.data, axis=2), axis=2)
        denom = self.xp.matmul(norm_ep, self.xp.transpose(norm_c, (0, 2, 1))) + 1e-10  # avoid zero division
        g = g / denom

        # Eq. (3)
        g = self.make_ngram(g)  # (batch_size, sentence_len,  window_size, n_class)
        g = F.reshape(g, (batch_size * sentence_len * n_class, self.n_window * 2 + 1))
        u = F.relu(self.l3(g))  # (batch_size * sentence_len * n_class, 1)
        u = F.reshape(u, (batch_size, sentence_len, n_class))

        # Eq. (4)
        m = F.max(u, axis=2)  # (batch_size, sentence_len)

        # Eq. (5)
        mask = (x == PAD).astype(self.xp.float32) * -1024.0  # make attention-scores for PAD 0
        m = m + mask
        beta = F.softmax(m)
        beta = F.expand_dims(beta, axis=1)

        # Eq. (6)
        z = F.squeeze(F.matmul(beta, e))  # (batch_size, n_embed)

        # f_2
        h = F.relu(self.l1(z))
        return self.l2(h)

    def pad_sequence(self, e):
        batch_size, sentence_len, n_embed = e.shape
        pad = self.xp.full((batch_size, self.n_window), PAD, dtype=self.xp.int32)
        ep = self.embed(pad)
        return F.concat((ep, e, ep), axis=1)

    def make_ngram(self, g):
        _, sentence_len, _ = g.shape
        sentence_len = sentence_len - self.n_window * 2
        return F.stack([g[:, i:i + self.n_window * 2 + 1:, :] for i in range(sentence_len)], axis=1)

    def regularize(self):
        # Eq. (9)
        h = F.relu(self.l1(self.c))
        y = self.l2(h)
        return F.softmax_cross_entropy(y, self.xp.arange(0, self.n_class))


class LeamClassifier(L.Classifier):

    def __call__(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        self.loss = self.loss + self.predictor.regularize()
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


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
    parser.add_argument('--window', '-w', type=int, default=20,
                        help='Window Size')
    parser.add_argument('--gradclip', '-c', type=float, default=3.0,
                        help='Gradient norm threshold to clip')
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
        n_window=args.window,
        W=word_vectors.vectors
    )

    model.embed.disable_update()

    model = LeamClassifier(model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradclip))

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
