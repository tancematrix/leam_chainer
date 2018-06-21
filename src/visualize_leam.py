"""Evaluate trained model.

Usage:
$ python evaluate.py --best-model <path/to/best_model>

"""
import argparse
import os
import sys

import numpy

import chainer
import chainer.links as L
import chainer.functions as F

from gensim.models import KeyedVectors

from leam import LEAM
from leam import load_data
from leam import assign_id_to_document
from sty import fg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_id', '-v', type=int, default=64,
                        help='Validation document id to analyze')
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of units')
    parser.add_argument('--best-model', help='path to best model')
    parser.add_argument('--window', '-w', type=int, default=20,
                        help='Window Size')
    args = parser.parse_args()

    DATA_DIR = '/baobab/kiyomaru/2018-shinjin/jumanpp.midasi'
    PATH_TO_TEST = os.path.join(DATA_DIR, 'test.csv')
    PATH_TO_WE = '/share/data/word2vec/2016.08.02/w2v.midasi.256.100K.bin'

    # load test data
    test_x, test_y = load_data(PATH_TO_TEST)
    word_vectors = KeyedVectors.load_word2vec_format(PATH_TO_WE, binary=True)
    word2index = {}
    for index, word in enumerate(word_vectors.index2word):
        word2index[word] = index

    # convert document to ids
    test_ids = assign_id_to_document(test_x, word2index)

    # convert test_y to numpy.array
    y_true = numpy.array(test_y)

    # define model
    model = LEAM(
        n_vocab=len(word2index),
        n_embed=word_vectors.vector_size,
        n_units=args.unit,
        n_class=4,
        n_window=args.window,
        W=None
    )
    model = L.Classifier(model)

    # load pre-trained model
    try:
        chainer.serializers.load_npz(args.best_model, model)
    except Exception as e:
        print('error:', str(e))
        sys.exit(1)

    # predict labels for test data
    with chainer.using_config('train', False):
        y_true = test_y[args.val_id]
        x = test_ids[args.val_id]
        x = x[None, :]
        y_pred, beta = model.predictor.analyze(x)

        # visualize
        doc = ["新聞", "雑誌", "教科書", "ブログ"]
        print("prediction:"+doc[numpy.argmax(y_pred)])
        print("fact:"+doc[y_true])

        print(beta.max(), beta.min())
        print('beta average:', numpy.average(beta))
        print('beta var:', numpy.var(beta))
        beta[0][0] = (beta[0][0] - beta[0][0].min()) / beta[0][0].max()  # scale beta

        for word, beta_for_word in zip(test_x[args.val_id].split(' '), beta[0][0]):
            r = 255
            g = 255 - int(255 * beta_for_word)
            b = 255 - int(255 * beta_for_word)
            print(fg(r, g, b) + word + fg.rs, end=' ')
        else:
            print('')

if __name__ == '__main__':
    main()
