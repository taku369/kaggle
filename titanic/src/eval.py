# -*- coding: utf-8 -*-

import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from model import TitanicModel
from utils import load_train_data, load_test_data


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--hidden_ch', default=6, type=int)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--small', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    train_ids, train_data, train_labels, med = load_train_data('data/train.csv', small=args.small)
    test_ids, test_data = load_test_data('data/test.csv', med, small=args.small)

    print(train_data.shape, test_data.shape)

    if args.normalize:
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        test_data = (test_data - train_mean) / train_std

    print(test_data.shape)

    model = TitanicModel(hidden_ch=args.hidden_ch, is_dropout=False)
    ckpt_path = './ckpt/%s/ckpt' % args.name
    model.load_weights(ckpt_path)

    predictions = model.predict(test_data)
    
    result = []
    for id, pred in zip(test_ids, predictions):
        if pred[0] < 0.5:
            label = 0
        else:
            label = 1
        result.append('%d,%d' % (id, label))
        
    with open('result_%s.csv' % args.name, 'w') as fout:
        fout.write('PassengerId,Survived\n')
        fout.write('\n'.join(result))
