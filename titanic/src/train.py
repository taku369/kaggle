# -*- coding: utf-8 -*-

import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from model import TitanicModel
from dataset import load_train_data, load_test_data


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_ch', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--val_num', default=100, type=int)
    return parser.parse_args()


def predict_labels(probs):
    labels = []
    for prob in probs:
        if prob > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels


if __name__ == '__main__':
    args = parse()

    # データ読み込み
    train_ids, train_data, train_labels, med, mean, std = load_train_data('data/train.csv')

    # バリデーションデータの用意
    val_data = train_data[:args.val_num]
    val_labels = train_labels[:args.val_num]
    train_data = train_data[args.val_num:]
    train_labels = train_labels[args.val_num:]

    # モデル設定
    model = TitanicModel(hidden_ch=args.hidden_ch)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=tf.losses.log_loss,
                  metrics=['accuracy'])
    
    # 学習
    ckpt_path = './ckpt/titanic'
    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss')]
    model.fit(train_data, train_labels, 
              batch_size=4, epochs=args.max_epoch, callbacks=callbacks,
              validation_data=(val_data, val_labels))

    # バリデーションデータ評価
    model.load_weights(ckpt_path)
    val_probs = model.predict(val_data)
    val_preds = predict_labels(val_probs)
    true_cnt = 0
    for val_label, val_pred in zip(val_labels, val_preds):
        if val_label == val_pred:
            true_cnt += 1
    print('val_acc %.2f' % (true_cnt / args.val_num))

    # テストデータ予測
    test_ids, test_data = load_test_data('data/test.csv', med, mean, std)
    test_probs = model.predict(test_data)
    test_preds = predict_labels(test_probs)
    
    result = []
    for id_, pred in zip(test_ids, test_preds):
        result.append('%d,%d' % (id_, pred))
        
    with open('result.csv', 'w') as fout:
        fout.write('PassengerId,Survived\n')
        fout.write('\n'.join(result))
