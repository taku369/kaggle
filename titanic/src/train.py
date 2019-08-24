# -*- coding: utf-8 -*-

import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from model import TitanicModel
from utils import load_train_data, scheduler


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--hidden_ch', default=6, type=int)
    parser.add_argument('--dropout', default=False, action='store_true')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--alldata', default=False, action='store_true')
    parser.add_argument('--small', default=False, action='store_true')
    parser.add_argument('--fix_lr', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    train_ids, train_data, train_labels, med = load_train_data('data/train.csv', small=args.small)
    print(train_data.shape)

    if args.normalize:
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        train_data = (train_data - train_mean) / train_std

    if not args.alldata:
        val_data = train_data[-100:]
        val_labels = train_labels[-100:]
        train_data = train_data[:-100]
        train_labels = train_labels[:-100]

    model = TitanicModel(hidden_ch=args.hidden_ch, is_dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=tf.losses.log_loss,
                  metrics=['accuracy'])
    
    ckpt_path = './ckpt/%s/ckpt' % args.name
    logdir="./logs/titanic-%s-%s" % (args.name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if args.alldata: 
        callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path)]
        if not args.fix_lr:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler(int(args.max_epoch * 0.9), args.lr, args.gamma)))
        model.fit(train_data, train_labels, 
                  batch_size=4, epochs=args.max_epoch, callbacks=callbacks)
    else:
        callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss'),
                     tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)]
        if not args.fix_lr:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler(int(args.max_epoch * 0.9), args.lr, args.gamma)))
        model.fit(train_data, train_labels, 
                  batch_size=4, epochs=args.max_epoch, callbacks=callbacks,
                  validation_data=(val_data, val_labels))

    model.load_weights(ckpt_path)
    cnt = 0
    loss_sum = 0
    for IDX in range(100):
        predict = model.predict(val_data[IDX:IDX + 1])[0][0]
        if predict > 0.5:
            pred_label = 1
        else:
            pred_label = 0
        # print(pred_label, val_labels[IDX], pred_label == val_labels[IDX])
        if pred_label == val_labels[IDX]:
            cnt += 1
    print('val_acc %d / 100' % cnt)
