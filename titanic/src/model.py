# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers


class TitanicModel(tf.keras.Model):
    def __init__(self, hidden_ch=16, is_dropout=False):
        super(TitanicModel, self).__init__()
        self.dense1 = layers.Dense(hidden_ch, activation='relu')
        self.dense2 = layers.Dense(hidden_ch, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(1, activation='sigmoid')

        self.is_dropout = is_dropout
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        if self.is_dropout:
            x = self.dropout(x)
        x = self.out(x)
        return x
