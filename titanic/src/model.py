# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers


class TitanicModel(tf.keras.Model):
    def __init__(self, hidden_ch=8):
        super(TitanicModel, self).__init__()
        self.dense1 = layers.Dense(hidden_ch, activation='relu')
        # self.dense2 = layers.Dense(hidden_ch, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        # x = self.dense2(x)
        x = self.out(x)
        return x
