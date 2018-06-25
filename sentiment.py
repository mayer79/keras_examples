# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:26:51 2018

@author: Michael
"""

import numpy as np

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train.shape # 25'000
x_train[0] # integer coded word sequence

y_train.shape # 25'000
y_train[0]
y_train.mean() # 0.5

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_features, 16, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2)

# Investigate development of accuracies
import matplotlib.pyplot as plt

def perf_plot(history, what = 'loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    
    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None

perf_plot(hist, "loss")
perf_plot(hist, "acc")

## Simple RNN
from keras.layers import SimpleRNN

model = Sequential()
model.add(Embedding(max_features, 32, input_length=maxlen))
model.add(SimpleRNN(units=32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2)

perf_plot(hist, "acc")

## LSTM
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32, input_length=maxlen))
model.add(LSTM(units=32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 epochs=5,
                 batch_size=32,
                 validation_split=0.2)

perf_plot(hist, "acc")


## 1d convnets
from keras.layers import Conv1D, MaxPooling1D

model = Sequential()
model.add(Embedding(max_features, 32, input_length=maxlen))
model.add(Conv1D(32, 7, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=128,
                 validation_split=0.2)

perf_plot(hist, "acc")



## 1d convnets combined with GRU
from keras.layers import GRU

model = Sequential()
model.add(Embedding(max_features, 32, input_length=maxlen))
model.add(Conv1D(32, 7, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation="relu"))
model.add(GRU(32, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=128,
                 validation_split=0.2)

perf_plot(hist, "acc")
