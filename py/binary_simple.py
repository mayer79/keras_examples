# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

# Import data
from keras.datasets import imdb

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words = 10000)

# Data prep
import numpy as np

def pad_sequence(seq, dim = 10000):
    out = np.zeros((len(seq), dim), dtype = 'int32')
    for i, s in enumerate(seq):
        out[i, s] = 1
    return out

train_x = pad_sequence(train_x)
test_x = pad_sequence(test_x)

# Model architecture
from keras import models
from keras import layers

def model_build():
    model = models.Sequential()
    model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model

mod = model_build()

# Compile
from keras import optimizers

opt = optimizers.RMSprop(lr = 0.0001)

mod.compile(optimizer = opt,
            loss = 'binary_crossentropy',
            metrics = ['binary_accuracy'])

# Fit
x_val = train_x[:10000]
y_val = train_y[:10000]
x_train = train_x[10000:]
y_train = train_y[10000:]

# Investigate development of accuracies
history = mod.fit(x_train, y_train, batch_size = 512, epochs = 10, validation_data = (x_val, y_val))
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

perf_plot(history, "loss")
perf_plot(history, "acc")
