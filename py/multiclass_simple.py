# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

# Import data
from keras.datasets import reuters

(train_x, train_y), (test_x, test_y) = reuters.load_data(num_words = 10000)

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
    model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(46, activation = 'softmax'))
    return model

mod = model_build()

# Compile
mod.compile(optimizer = "Adam",
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

# Fit
x_val = train_x[:1000]
y_val = train_y[:1000]
x_train = train_x[1000:]
y_train = train_y[1000:]

history = mod.fit(x_train, y_train, batch_size = 512, epochs = 20, validation_data = (x_val, y_val))

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

perf_plot(history, "loss")
perf_plot(history, "acc")

# After 5 epochs on full training
mod.fit(train_x, train_y, batch_size = 512, epochs = 5)

# Accuracy
(test_loss, test_acc) = mod.evaluate(test_x, test_y)
print("Test accuracy is", test_acc)

preds = mod.predict(test_x[0:2,:])
preds.argmax(axis = 1)
