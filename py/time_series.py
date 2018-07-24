# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:03:40 2018

@author: Michael
"""

#############
# Import data (fetch first from https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip)

import os
path = "C:/projects/foodDetector"
os.chdir(path)

file = "jena_climate_2009_2016.csv"

import pandas as pd
import numpy as np

float_data = pd.read_csv(os.path.join(path, file), dtype="float32", usecols=lambda x: x not in ['Date Time']).values
float_data[0]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
n_train = 200000
n_valid = 100000
sc.fit(float_data[:n_train])

float_data = sc.transform(float_data)
float_data[0]

def generator(data, lookback=1440, delay=144, min_index=0, max_index=None, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        
        for j, row in enumerate(rows):
            indices = range(row - lookback, row, step)
            samples[j] = data[indices]
            targets[j] = data[row + delay][1]
        yield samples, targets
        
# Initialize generators
lookback =  1440
batch_size = 128

train_gen = generator(float_data,
                      lookback=1440,
                      min_index=0,
                      max_index=n_train,
                      shuffle=True,
                      batch_size=128)

valid_gen = generator(float_data,
                      lookback=1440,
                      min_index=n_train + 1,
                      max_index=n_train + 1 + n_valid,
                      shuffle=True,
                      batch_size=128)

test_gen = generator(float_data,
                     lookback=1440,
                     min_index=n_train + n_valid + 1,
                     batch_size=128)

# Define model
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam

model = Sequential()
model.add(CuDNNGRU(32, input_shape = (None, float_data.shape[-1])))
model.add(Dense(1))

model.summary()

model.compile(optimizer=adam(lr=0.001), loss="mse", metrics=["mae"])

history = model.fit_generator(train_gen,
                              steps_per_epoch=n_train // batch_size,
                              epochs = 2,
                              validation_data=valid_gen,
                              validation_steps=n_valid // batch_size)

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

perf_plot(history, "mean_absolute_error")
