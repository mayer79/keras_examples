# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:03:40 2018

@author: Michael
"""

#############
# Import data

import os
path = "C:/projects/foodDetector"
os.chdir(path)

file = "jena_climate_2009_2016.csv"

import pandas as pd
import numpy as np

raw = pd.read_csv(os.path.join(path, file))
raw.drop('Date Time', inplace=True, axis = 1)
raw.head()

float_data = raw.values.astype("float32")

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
train_gen = generator(float_data,
                      lookback=1440,
                      min_index=0,
                      max_index=n_train,
                      shuffle=True)

valid_gen = generator(float_data,
                      lookback=1440,
                      min_index=n_train + 1,
                      max_index=n_train + 1 + n_valid,
                      shuffle=True)

test_gen = generator(float_data,
                     lookback=1440,
                     min_index=n_train + 1 + 1000000)

valid_steps = (n_valid - lookback) // 100
test_steps = len(float_data) - n_train - n_valid

# Define model
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam

model = Sequential()
model.add(CuDNNGRU(32, input_shape = (None, float_data.shape[-1])))
model.add(Dense(1))

model.summary()

model.compile(optimizer=adam(), loss="mse", metrics=["mae"])

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs = 2,
                              validation_data=valid_gen,
                              validation_steps=valid_steps)

