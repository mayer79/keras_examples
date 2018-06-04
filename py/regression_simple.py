# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

# Import data
from keras.datasets import boston_housing

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

# Data prep
import numpy as np

m = train_x.mean(axis = 0)
s = train_x.std(axis = 0)

def scale(data, mean, sd):
    return (data - mean) / sd

train_x = scale(train_x, m, s)
test_x = scale(test_x, m, s)

train_y = np.log(train_y)
test_y = np.log(test_y)


# Model architecture
from keras import models
from keras import layers

def model_build():
    model = models.Sequential()
    model.add(layers.Dense(8, activation = 'relu', input_shape = (train_x.shape[1], )))
    model.add(layers.Dense(4, activation = 'relu'))
    model.add(layers.Dense(1))
    return model

mod = model_build()

# Compile
mod.compile(optimizer = "Adam",
            loss = 'mse',
            metrics = ['mae'])

# Fit
x_val = train_x[:100]
y_val = train_y[:100]
x_train = train_x[100:]
y_train = train_y[100:]

history = mod.fit(x_train, y_train, batch_size = 4, epochs = 100, validation_data = (x_val, y_val))

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
perf_plot(history, "mean_absolute_error")

# After 5 epochs on full training
mod.fit(train_x, train_y, batch_size = 4, epochs = 50)

# Accuracy
(test_loss, test_acc) = mod.evaluate(test_x, test_y)
print("Test accuracy is", test_acc)

preds = mod.predict(test_x[0:2,:])
test_y[0:2]
