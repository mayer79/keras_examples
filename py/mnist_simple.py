# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:45:48 2018

@author: Michael
"""

import numpy as np

# Load data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Look at one image
from keras.preprocessing import image
image.array_to_img(np.expand_dims(x_train[1,:,:], -1))


# Prepare data
def prep_x(x):
    return x.reshape((len(x), 28 * 28)).astype('float32') / 255

from keras.utils import to_categorical

def prep_y(y):
    return to_categorical(y)
    
x_train = prep_x(x_train)
x_test = prep_x(x_test)

y_train = prep_y(y_train)
y_test = prep_y(y_test)


# mymodel architecture
from keras import models
from keras import layers

mymodel = models.Sequential()
mymodel.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28, )))
mymodel.add(layers.Dense(10, activation = 'softmax'))

# Compile
mymodel.compile(optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])

# Fit
history = mymodel.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_split=0.05)

# Assess overfit
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


# Evaluate
test_loss, test_metric = mymodel.evaluate(x_test, y_test)
print('Accuracy:', test_metric)

