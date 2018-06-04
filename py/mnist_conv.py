# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

# Import data
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()


# Data prep
import numpy as np

def prep_mnist(data):
    data = data.reshape(len(data), 28, 28, 1)
    return data.astype('float32') / 255

train_x = prep_mnist(train_x)
test_x = prep_mnist(test_x)

# Look at one single image
from keras.preprocessing import image
image.array_to_img(train_x[1,:,:,:])

# Model architecture
from keras import models
from keras import layers

def model_build():
    model = models.Sequential()
    
    # First conv
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 2nd conv
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 3nd conv
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Out
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

mod = model_build()
mod.summary()

# Compile
mod.compile(optimizer = "Adam",
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

history = mod.fit(train_x, train_y, batch_size = 64, epochs = 10, validation_split = 0.1)

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

# Accuracy
(test_loss, test_acc) = mod.evaluate(test_x, test_y)
print("Test accuracy is", test_acc)

preds = mod.predict(test_x[0:2,:])
test_y[0:2]
