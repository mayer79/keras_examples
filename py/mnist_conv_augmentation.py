# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

# Import data
from keras.datasets import mnist

(X, y), (test_x, test_y) = mnist.load_data()

# Split train into train & valid
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state=1, stratify=y)

# Data prep
import numpy as np

def prep_mnist(data):
    data = data.reshape(len(data), 28, 28, 1)
    return data.astype('float32') / 255

train_x = prep_mnist(train_x)
valid_x = prep_mnist(valid_x)
test_x = prep_mnist(test_x)


# Generator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=6, 
                                   zoom_range = 0.12,  
                                   shear_range=0.05,
                                   width_shift_range=0.08,
                                   height_shift_range=0.08) 

valid_datagen = ImageDataGenerator() 

BATCH_SIZE = 64

train_generator = train_datagen.flow(
        train_x, 
        train_y,
        batch_size=BATCH_SIZE,
        shuffle=True)

valid_generator = valid_datagen.flow(
        valid_x, 
        valid_y,
        batch_size=BATCH_SIZE,
        shuffle=True)


# Model architecture
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #, SeparableConv2D

def model_build():
    model = models.Sequential()
    
    # First block
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # 2nd block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Dense block
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

mod = model_build()
mod.summary()

# Compile
mod.compile(optimizer = "adam",
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

# Fit the model
TRAIN_N = train_x.shape[0]
VALID_N = valid_x.shape[0]

history = mod.fit_generator(train_generator,
                            validation_data=valid_generator,
                            steps_per_epoch=TRAIN_N // BATCH_SIZE,
                            validation_steps=VALID_N // BATCH_SIZE,
                            epochs=4)

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
