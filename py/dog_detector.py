# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:09:33 2018

@author: Michael
"""

import os
from os.path import join

# Set working dir
os.chdir("C:/projects/dogDetector")

# data dir
datadir = "data"

import numpy as np
import pandas as pd

#==========================================================================================
# Prepare response
#==========================================================================================

# import train labels data set with column id (image name) and breed (response)
labels = pd.read_csv(join(datadir, "labels.csv"))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(labels.breed.values)


#==========================================================================================
# Prepare images
#==========================================================================================

# Prepare train and valid data
from keras.preprocessing import image

def preprocess_from_file(fp, target_size=(299, 299)):
    try:
        im = image.load_img(fp, target_size=target_size, interpolation='bilinear')
        ar = image.img_to_array(im)  
    except:
        print("Error in file", fp)
    
    return np.array(ar, dtype="uint8")

# Prepare each image in a loop
X = list()
for f in labels.id.values:
    fp = join(datadir, "train", f) + ".jpg"
    X.append(preprocess_from_file(fp))  
    print(".", end='')
    
# Combine to one big array
X = np.array(X)

# Split into train and validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y)

del X


#==========================================================================================
# Load pretrained net and initialize batch generator
#==========================================================================================

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator


# Generator
train_datagen = ImageDataGenerator(rotation_range=6, 
                                   zoom_range = 0.12,  
                                   shear_range=0.05,
                                   width_shift_range=0.08,
                                   height_shift_range=0.08,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_input) 

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

BATCH_SIZE = 32

train_generator = train_datagen.flow(
        X_train, 
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True)

valid_generator = valid_datagen.flow(
        X_valid, 
        y_valid,
        batch_size=BATCH_SIZE,
        shuffle=True)


############## PRETRAINED
from keras.models import Model
from keras import layers

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = layers.Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = layers.Dense(le.classes_.shape[0], activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


TRAIN_N = X_train.shape[0]
VALID_N = X_valid.shape[0]
THREADS = 4

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    steps_per_epoch=TRAIN_N // BATCH_SIZE,
                    epochs=3,
                    nb_worker=THREADS)


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use adam with a low learning rate
from keras.optimizers import rmsprop
model.compile(optimizer=rmsprop(lr=0.00001),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1)
chkp = ModelCheckpoint(join("model", "fine_tuned.hdf5"), save_best_only=True, verbose=1)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

# train the model on the new data for a few epochs
history = model.fit_generator(train_generator,
                              steps_per_epoch=TRAIN_N // BATCH_SIZE,
                              epochs=50,
                              nb_worker=THREADS,
                              validation_data=valid_generator,
                              validation_steps=VALID_N // BATCH_SIZE,
                              callbacks=[early_stop, chkp])


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


# Read all test images
# import test labels
files_test = pd.read_csv(join(datadir, "sample_submission.csv"), usecols=['id']).id.values

y_test = list()
for f in files_test: # f = files_test[0]
    fp = join(datadir, "test", f) + ".jpg"
    x = preprocess_input(np.expand_dims(preprocess_from_file(fp), axis=0))
    y_test.append(model.predict(x))
    print(".", end="")

Y_test = np.array(y_test)

out = pd.DataFrame(np.squeeze(np.array(y_test)),
                   columns=le.inverse_transform(np.arange(0, le.classes_.shape[0])))

out.insert(loc=0, column='id', value=files_test)
out.to_csv("sample_submission.csv", index=False)
