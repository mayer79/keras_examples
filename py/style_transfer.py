# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:45:02 2018

@author: Michael
"""

import os
path = "C:/projects/style_transfer/img"
os.chdir(path)

from keras.preprocessing.image import load_img, img_to_array

# Initial variables required
target_img_path = os.path.join(path, "dragon_whelp_amy_weber.JPG")
style_img_path = os.path.join(path, "picasso.jpg")

# style_img_path = os.path.join(path, "picasso.JPG")
# target_img_path = os.path.join(path, "me.jpg")

width, height = load_img(target_img_path).size
img_height = 400
img_width = int(width / height * img_height)

# Some functions
import numpy as np
from keras.applications import vgg19

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_img(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# Load VVG19
from keras import backend as K

target_img = K.constant(preprocess_img(target_img_path))
style_img = K.constant(preprocess_img(style_img_path))
comb_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_img, style_img, comb_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Content and style loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features))
    
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * size ** 2)

# Total variation loss
def total_var_loss(x):
    a = K.square(
            x[:, :img_height - 1, :img_width - 1, :] - 
            x[:, 1:, :img_width - 1, :])
    
    b = K.square(
            x[:, :img_height - 1, :img_width - 1, :] - 
            x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Definint the final loss to be minized
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
total_variation_weight = 0.0001
style_weight = 1.0
content_weight = 0.025
loss = K.variable(0.0)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
    
loss += total_variation_weight * total_var_loss(comb_image)

# Gradient-descent process
grads = K.gradients(loss, comb_image)[0]
fetch_loss_and_grads = K.function([comb_image], [loss, grads])

class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype("float64")
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
evaluator = Evaluator()

# Style-transfer 
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'picasso_style'
iterations = 20

x = preprocess_img(target_img_path)
x = x.flatten()

for i in range(iterations):
    print("Iteration", i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print("Current loss:", min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_img(img)
    fname = os.path.join("results", "..", result_prefix + "_at_iteration_%d.png" % i)
    imsave(fname, img)
    print("Image saved as", fname)
    end_time = time.time()
    print("Iteration %d completed in %ds" % (i, end_time - start_time))
    
    
    
