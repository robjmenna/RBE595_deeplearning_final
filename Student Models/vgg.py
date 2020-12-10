# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:41:09 2020

@author: 13473
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 

from numpy import expand_dims
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GaussianNoise


#Define CNN Model
def vgg_model():
    model = Sequential([
      #Standardize Dataset
      # layers.experimental.preprocessing.Rescaling((1./255), input_shape=(64, 64, 3)),
      #Add a layer of Gaussian Noise
      layers.Conv2D(64, 1, input_shape=[224,224,3], activation='relu'),
      layers.Conv2D(64, 1, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(128, 1, activation='relu'),
      layers.Conv2D(128, 1, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(256, 1, activation='relu'),
      layers.Conv2D(256, 1, activation='relu'),
      layers.Conv2D(256, 1, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(512, 1, activation='relu'),
      layers.Conv2D(512, 1, activation='relu'),
      layers.Conv2D(512, 1, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1000) ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

model = vgg_model()
model.summary()

# epochs= 5
# history = model.fit(training_ds, validation_data=validation_ds, epochs=epochs, batch_size = 128, shuffle=True)

# model.save("model.h5")
# print("Saved model to disk")