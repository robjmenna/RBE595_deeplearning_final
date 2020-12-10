# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:23:43 2020

@author: 13473
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.applications.resnet50 import preprocess_input
from keras.layers import GaussianNoise

# training_ds
# validation_ds

#Define CNN Model
def cnn_model():
    model = Sequential([
      #Standardize Dataset
      layers.experimental.preprocessing.Rescaling((1./255), input_shape=(64, 64, 3)),
      layers.Dropout(0.2),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Conv2D(128, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Conv2D(256, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(32, activation='relu'),
      layers.Dense(8, activation='softmax') ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

model = cnn_model()
model.summary()

# epochs= 5
# history = model.fit(training_ds, validation_data=validation_ds, epochs=epochs, batch_size = 128, shuffle=True)

# model.save("model.h5")
# print("Saved model to disk")