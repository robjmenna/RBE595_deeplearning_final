# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:28:49 2020

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GaussianNoise
from numpy import expand_dims



def create_model():
	# create model
    model = Sequential()
    model.add(Dense(1000, input_shape=[50176], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
    return model

model = create_model()
model.summary()