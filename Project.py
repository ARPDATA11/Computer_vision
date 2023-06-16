import cv2


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import keras
import tensorflow as tf


from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_uniform, glorot_uniform

#image augmentation
from imgaug import augmenters as iaa
import imgaug as ia
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train.shape
number_of_categories = len(np.unique(y_train))
number_of_categories

fig, axs = plt.subplots(2,5, figsize = (10,5))
axs = axs.flatten()

for i in range(10):
  axs[i].imshow(X_train[i])
  axs[i].axis('off')
plt.tight_layout()
plt.show()

#labels conversion into one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#picsels normalization
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Image reshape to format (mumber of images, width, height, number of canals)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

def plot_history(history):
  #Plot the Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'r',linewidth=3.0)
  plt.plot(history.history['val_loss'],'b',linewidth=3.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=16)

  #Plot the Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(history.history['accuracy'], 'r', linewidth=3.0)

  plt.plot(history.history['val_accuracy'], 'b',linewidth=3.0)
  plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=16)

  # Classificator preparation using a convolutional neural network model
  # model preparation
  model = Sequential()

  model.add(
      Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())

  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

  model.add(Dense(10, activation='softmax'))

  # model compilation
  model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics='accuracy')


# image augmentation

imagegenerator = ImageDataGenerator(
    zoom_range = 0.6,
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 25,
    width_shift_range = 0.2,
    shear_range = 20,
    brightness_range = [0.6, 0.9]
)

from re import X
# Fitting model to augmented images
imagegenerator.fit(X_train)
history = model.fit(imagegenerator.flow(X_train, y_train, batch_size=128), epochs = 10, validation_data=(X_test, y_test))




