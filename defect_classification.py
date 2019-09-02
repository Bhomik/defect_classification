import os
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras import layers
# from tf.keras import layers
from keras import Model

from keras.utils import np_utils

import pandas as pd


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.switch_backend('TkAgg')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# from keras.optimizers import RMSprop
filepath = '/mnt/disk3/rohit2/bhomik_work/JBM_data/best_model.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
callbacks_list = [checkpoint]

model.compile(loss='binary_crossentropy',
              optimizer = 'RMSprop',
              metrics=['acc'])



# All images will be rescaled by 1./255
# train_datagen = ImageDataGenerator(rescale=1/255)
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/mnt/disk3/rohit2/bhomik_work/JBM_data/train/',  # This is the source directory for training images
        target_size=(300, 300), 
        batch_size=25,
        class_mode='binary')


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        '/mnt/disk3/rohit2/bhomik_work/JBM_data/valid/',  # This is the source directory for training images
        target_size=(300, 300),  
        batch_size=6,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=70,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=10,
      callbacks=callbacks_list)

# model.save("jbm.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_loss', 'Valid_loss'], loc='upper left')

plt.show()
fig.savefig('/mnt/disk3/rohit2/bhomik_work/JBM_data/JBM_loss.png', format = 'png')

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_accuracy', 'Valid_accuracy'], loc='upper left')

plt.show()
fig.savefig('/mnt/disk3/rohit2/bhomik_work/JBM_data/JBM_accuracy.png', format = 'png')

