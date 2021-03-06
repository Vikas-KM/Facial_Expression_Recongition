# -*- coding: utf-8 -*-
"""fer-vgg16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Vikas-KM/Facial_Expression_Recongition/blob/main/fer_vgg16.ipynb
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# When running on colab
! unzip "/content/drive/My Drive/fer.zip" -d "/content/drive/My Drive/fer"

train_dir = '/content/drive/MyDrive/fer/images/images/train/'
validation_dir = '/content/drive/MyDrive/fer/images/images/validation/'

batch_size = 32
img_height = 48
img_width = 48

train_datagen = ImageDataGenerator(
                                    rescale=1/255, 
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest',)

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
                                    directory=train_dir,
                                    shuffle=True,
                                    target_size=(img_height, img_width), 
                                    class_mode='categorical',
                                    color_mode='grayscale',
                                    batch_size=batch_size,)

validation_generator = validation_datagen.flow_from_directory(
                                    directory=validation_dir,
                                    shuffle=True,
                                    target_size=(img_height, img_width), 
                                    class_mode='categorical',
                                    color_mode='grayscale',
                                    batch_size=batch_size,)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(7, activation='softmax')
])

# Checking the Model Summary for Paramters
print(model.summary())

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

history = model.fit(
  train_generator,
  validation_data=validation_generator,
  batch_size=batch_size,
  epochs=100,
  verbose=1,
  steps_per_epoch = 900,
  validation_steps = 220,
)

import matplotlib.pyplot as plt 
%matplotlib inline

# Train Accuracy vs Validation Accuracy Plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend()

# Train Loss vs Validation Loss Plot
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0.5, 1])
plt.legend()

