import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from jmd_imagescraper.core import *
from pathlib import Path

root = Path().cwd() / "images"

# Images downloaded and cleaned, so commented below code
# More info on the imagescrapper lib used here: https://pypi.org/project/jmd-imagescraper/

# duckduckgo_search(root, "Angry", "angry face people", max_results=10000)
# duckduckgo_search(root, "Happy", "happy face people", max_results=10000)
# duckduckgo_search(root, "Neutral", "neutral face people", max_results=10000)
# duckduckgo_search(root, "Sad", "sad face people", max_results=10000)
# duckduckgo_search(root, "Fear", "fear face people", max_results=10000)

# Steps
# Data needs to be split into train and test folders
# Data needs to be augmented - ImageDataGenerator
# Data needs to be rescaled / Normalized


training_dir = './images/train'
test_dir = './images/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(48, 48),
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),
    #
    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(7, activation='softmax')
])

print(model.summary())

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    verbose=1
)
