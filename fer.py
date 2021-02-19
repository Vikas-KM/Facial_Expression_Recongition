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

# duckduckgo_search(root, "Angry", "angry face", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face male", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face female", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face boy", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face girl", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face person", max_results=1000)
# duckduckgo_search(root, "Angry", "angry face human", max_results=1000)
#
# duckduckgo_search(root, "Happy", "happy face people", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face male", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face female", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face boy", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face girl", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face person", max_results=1000)
# duckduckgo_search(root, "Happy", "happy face human", max_results=1000)

# duckduckgo_search(root, "Neutral", "neutral face people", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face male", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face female", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face boy", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face girl", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face person", max_results=1000)
# duckduckgo_search(root, "Neutral", "neutral face human", max_results=1000)

# duckduckgo_search(root, "Sad", "sad face people", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face male", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face female", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face boy", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face girl", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face person", max_results=1000)
# duckduckgo_search(root, "Sad", "sad face human", max_results=1000)
#
# duckduckgo_search(root, "Fear", "fear face people", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face male", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face female", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face boy", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face girl", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face person", max_results=1000)
# duckduckgo_search(root, "Fear", "fear face human", max_results=1000)
#
# duckduckgo_search(root, "Disgust", "disgust face people", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face male", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face female", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face boy", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face girl", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face person", max_results=1000)
# duckduckgo_search(root, "Disgust", "disgust face human", max_results=1000)
#
# duckduckgo_search(root, "Surprise", "surprise face people", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face male", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face female", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face boy", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face girl", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face person", max_results=1000)
# duckduckgo_search(root, "Surprise", "surprise face human", max_results=1000)


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
