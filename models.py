import glob
import random
import re

import natsort
import numpy as np
import tensorflow as tf
from keras.applications.xception import Xception
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import settings

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

LOAD_PREV_MODEL = settings.LOAD_PREV_MODEL
MODEL_TO_LOAD = settings.MODEL_TO_LOAD
LEARNING_RATE = settings.LEARNING_RATE


def model_xception(num_actions, sample_state):
    base_model = Xception(weights=None, include_top=False, input_shape=sample_state.shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(num_actions, activation="linear")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
    return model


def model_64(num_actions, sample_state):
    model = Sequential()

    model.add(Dense(64, input_shape=sample_state.shape, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))

    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

    return model


def model_256_512(num_actions, sample_state):
    model = Sequential()

    model.add(Conv2D(256, (3, 3),
                     input_shape=sample_state.shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))

    model.add(Dense(num_actions, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
    return model


def model_64x3(num_actions, sample_state):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=sample_state.shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(num_actions, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
    return model


def model_512x3(num_actions, sample_state):
    model = Sequential()

    model.add(Conv2D(512, (3, 3), input_shape=sample_state.shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Flatten())
    model.add(Dense(512))

    model.add(Dense(num_actions, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
    return model


def main_model(num_actions, sample_state):
    return model_512x3(num_actions, sample_state)


def choose_model():
    if MODEL_TO_LOAD:
        return f"models/{MODEL_TO_LOAD}.model"
    elif settings.LOAD_BEST_MODEL:
        path = 'models/'
        files = [f for f in glob.glob(path + "**/*.model", recursive=True)]
        unit_to_find = settings.unit_to_find
        unit_scores = {}
        for file in files:
            unit_score = re.search(f"{unit_to_find}_\d+", file)
            if unit_score:
                unit_score = int(unit_score.group().replace("unit_", ""))
                unit_scores[unit_score] = file
        max_score = max(list(unit_scores.keys()))
        best_model = unit_scores[max_score]
        return best_model
    else:
        path = 'models/'
        files = [f for f in glob.glob(path + "**/*.model", recursive=True)]
        files = natsort.natsorted(files)
        return files[-1]
