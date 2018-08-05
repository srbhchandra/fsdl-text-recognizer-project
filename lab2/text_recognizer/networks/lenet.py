from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K

def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: K.expand_dims(x, axis=-1), input_shape=(input_shape[0], input_shape[1])))
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    ##### Your code above (Lab 2)
    return model

