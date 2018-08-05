from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model


def _inception_conv_block(input_layer, input_shape=None):
    """
    Method to instantiate an Inception convolutional block.
     - The block will have parallel Conv layers with kernel sizes 1x1, 3x3, 5x5, 7x7 and a MaxPooling layer.
     - The Conv layers with kernel sizes 5x5 and 7x7 will be preceded with a 1x1 Conv kernel layer of
       lower depth (num filters) as compared to the input layer for dimensionality reduction.
     - The MaxPooling layer will be followed by a 1x1 Conv kernel layer for the same reason and for non-linearity.
     - The outputs of all these parallel layers are concatenated and returned.
     - padding has to be 'same' for all layers in an inception block.
    """
    act = 'relu'
    
    if input_shape is not None:
        output1 = Conv2D(24, 1, padding='same', activation=act, input_shape=input_shape)(input_layer)
        output2 = Conv2D(24, 3, padding='same', activation=act, input_shape=input_shape)(input_layer)
        output3 = Conv2D(16, 5, padding='same', activation=act, input_shape=input_shape)(input_layer)
        output4 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', input_shape=input_shape)(input_layer)
    else:
        output1 = Conv2D(24, 1, padding='same', activation=act)(input_layer)
        output2 = Conv2D(24, 3, padding='same', activation=act)(input_layer)
        output3 = Conv2D(16, 5, padding='same', activation=act)(input_layer)
        output4 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(input_layer)

    output4 = Conv2D(8, 1, padding='same', activation=act)(output4)
    output_layer = Concatenate()([output1, output2, output3, output4])

    return output_layer


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    padding = 'same'
    
    image = Input(shape=input_shape)
    
    # first_conv = Conv2D(64, (3, 3), activation='relu', padding=padding)(image)
    # first_conv = _inception_conv_block(image, input_shape)

    inc1 = _inception_conv_block(image, input_shape)
    inc2 = _inception_conv_block(inc1)
    mp1 = MaxPooling2D(pool_size=(2, 2), padding=padding)(inc2)    

    inc3 = _inception_conv_block(mp1)
    inc4 = _inception_conv_block(inc3)
    mp2 = MaxPooling2D(pool_size=(2, 2), padding=padding)(inc4)    

    # inc5 = _inception_conv_block(mp2)
    # inc6 = _inception_conv_block(inc5)
    
    flat_conv = Flatten()(mp2)
    feature = Dense(256, activation='relu')(flat_conv)
    model = Model(inputs=image, outputs=feature)
    print(model.summary())
    return model

