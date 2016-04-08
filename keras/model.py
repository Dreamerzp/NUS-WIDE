from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


def alexnet_model():
    """
    AlexNet model implementation in Keras
    :return:
    """

    model = Sequential()
    #conv1
    model.add(Convolution2D(48, 11, 11, border_mode='same', subsample=(4, 4),
                            input_shape=(224, 224, 3), dim_ordering='tf'))
    model.add(Activation('relu'))
    #pool1
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid'))

    #conv2
    model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid'))

    '''
    #conv3
    model.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    model.add(Activation('relu'))


    #conv4
    model.add(Convolution2D(192, 3, 3, border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    model.add(Activation('relu'))

    #conv5
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    model.add(Activation('relu'))
    '''

    #pool2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(4096, init='normal'))
    model.add(Dropout(0.5))
    '''
    model.add(Activation('relu'))
    model.add(Dense(4096, init='normal'))
    model.add(Dropout(0.5))
    '''
    model.add(Activation('relu'))
    model.add(Dense(81, init='normal'))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam())

    return model
