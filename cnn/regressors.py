from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.activations import elu
from keras import regularizers


class RegressorV2:

    def __init__(self, channels, img_rows, img_cols):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        cf = 'channels_first'
        ishape = (self.channels, self.img_rows, self.img_cols)

        self.model.add(Conv2D(16, kernel_size=(3, 3), input_shape=ishape,  data_format=cf, activation='relu'))
        self.model.add(Conv2D(16, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='linear'))

        return self.model


class RegressorV3:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        cf = 'channels_first'
        ishape = (1, self.img_rows, self.img_cols)

        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=ishape,  data_format=cf, activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), data_format=cf, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format=cf))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='linear'))

        return self.model