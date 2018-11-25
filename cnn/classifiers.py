from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers


class ClassifierV1:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                              input_shape=(1, self.img_rows, self.img_cols), data_format='channels_first'))
        self.model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1, activation='sigmoid'))

        return self.model


class ClassifierV2:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        weight_decay = 1e-4

        self.model.add(Conv2D(32, kernel_size=5, input_shape=(1, self.img_rows, self.img_cols),
                              data_format='channels_first',
                              kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        self.model.add(Conv2D(32, kernel_size=5, data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(64, kernel_size=3, data_format='channels_first',
                              kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        self.model.add(Conv2D(64, kernel_size=3, data_format='channels_first',
                              kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(128, kernel_size=3, data_format='channels_first',
                              kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(units=1, activation="sigmoid"))

        return self.model


class ClassifierV3:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        self.model.add(Conv2D(64, (3, 3), input_shape=(1, self.img_rows, self.img_cols), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(128, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(256, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model


class ClassifierV4:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        self.model.add(Conv2D(64, kernel_size=3, input_shape=(1, self.img_rows, self.img_cols),
                              data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(64, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(128, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(128, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(256, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(1024, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(1024, kernel_size=3, data_format='channels_first', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(2048, activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1024, activation="relu"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(units=1, activation="sigmoid"))

        return self.model
