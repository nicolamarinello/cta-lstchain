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

        self.model.add(Conv2D(16, (3, 3), input_shape=(1, self.img_rows, self.img_cols),
                              data_format='channels_first', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(16, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(32, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, (3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model
