import keras
from keras.layers import Dropout, Flatten, Dense, Conv2D, AveragePooling2D, BatchNormalization, Activation, Input
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2


class RegressorV2:

    def __init__(self, channels, img_rows, img_cols):
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()  # define the network model

    def get_model(self):
        cf = 'channels_first'
        ishape = (self.channels, self.img_rows, self.img_cols)

        self.model.add(Conv2D(16, kernel_size=(3, 3), input_shape=ishape, data_format=cf, activation='relu'))
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
        self.model = Sequential()  # define the network model

    def get_model(self):
        cf = 'channels_first'
        ishape = (1, self.img_rows, self.img_cols)

        self.model.add(Conv2D(32, kernel_size=(3, 3), input_shape=ishape, data_format=cf, activation='relu'))
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


class ResNetF:

    def __init__(self, outcomes, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd
        self.outcomes = outcomes

    def get_model(self):

        wd = self.wd

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(wd),
                          data_format="channels_first")

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        """
        Total params: 1,100,369
        Trainable params: 1,097,913
        Non-trainable params: 2,456     
        """

        input_shape = (self.channels, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=64, strides=2)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=64, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=128, strides=2)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None,
                         batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = AveragePooling2D(pool_size=2, data_format='channels_first')(y)
        y = Flatten()(x)
        outputs = Dense(self.outcomes, activation='linear', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model
