from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation
from keras import layers
from keras import models
from keras import activations
from keras.activations import elu
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

# TODO: batch normalization?


class ClassifierV2:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        self.model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(1, self.img_rows, self.img_cols),  data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(16, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first', activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model


class ClassifierV3:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = Sequential()   # define the network model

    def get_model(self):

        self.model.add(Conv2D(32, (3, 3), input_shape=(1, self.img_rows, self.img_cols), padding='same', data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(128, (3, 3), padding='same', data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Conv2D(256, (3, 3), padding='same', data_format='channels_first'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.20))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.20))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model


class CResNet:

    """
    Clean and simple Keras implementation of network architectures described in:
        - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
        - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

    Python 3.
    """

    #
    # image dimensions
    #

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols

        #
        # network params
        #

        #self.cardinality = 3

    def get_model(self, cardinality=1):

        def residual_network(x):
            """
            ResNeXt by default. For ResNet set `cardinality` = 1 above.

            """

            def add_common_layers(y):
                y = layers.BatchNormalization()(y)
                y = layers.LeakyReLU()(y)

                return y

            def grouped_convolution(y, nb_channels, _strides):
                # when `cardinality` == 1 this is just a standard convolution
                if cardinality == 1:
                    return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, data_format='channels_first', padding='same')(y)
                # print('cardinality:', cardinality)
                # print('nb_channels:', nb_channels)
                assert not nb_channels % cardinality
                _d = nb_channels // cardinality

                # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
                # and convolutions are separately performed within each group
                groups = []
                for j in range(cardinality):
                    group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                    groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, data_format='channels_first',
                                                padding='same')(group))

                # the grouped convolutional layer concatenates them as the outputs of the layer
                y = layers.concatenate(groups)

                return y

            def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
                """
                Our network consists of a stack of residual blocks. These blocks have the same topology,
                and are subject to two simple rules:

                - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
                - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
                """
                shortcut = y

                # we modify the residual building block as a bottleneck design to make the network more economical
                y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), data_format='channels_first',
                                  padding='same')(y)
                y = add_common_layers(y)

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                y = grouped_convolution(y, nb_channels_in, _strides=_strides)
                y = add_common_layers(y)

                y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), data_format='channels_first',
                                  padding='same')(y)
                # batch normalization is employed after aggregating the transformations and before adding to the shortcut
                y = layers.BatchNormalization()(y)

                # identity shortcuts used directly when the input and output are of the same dimensions
                if _project_shortcut or _strides != (1, 1):
                    # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                    # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                    shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides,
                                             data_format='channels_first', padding='same')(shortcut)
                    shortcut = layers.BatchNormalization()(shortcut)

                y = layers.add([shortcut, y])

                # relu is performed right after each batch normalization,
                # expect for the output of the block where relu is performed after the adding to the shortcut
                y = layers.LeakyReLU()(y)

                return y

            # conv1
            x = layers.Conv2D(8, kernel_size=(7, 7), strides=(2, 2), data_format='channels_first', padding='same')(x)
            x = add_common_layers(x)

            # conv2
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first', padding='same')(x)
            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, 16, 32, _project_shortcut=project_shortcut)

            # conv3
            for i in range(4):
                # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 32, 64, _strides=strides)

            # conv4
            for i in range(6):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 64, 128, _strides=strides)

            # conv5
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 128, 256, _strides=strides)

            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1, activation='sigmoid')(x)

            return x

        image_tensor = layers.Input(shape=(1, self.img_rows, self.img_cols))
        network_output = residual_network(image_tensor)

        model = models.Model(inputs=[image_tensor], outputs=[network_output])

        return model