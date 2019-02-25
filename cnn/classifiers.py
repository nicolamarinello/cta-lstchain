from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, Input
from keras import layers
from keras import models
import keras
from keras.models import Model
from keras.regularizers import l2
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


class ResNetXt:

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
            x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', data_format='channels_first')(x)
            x = add_common_layers(x)

            # conv2
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(x)
            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, 64, 128, _project_shortcut=project_shortcut)

            # conv3
            for i in range(4):
                # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 128, 256, _strides=strides)

            # conv4
            for i in range(6):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 256, 512, _strides=strides)

            # conv5
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 512, 1024, _strides=strides)

            x = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
            x = layers.Dense(1, activation='sigmoid')(x)

            return x

        image_tensor = layers.Input(shape=(1, self.img_rows, self.img_cols))
        network_output = residual_network(image_tensor)

        model = models.Model(inputs=[image_tensor], outputs=[network_output])

        return model


class ResNet:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self, version, n):

        def get_resnetv1(input_shape, depth):

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
                              kernel_regularizer=l2(1e-4),
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

            def resnet_v1(input_shape, depth):
                """ResNet Version 1 Model builder [a]
                Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
                Last ReLU is after the shortcut connection.
                At the beginning of each stage, the feature map size is halved (downsampled)
                by a convolutional layer with strides=2, while the number of filters is
                doubled. Within each stage, the layers have the same number filters and the
                same number of filters.
                Features maps sizes:
                stage 0: 32x32, 16
                stage 1: 16x16, 32
                stage 2:  8x8,  64
                The Number of parameters is approx the same as Table 6 of [a]:
                ResNet20 0.27M
                ResNet32 0.46M
                ResNet44 0.66M
                ResNet56 0.85M
                ResNet110 1.7M
                # Arguments
                    input_shape (tensor): shape of input image tensor
                    depth (int): number of core convolutional layers
                    num_classes (int): number of classes (CIFAR10 has 10)
                # Returns
                    model (Model): Keras model instance
                """
                if (depth - 2) % 6 != 0:
                    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
                # Start model definition.
                num_filters = 16
                num_res_blocks = int((depth - 2) / 6)

                inputs = Input(shape=input_shape)
                x = resnet_layer(inputs=inputs)
                # Instantiate the stack of residual units
                for stack in range(3):
                    for res_block in range(num_res_blocks):
                        strides = 1
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            strides = 2  # downsample
                        y = resnet_layer(inputs=x,
                                         num_filters=num_filters,
                                         strides=strides)
                        y = resnet_layer(inputs=y,
                                         num_filters=num_filters,
                                         activation=None)
                        if stack > 0 and res_block == 0:  # first layer but not first stack
                            # linear projection residual shortcut connection to match
                            # changed dims
                            x = resnet_layer(inputs=x,
                                             num_filters=num_filters,
                                             kernel_size=1,
                                             strides=strides,
                                             activation=None,
                                             batch_normalization=False)
                        x = keras.layers.add([x, y])
                        x = Activation('relu')(x)
                    num_filters *= 2

                # Add classifier on top.
                # v1 does not use BN after last shortcut connection-ReLU
                x = AveragePooling2D(pool_size=8, data_format='channels_first')(x)
                y = Flatten()(x)
                outputs = Dense(1,
                                activation='sigmoid',
                                kernel_initializer='he_normal')(y)

                # Instantiate model.
                model = Model(inputs=inputs, outputs=outputs)
                return model

            return resnet_v1(input_shape, depth)

        # Model version
        # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

        # Computed depth from supplied model parameter n
        if version == 1:
            depth = n * 6 + 2

            # Model name, depth and version
            model_type = 'ResNet%dv%d' % (depth, version)

            print(model_type)

            model = get_resnetv1((1, self.img_rows, self.img_cols), depth)

        # elif version == 2:
        #    depth = n * 9 + 2

        return model


class ResNetA:

    def __init__(self, img_rows, img_cols):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_model(self):

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
                          kernel_regularizer=l2(1e-4),
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

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=16, strides=1)  # output (16, 100, 100)

        # stack 0
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
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=8, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetB:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (8, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetC:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        
        batch_size 128
        adam
        
        Total params: 23,161
        Trainable params: 22,261
        Non-trainable params: 900
        __________________________________________________________________________________________________
        Epoch 1/75
        6564/6564 [==============================] - 1514s 231ms/step - loss: 0.5753 - acc: 0.6771 - val_loss: 0.5251 - val_acc: 0.7201
        Epoch 2/75
        6564/6564 [==============================] - 1487s 226ms/step - loss: 0.5078 - acc: 0.7370 - val_loss: 0.5305 - val_acc: 0.7180
        Epoch 3/75
        6564/6564 [==============================] - 1494s 228ms/step - loss: 0.4926 - acc: 0.7469 - val_loss: 0.5266 - val_acc: 0.7200
        Epoch 4/75
        6564/6564 [==============================] - 1407s 214ms/step - loss: 0.4843 - acc: 0.7525 - val_loss: 0.5110 - val_acc: 0.7357
        Epoch 5/75
        6564/6564 [==============================] - 1475s 225ms/step - loss: 0.4782 - acc: 0.7558 - val_loss: 0.5173 - val_acc: 0.7213
        Epoch 6/75
        6564/6564 [==============================] - 1503s 229ms/step - loss: 0.4739 - acc: 0.7585 - val_loss: 0.4838 - val_acc: 0.7515
        Epoch 7/75
        6564/6564 [==============================] - 1484s 226ms/step - loss: 0.4702 - acc: 0.7609 - val_loss: 0.5082 - val_acc: 0.7339
        Epoch 8/75
        6564/6564 [==============================] - 1484s 226ms/step - loss: 0.4672 - acc: 0.7624 - val_loss: 0.4913 - val_acc: 0.7451
        Epoch 9/75
        6564/6564 [==============================] - 1493s 227ms/step - loss: 0.4647 - acc: 0.7643 - val_loss: 0.5306 - val_acc: 0.7184
        Epoch 10/75
        6564/6564 [==============================] - 1496s 228ms/step - loss: 0.4628 - acc: 0.7653 - val_loss: 0.4680 - val_acc: 0.7612
        Epoch 11/75
        6564/6564 [==============================] - 1496s 228ms/step - loss: 0.4614 - acc: 0.7653 - val_loss: 0.4946 - val_acc: 0.7380
        Epoch 12/75
        6564/6564 [==============================] - 1501s 229ms/step - loss: 0.4597 - acc: 0.7668 - val_loss: 0.4843 - val_acc: 0.7482
        Epoch 13/75
        6564/6564 [==============================] - 1500s 229ms/step - loss: 0.4580 - acc: 0.7674 - val_loss: 0.5041 - val_acc: 0.7302
        Epoch 14/75
        4774/6564 [====================>.........] - ETA: 6:21 - loss: 0.4570 - acc: 0.7684
        
        """

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
        x = resnet_layer(inputs=y, num_filters=32, strides=2)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=32, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetD:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        batch size 128
        adam
        overfits
        
        Total params: 181,057
        Trainable params: 179,457
        Non-trainable params: 1,600
        __________________________________________________________________________________________________
        Epoch 1/75
        6607/6607 [==============================] - 1571s 238ms/step - loss: 0.5509 - acc: 0.6958 - val_loss: 0.5315 - val_acc: 0.7138
        Epoch 2/75
        6607/6607 [==============================] - 1559s 236ms/step - loss: 0.4804 - acc: 0.7533 - val_loss: 0.5282 - val_acc: 0.7161
        Epoch 3/75
        6607/6607 [==============================] - 1598s 242ms/step - loss: 0.4653 - acc: 0.7621 - val_loss: 0.4703 - val_acc: 0.7608
        Epoch 4/75
        6607/6607 [==============================] - 1584s 240ms/step - loss: 0.4554 - acc: 0.7675 - val_loss: 0.4998 - val_acc: 0.7352
        Epoch 5/75
        6607/6607 [==============================] - 1587s 240ms/step - loss: 0.4482 - acc: 0.7720 - val_loss: 0.4786 - val_acc: 0.7558
        Epoch 6/75
        6607/6607 [==============================] - 1572s 238ms/step - loss: 0.4433 - acc: 0.7754 - val_loss: 0.4537 - val_acc: 0.7678
        Epoch 7/75
        6607/6607 [==============================] - 1628s 246ms/step - loss: 0.4384 - acc: 0.7776 - val_loss: 0.4533 - val_acc: 0.7710
        Epoch 8/75
        6607/6607 [==============================] - 1629s 247ms/step - loss: 0.4347 - acc: 0.7796 - val_loss: 0.4749 - val_acc: 0.7549
        Epoch 9/75
        6607/6607 [==============================] - 1593s 241ms/step - loss: 0.4308 - acc: 0.7820 - val_loss: 0.4525 - val_acc: 0.7712
        Epoch 10/75
        6607/6607 [==============================] - 1579s 239ms/step - loss: 0.4274 - acc: 0.7838 - val_loss: 0.4478 - val_acc: 0.7739
        Epoch 11/75
        6607/6607 [==============================] - 1564s 237ms/step - loss: 0.4242 - acc: 0.7856 - val_loss: 0.4614 - val_acc: 0.7653
        Epoch 12/75
        6607/6607 [==============================] - 1578s 239ms/step - loss: 0.4208 - acc: 0.7871 - val_loss: 0.4569 - val_acc: 0.7670
        Epoch 13/75
        6607/6607 [==============================] - 1569s 237ms/step - loss: 0.4176 - acc: 0.7890 - val_loss: 0.4648 - val_acc: 0.7623
        Epoch 14/75
        6607/6607 [==============================] - 1583s 240ms/step - loss: 0.4142 - acc: 0.7911 - val_loss: 0.4524 - val_acc: 0.7685
        Epoch 15/75
        6607/6607 [==============================] - 1599s 242ms/step - loss: 0.4109 - acc: 0.7926 - val_loss: 0.4541 - val_acc: 0.7684
        Epoch 16/75
        6607/6607 [==============================] - 1594s 241ms/step - loss: 0.4077 - acc: 0.7940 - val_loss: 0.4498 - val_acc: 0.7719
        Epoch 17/75
        6607/6607 [==============================] - 1581s 239ms/step - loss: 0.4043 - acc: 0.7961 - val_loss: 0.4516 - val_acc: 0.7714
        Epoch 18/75
        6607/6607 [==============================] - 1613s 244ms/step - loss: 0.4013 - acc: 0.7980 - val_loss: 0.4518 - val_acc: 0.7707
        Epoch 19/75
        6607/6607 [==============================] - 1591s 241ms/step - loss: 0.3982 - acc: 0.7996 - val_loss: 0.4596 - val_acc: 0.7669
        Epoch 20/75
        6607/6607 [==============================] - 1557s 236ms/step - loss: 0.3947 - acc: 0.8016 - val_loss: 0.4650 - val_acc: 0.7637
        Epoch 21/75
        6607/6607 [==============================] - 1579s 239ms/step - loss: 0.3914 - acc: 0.8032 - val_loss: 0.4634 - val_acc: 0.7651
        Epoch 22/75
        6607/6607 [==============================] - 1593s 241ms/step - loss: 0.3881 - acc: 0.8051 - val_loss: 0.4672 - val_acc: 0.7634
        Epoch 23/75
        6607/6607 [==============================] - 1596s 242ms/step - loss: 0.3847 - acc: 0.8071 - val_loss: 0.4686 - val_acc: 0.7628
        Epoch 24/75
        6607/6607 [==============================] - 1573s 238ms/step - loss: 0.3820 - acc: 0.8085 - val_loss: 0.4717 - val_acc: 0.7628
        Epoch 25/75
        6607/6607 [==============================] - 1575s 238ms/step - loss: 0.3788 - acc: 0.8104 - val_loss: 0.4795 - val_acc: 0.7545
        Epoch 26/75
        6607/6607 [==============================] - 1581s 239ms/step - loss: 0.3754 - acc: 0.8122 - val_loss: 0.4718 - val_acc: 0.7619
        Epoch 27/75
        2085/6607 [========>.....................] - ETA: 15:18 - loss: 0.3672 - acc: 0.8173
        
        

        """

        input_shape = (1, self.img_rows, self.img_cols)

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
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(x)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetE:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        Total params: 179,441
        Trainable params: 177,737
        Non-trainable params: 1,704
        """

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 1
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=16, strides=1)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 2
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

        # stack 3
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

        x = AveragePooling2D(pool_size=3, data_format='channels_first')(y)
        y = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        # outputs = Dense(1, activation='sigmoid')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetF:

    def __init__(self, channels, img_rows, img_cols, wd):

        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        y = resnet_layer(inputs=y, num_filters=128, kernel_size=1, strides=2, activation=None, batch_normalization=False)
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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetG:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        Total params: 1,783,865
        Trainable params: 1,781,005
        Non-trainable params: 2,860    
        """

        input_shape = (1, self.img_rows, self.img_cols)

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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model


class ResNetH:

    def __init__(self, img_rows, img_cols, wd):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.wd = wd

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
        Total params: 3,153,849
        Trainable params: 3,150,485
        Non-trainable params: 3,364  
        """

        input_shape = (1, self.img_rows, self.img_cols)

        inputs = Input(shape=input_shape)  # output (1, 100, 100)
        y = resnet_layer(inputs=inputs, num_filters=8, strides=1)  # output (16, 100, 100)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        x = resnet_layer(inputs=y, num_filters=8, strides=1)
        x = resnet_layer(inputs=x, num_filters=8, strides=1, activation=None)
        x = keras.layers.add([x, y])
        y = Activation('relu')(x)

        # stack 0
        x = resnet_layer(inputs=y, num_filters=16, strides=2)
        x = resnet_layer(inputs=x, num_filters=16, strides=1, activation=None)
        # linear projection
        y = resnet_layer(inputs=y, num_filters=16, kernel_size=1, strides=2, activation=None, batch_normalization=False)
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

        x = resnet_layer(inputs=y, num_filters=32, strides=1)
        x = resnet_layer(inputs=x, num_filters=32, strides=1, activation=None)
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

        x = resnet_layer(inputs=y, num_filters=64, strides=1)
        x = resnet_layer(inputs=x, num_filters=64, strides=1, activation=None)
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

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
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

        x = resnet_layer(inputs=y, num_filters=128, strides=1)
        x = resnet_layer(inputs=x, num_filters=128, strides=1, activation=None)
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
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(y)
        model = Model(inputs=inputs, outputs=outputs)

        return model
