import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model


class MiniGoogleNet:

    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding='same'):
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)

        return x

    @staticmethod
    def incept_module(x, numK1x1, numK3x3, chanDim):
        conv1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv3x3 = MiniGoogleNet.conv_module(x, numK3x3, 1, 1, (1, 1), chanDim)
        x = concatenate([conv1x1, conv3x3], axis=chanDim)

        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        conv3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv3x3, pool], axis=chanDim)

        return x

    @staticmethod
    def build(height, width, depth, num_classes=None):
        if tf.keras.backend.image_data_format() == "channels_last":
            inputShape = (height, width, depth)
            chanDim = -1
        else:
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)

        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        x = MiniGoogleNet.incept_module(x, 32, 32, chanDim)
        x = MiniGoogleNet.incept_module(x, 32, 48, chanDim)
        x = MiniGoogleNet.downsample_module(x, 60, chanDim)

        x = MiniGoogleNet.incept_module(x, 112, 48, chanDim)
        x = MiniGoogleNet.incept_module(x, 96, 64, chanDim)
        x = MiniGoogleNet.incept_module(x, 80, 80, chanDim)
        x = MiniGoogleNet.incept_module(x, 48, 96, chanDim)
        x = MiniGoogleNet.downsample_module(x, 96, chanDim)

        x = MiniGoogleNet.incept_module(x, 176, 160, chanDim)
        x = MiniGoogleNet.incept_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="minigooglenet")

        return model