from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


def create_model(image_size=(224, 224), n_outs=None, dropout=False,
                 fc_l2_reg=None, conv_l2_reg=None, **kwargs):
    # this model is followed after the model from deepsenseio:
    # https://deepsense.io/deep-learning-right-whale-recognition-kaggle/

    image_shape = (3, image_size[0], image_size[1])
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(sum(n_outs), activation='softmax', name='predictions'))
    model.name = 'deepsenseio_localizing_whale'

    return model

