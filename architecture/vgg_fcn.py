import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.utils.data_utils import get_file


def get_model_vgg_fcn(n_out=2, top_model_weights_path='../exp_dir/fish_localise/training/bottleneck_fc_model.h5'):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, None, None)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1'))
    # Block 1
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # load the model weights
    weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', '', cache_subdir='models')
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    model.load_weights(weights_path)
    print('Model loaded...............')

    # build a classifier model to put on top of the convolutional model
    old_top_model = Sequential()
    old_top_model.add(Flatten(input_shape=(512, 5, 5)))
    old_top_model.add(Dense(n_out, activation='softmax'))
    old_top_model.load_weights(top_model_weights_path)
    dense_weights = old_top_model.layers[-1].weights[0].get_value()
    print(dense_weights.shape)
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Convolution2D(2, 5, 5, activation='relu', border_mode='valid', input_shape=(512, None, None)))
    #top_model.add(Activation(activation='sigmoid'))
    new_conv_weights = dense_weights.transpose(1,0).reshape((2, 512, 5, 5))[:,:,::-1,::-1]
    # set the value--> from dense to convolutional
    # according to http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    top_model.layers[0].weights[0].set_value(new_conv_weights)
    top_model.layers[0].weights[1].set_value(old_top_model.layers[-1].weights[1].get_value())

    print('TOP convolutional parameters loaded...............')
    # add the model on top of the convolutional base
    model.add(top_model)

    return model


def get_model_vgg_fully_connect(image_size=(180, 180), n_out=2,
                      top_model_weights_path='../exp_dir/fish_localise/training/bottleneck_fc_model.h5'):
        model = Sequential()

        model.add(ZeroPadding2D((1, 1), input_shape=(3, image_size[0], image_size[1])))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1'))
        # Block 1
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # load the model weights
        weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', '', cache_subdir='models')
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        model.load_weights(weights_path)
        print('Model loaded...............')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n_out, activation='softmax'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(top_model_weights_path)

        # add the model on top of the convolutional base
        model.add(top_model)
        print('TOP Model loaded...............')

        return model

def softmax(x):
    import numpy as np
    """Compute softmax values for each sets of scores in x.
    x is a 4 dim tensor"""
    for i in range(x.shape[0]):
        for m in range(x.shape[2]):
            for n in range(x.shape[3]):
                x1 = x[i, 0, m, n]
                x2 = x[i, 1, m, n]
                ex1 = np.exp(x1 - np.max([x1, x2]))
                ex2 = np.exp(x2 - np.max([x1, x2]))
                x[i, 0, m, n] = ex1 / (ex1 + ex2)
                x[i, 1, m, n] = ex2 / (ex1 + ex2)

    return x
