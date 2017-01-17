from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers import Convolution2D, Flatten, Dense, AveragePooling2D
from keras.models import load_model, Model


from keras import backend as K
from keras.engine.topology import Layer


class MyScaleLayer(Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(MyScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True
        #super(MyScaleLayer, self).build()  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return K.dot(x, self.scale)


def get_model_resent50_fcn_retrain(final_conv_area=2, n_out=2, image_size = (200, 200), average_pool=3,
        weight_path='/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise/training/fine_tune_all_conv_resnet50.h5'):
    resn50 = ResNet50(include_top=False, weights='imagenet',  input_shape=(3, image_size[0], image_size[1]))
    base_model = Model(input=resn50.input, output=resn50.layers[-2].output)

    print('RESNET50 Model loaded...............')
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=(2048, 2, 2)))
    top_model.add(Dense(n_out, activation='softmax'))
    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning

    # add the model on top of the convolutional base
    old_model = Sequential()
    old_model.add(base_model)
    old_model.add(AveragePooling2D((3, 3), name='avg_pool', input_shape=base_model.output_shape[1:]))
    old_model.add(top_model)
    old_model.load_weights(weight_path)
    print('Retrained Model loaded...............')

    dense_weights = old_model.layers[-1].weights[0].get_value()
    # new_conv_weights = dense_weights.transpose(1,0).reshape(new_conv_shape)[:,:,::-1,::-1]
    new_conv_weights = dense_weights.transpose(1, 0).reshape((2, 2048, final_conv_area, final_conv_area))[:, :, ::-1,
                       ::-1]
    # set the value--> from dense to convolutional
    # according to http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    # add the model on top of the convolutional base
    top_model = Sequential()
    top_model.add(Convolution2D(2, final_conv_area, final_conv_area, activation='relu', border_mode='same', input_shape=(2048, None, None)))

    top_model.layers[0].weights[0].set_value(new_conv_weights)
    top_model.layers[0].weights[1].set_value(old_model.layers[-1].weights[1].get_value())

    print('TOP convolutional parameters loaded...............')
    model = Sequential()
    resn50 = ResNet50(include_top=False, weights='imagenet', input_shape=(3, None, None))
    base_model = Model(input=resn50.input, output=resn50.layers[-2].output)
    model.add(base_model)
    if average_pool:
        model.add(
            AveragePooling2D((average_pool, average_pool), name='avg_pool', input_shape=base_model.output_shape[1:]))
    # else:
    #     # because originally we define the network as a average pooling
    #     model.add(MyScaleLayer(1/3.))  ## THIS METHOD WILL NOT WORK!!!!!!!!!!!!!!!!!!!
    model.add(top_model)
    print('FINAL Model loaded...............')

    return model


def get_model_resnet50_fcn(average_pool=None,
              n_out=2,
              final_conv_area=2,
              top_model_weights_path='../exp_dir/fish_localise/training/bottleneck_fc_model_resnet50.h5'):
    """
    Transform the Resnet50 model to fully connected
    :param average_pool:
    :param n_out:
    :param final_conv_area:
    :param top_model_weights_path:
    :return:
    """
    resn50 = ResNet50(include_top=False, weights='imagenet', input_shape=(3, None, None))
    base_model = Model(input=resn50.input, output=resn50.layers[-2].output)

    print('RESNET50 Model loaded...............')
    # build a classifier model to put on top of the convolutional model
    old_top_model = Sequential()
    old_top_model.add(Flatten(input_shape=(2048, final_conv_area, final_conv_area)))
    old_top_model.add(Dense(n_out, activation='relu'))
    old_top_model.load_weights(top_model_weights_path)
    dense_weights = old_top_model.layers[-1].weights[0].get_value()
    print(dense_weights.shape)
    # add the model on top of the convolutional base
    top_model = Sequential()
    top_model.add(Convolution2D(2, final_conv_area, final_conv_area, activation='relu', border_mode='same', input_shape=(2048, None, None)))
    new_conv_shape = top_model.layers[0].weights[0].get_value().shape
    print(new_conv_shape)
    #new_conv_weights = dense_weights.transpose(1,0).reshape(new_conv_shape)[:,:,::-1,::-1]
    new_conv_weights = dense_weights.transpose(1,0).reshape((2, 2048, final_conv_area, final_conv_area))[:,:,::-1,::-1]
    # set the value--> from dense to convolutional
    # according to http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
    top_model.layers[0].weights[0].set_value(new_conv_weights)
    top_model.layers[0].weights[1].set_value(old_top_model.layers[-1].weights[1].get_value())

    print('TOP convolutional parameters loaded...............')
    model = Sequential()
    model.add(base_model)
    if average_pool:
        model.add(
            AveragePooling2D((average_pool, average_pool), name='avg_pool', input_shape=base_model.output_shape[1:]))
    # else:
    #     # because originally we define the network as a average pooling
    #     model.add(MyScaleLayer(1/3.))  ## THIS METHOD WILL NOT WORK!!!!!!!!!!!!!!!!!!!
    model.add(top_model)
    print('TOP Model loaded...............')

    return model