'''
Note that:
(1)in order to perform fine-tuning, all layers should start with properly trained weights:
for instance you should not slap a randomly initialized fully-connected network on top of
a pre-trained convolutional base. This is because the large gradient updates triggered by the
randomly initialized weights would wreck the learned weights in the convolutional base.
In our case this is why we first train the top-level classifier, and only then start fine-tuning
convolutional weights alongside it.
(2) we choose to only fine-tune the last convolutional block rather than the entire network
in order to prevent overfitting, since the entire network would have a very large entropic
capacity and thus a strong tendency to overfit. The features learned by low-level convolutional
blocks are more general, less abstract than those found higher-up, so it is sensible to keep the
first few blocks fixed (more general features) and only fine-tune the last one (more specialized
features).
(3) fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer
 rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the
  magnitude of the updates stays very small, so as not to wreck the previously learned features.
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.utils import np_utils
from keras.utils.data_utils import get_file
import pickle
from classes.DataLoaderClass import my_portable_hash

exp_dir_path = '../exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model.h5')

# number of image scene classes
n_out = 2

train_data_dir = os.path.join(exp_dir_path, 'train_binary')
validation_data_dir = os.path.join(exp_dir_path, 'valid_binary')
nb_train_samples = 32716
nb_validation_samples = 3615
nb_epoch = 500

h = my_portable_hash([os.listdir(os.path.join(exp_dir_path, 'train_binary', 'ALL')), os.listdir(os.path.join(exp_dir_path, 'train_binary', 'BACKGROUND'))])
mean_shape_path = 'mean_shape_{}'.format(h)
obj = pickle.load(open(os.path.join('../global/objs', mean_shape_path), 'rb'))

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(3, None, None)))
#model.add(ZeroPadding2D((1, 1), input_shape=(3, image_size[0], image_size[1])))
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
old_top_model.add(Flatten(input_shape=(512,5,5)))
old_top_model.add(Dense(n_out, activation='softmax'))
old_top_model.load_weights(top_model_weights_path)
dense_weights = old_top_model.layers[-1].weights[0].get_value()
print(dense_weights.shape)
# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Convolution2D(2, 5, 5, activation='relu', border_mode='valid', input_shape=(512, None, None)))
new_conv_shape = top_model.layers[0].weights[0].get_value().shape
print(new_conv_shape)
new_conv_weights = dense_weights.transpose(1,0).reshape(new_conv_shape)[:,:,::-1,::-1]
new_conv_weights = dense_weights.transpose(1,0).reshape((2, 512, 5, 5))[:,:,::-1,::-1]

# set the value--> from dense to convolutional
# according to http://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
top_model.layers[0].weights[0].set_value(new_conv_weights)
top_model.layers[0].weights[1].set_value(old_top_model.layers[-1].weights[1].get_value())

print('TOP convolutional parameters loaded...............')
# add the model on top of the convolutional base
model.add(top_model)

# now we test our fully convolutional network
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from matplotlib import pyplot as plt
import scipy
test_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_stg1'
img_list = os.listdir(test_directory)

for im_num in range(10):
    img = load_img(os.path.join(test_directory, img_list[im_num]))  # this is a PIL image
    img_origin = img.copy()
    total_scale = 5
    scale_prop = 1.1
    scale_list = [scale_prop**(x-2) for x in range(total_scale)]
    ax = plt.figure(2)
    plt.clf()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # we sample 4 times different scale
    out_list = []
    for i in range(total_scale):
        basewidth = int(float(img_origin.size[0]) * scale_list[i])
        hsize = int((float(img_origin.size[1]) * scale_list[i]))
        img = img.resize((basewidth, hsize))
        x = img_to_array(img)  #
        print("test image is of shape: "+str(x.shape))  #this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)
        x_new = x - obj[0][None, :,  None, None]

        # predict the model output
        out = model.predict(x_new)
        out_list.append(out[0,0,:,])

        # visualise the fish detection result
        # plt.subplot(total_scale+1, 2, i*2+1)
        # plt.imshow(x[0].transpose(1,2,0) /255.)
        # plt.subplot(total_scale+1, 2, i*2+2)
        # plt.imshow(out[0,0,:,])
        # plt.colorbar()
    # we average the ouput
    out_shape =out_list[0].shape

    resize_shape = [x-32*4 for x in img_origin.size[::-1]]
    out_mean = np.mean(np.asarray([scipy.misc.imresize(x, resize_shape) for x in out_list]), axis=0)
    max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img_origin)
    plt.scatter(x=[max_col+2*32+16], y=[max_row+2*32+16], c='r', s=20)
    row_add = int(32*(img_origin.size[::-1][0]/32. - int(img_origin.size[::-1][0]/32)))
    col_add = int(32*(img_origin.size[::-1][1]/32. - int(img_origin.size[::-1][1]/32)))
    plt.scatter(x=[max_col+2*32+row_add], y=[max_row+2*32+col_add], c='b', s=20)

    plt.title('maximum row and col are: %d, %d' %(max_row+2*32+16, max_col+2*32+16))
    plt.subplot(1, 2, 2)
    plt.imshow(out_mean * 1.0/out_mean.max())
    plt.colorbar()
    plt.draw()
    plt.waitforbuttonpress(0.1)
    plt.savefig('../exp_dir/fish_localise/imgs/vgg_fish_detection/'+img_list[im_num], bbox_inches='tight')


print('Finish visualising')


alb_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train/ALB'
img_list = os.listdir(alb_directory)

for im_num in range(len(alb_directory)):
    img = load_img(os.path.join(alb_directory, img_list[im_num]))  # this is a PIL image
    img_origin = img.copy()
    total_scale = 5
    scale_prop = 1.1
    scale_list = [scale_prop**(x-2) for x in range(total_scale)]
    ax = plt.figure(2)
    plt.clf()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # we sample 4 times different scale
    out_list = []
    for i in range(total_scale):
        basewidth = int(float(img_origin.size[0]) * scale_list[i])
        hsize = int((float(img_origin.size[1]) * scale_list[i]))
        img = img.resize((basewidth, hsize))
        x = img_to_array(img)  #
        print("test image is of shape: "+str(x.shape))  #this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)
        x_new = x - obj[0][None, :,  None, None]

        # predict the model output
        out = model.predict(x_new)
        out_list.append(out[0,0,:,])

        # visualise the fish detection result
        # plt.subplot(total_scale+1, 2, i*2+1)
        # plt.imshow(x[0].transpose(1,2,0) /255.)
        # plt.subplot(total_scale+1, 2, i*2+2)
        # plt.imshow(out[0,0,:,])
        # plt.colorbar()
    # we average the ouput
    out_shape =out_list[0].shape

    resize_shape = [x-32*4 for x in img_origin.size[::-1]]
    out_mean = np.mean(np.asarray([scipy.misc.imresize(x, resize_shape) for x in out_list]), axis=0)
    max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img_origin)
    plt.scatter(x=[max_col+2*32+16], y=[max_row+2*32+16], c='r', s=20)
    row_add = int(32*(img_origin.size[::-1][0]/32. - int(img_origin.size[::-1][0]/32)))
    col_add = int(32*(img_origin.size[::-1][1]/32. - int(img_origin.size[::-1][1]/32)))
    plt.scatter(x=[max_col+2*32+row_add], y=[max_row+2*32+col_add], c='b', s=20)

    plt.title('maximum row and col are: %d, %d' %(max_row+2*32+16, max_col+2*32+16))
    plt.subplot(1, 2, 2)
    plt.imshow(out_mean * 1.0/out_mean.max())
    plt.colorbar()
    plt.draw()
    plt.waitforbuttonpress(0.1)
    plt.savefig('../exp_dir/fish_localise/imgs/vgg_fish_detect_train/'+img_list[im_num], bbox_inches='tight')
