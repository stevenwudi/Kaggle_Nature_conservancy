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

top_dir = "/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy"
exp_dir_path = 'exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model.h5')
# dimensions of our images.
image_size = (180, 180)
# number of image scene classes
n_out = 2

train_data_dir = os.path.join(exp_dir_path, 'train_binary')
validation_data_dir = os.path.join(exp_dir_path, 'valid_binary')
nb_train_samples = 32716
nb_validation_samples = 3615
nb_epoch = 30

h = my_portable_hash([os.listdir(os.path.join(top_dir, exp_dir_path, 'train_binary', 'ALL')),
                      os.listdir(os.path.join(top_dir, exp_dir_path, 'train_binary', 'BACKGROUND'))])
mean_shape_path = 'mean_shape_{}'.format(h)
obj = pickle.load(open(os.path.join(top_dir, 'global/objs', mean_shape_path), 'rb'))

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
top_model.add(Dense(n_out, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)
print('TOP Model loaded...............')

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1.,
    featurewise_center=True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.mean = np.array(obj[0], dtype=np.float32).reshape(3, 1, 1)# (rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_datagen.mean = np.array(obj[0], dtype=np.float32).reshape(3, 1, 1)  # (rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical')

# fine-tune the model
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

model.save_weights(os.path.join(top_dir, exp_dir_path, 'training', 'fine_tune_all_conv.h5'))

