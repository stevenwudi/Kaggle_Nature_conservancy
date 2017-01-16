'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.utils import np_utils

import pickle
from classes.DataLoaderClass import my_portable_hash

# path to the model weights file.
exp_dir_path = '../exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model.h5')
# dimensions of our images.
img_width, img_height = 180, 180

train_data_dir = os.path.join(exp_dir_path, 'train_binary')
validation_data_dir = os.path.join(exp_dir_path, 'valid_binary')
nb_train_samples = 32716
nb_validation_samples = 3615
nb_epoch = 50
n_out = 2

h = my_portable_hash([os.listdir(os.path.join(exp_dir_path, 'train_binary', 'ALL')), os.listdir(os.path.join(exp_dir_path, 'train_binary', 'BACKGROUND'))])
mean_shape_path = 'mean_shape_{}'.format(h)
obj = pickle.load(open(os.path.join('../global/objs',mean_shape_path), 'rb'))


def hard_negative_mining():

    train_data = np.load(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_train_resnet50.npy')))

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
    datagen.mean = np.array(obj[0], dtype=np.float32).reshape(3, 1, 1)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    train_labels = np_utils.to_categorical(generator.classes, n_out)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    validation_data = np.load(
        open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation_resnet50.npy')))
    validation_labels = np_utils.to_categorical(generator.classes, n_out)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(n_out, activation='softmax'))

    if True:
        model.load_weights(top_model_weights_path)
        model.compile(optimizer=optimizers.SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=64,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

hard_negative_mining()

