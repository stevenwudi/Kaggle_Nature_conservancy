'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input

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


def save_bottlebeck_features():
    # Ideally we want use original VGG image prepossessing mean and no scale
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
    datagen.mean = np.array(obj[0], dtype=np.float32).reshape(3, 1, 1)

    base_model = VGG19(include_top=False, weights='imagenet')
    extract_model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='binary',
            shuffle=False)

    bottleneck_features_train = extract_model.predict_generator(generator, nb_train_samples)
    np.save(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_train.npy'), 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='binary',
            shuffle=False)
    bottleneck_features_validation = extract_model.predict_generator(generator, nb_validation_samples)
    np.save(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation.npy'), 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_train.npy')))

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

    validation_data = np.load(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation.npy')))
    validation_labels = np_utils.to_categorical(generator.classes, n_out)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    #model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(top_model_weights_path)
    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


#save_bottlebeck_features()
train_top_model()


# Epoch 49/50
# 32716/32716 [==============================] - 2s - loss: 0.0485 - acc: 0.9970 - val_loss: 0.1134 - val_acc: 0.9925
# Epoch 50/50
# 32716/32716 [==============================] - 2s - loss: 0.0485 - acc: 0.9970 - val_loss: 0.1134 - val_acc: 0.9925
