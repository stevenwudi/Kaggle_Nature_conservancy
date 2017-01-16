'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers

import pickle
from classes.DataLoaderClass import my_portable_hash

# path to the model weights file.
exp_dir_path = '../exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model_resnet50.h5')
# dimensions of our images.
img_width, img_height = 200, 200

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

    base_model = ResNet50(include_top=False, weights='imagenet')
    extract_model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='binary',
            shuffle=False)

    bottleneck_features_train = extract_model.predict_generator(generator, nb_train_samples)
    np.save(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_train_resnet50.npy'), 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='binary',
            shuffle=False)
    bottleneck_features_validation = extract_model.predict_generator(generator, nb_validation_samples)
    np.save(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation_resnet50.npy'), 'w'), bottleneck_features_validation)


def train_top_model():
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

    validation_data = np.load(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation_resnet50.npy')))
    validation_labels = np_utils.to_categorical(generator.classes, n_out)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(n_out, activation='softmax'))

    if True:
        model.load_weights(top_model_weights_path)
        model.compile(optimizer=optimizers.SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=64,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

#save_bottlebeck_features()
train_top_model()

# RESNET50 val_loss is small than vgg!!!
# Epoch 49/50
# 32716/32716 [==============================] - 0s - loss: 0.0014 - acc: 0.9998 - val_loss: 0.0260 - val_acc: 0.9925
# Epoch 50/50
# 32716/32716 [==============================] - 0s - loss: 0.0014 - acc: 0.9998 - val_loss: 0.0260 - val_acc: 0.9925
