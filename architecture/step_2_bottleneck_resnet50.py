'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, AveragePooling2D
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
nb_epoch = 100
n_out = 2

# h = my_portable_hash([os.listdir(os.path.join(exp_dir_path, 'train_binary', 'ALL')), os.listdir(os.path.join(exp_dir_path, 'train_binary', 'BACKGROUND'))])
# mean_shape_path = 'mean_shape_{}'.format(h)
# obj = pickle.load(open(os.path.join('../global/objs',mean_shape_path), 'rb'))
resnet50_data_mean = [103.939, 116.779, 123.68]
from keras.applications.resnet50 import preprocess_input


def save_bottlebeck_features():
    # Ideally we want use original VGG image prepossessing mean and no scale
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(3, img_width, img_height))
    base_model_1 = Model(input=base_model.input, output=base_model.layers[-2].output)

    top_model = Sequential()
    top_model.add(AveragePooling2D((3, 3), name='avg_pool', input_shape=base_model_1.output_shape[1:]))

    # add the model on top of the convolutional base
    extract_model = Sequential()
    extract_model.add(base_model_1)
    extract_model.add(top_model)

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

    if False:
        model.load_weights(top_model_weights_path)
        model.compile(optimizer=optimizers.SGD(lr=1e-6, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=64,
              validation_data=(validation_data, validation_labels))
    # it's better to save the model...
    model.save_weights(top_model_weights_path)
    #model.save(top_model_weights_path)

save_bottlebeck_features()
train_top_model()

# RESNET50 val_loss is small than vgg!!!
# Epoch 49/50
# 32716/32716 [==============================] - 1s - loss: 1.1932e-07 - acc: 1.0000 - val_loss: 0.0521 - val_acc: 0.9947
# Epoch 50/50
# 32716/32716 [==============================] - 1s - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 0.0512 - val_acc: 0.9950

# Retrained... WOW
# Epoch 49/50
# 32716/32716 [==============================] - 1s - loss: 1.1926e-07 - acc: 1.0000 - val_loss: 0.0469 - val_acc: 0.9956
# Epoch 50/50
# 32716/32716 [==============================] - 1s - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 0.0465 - val_acc: 0.9956

