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
from architecture.resnet50_fcn import get_model_resnet50_fully_connected_no_pooling

# path to the model weights file.
exp_dir_path = './exp_dir/fish_classification'
# dimensions of our images.
img_width, img_height = 200, 200
train_data_dir = os.path.join(exp_dir_path, 'train')
validation_data_dir = os.path.join(exp_dir_path, 'valid')

nb_train_samples = 61264
nb_validation_samples = 752
nb_epoch = 15
n_out = 8

resnet50_data_mean = [103.939, 116.779, 123.68]
classification_model_path = './exp_dir/fish_classification/classification_bottleneck_from_detection.h5'
classification_combine_model_path = './exp_dir/fish_classification/classification_bottleneck_from_detection_combined.h5'

def save_bottlebeck_features():
    # This base_model is trained on fish detection, we assume that the convolutional features
    # will be more relevant
    model = get_model_resnet50_fully_connected_no_pooling()
    model.load_weights('./exp_dir/fish_localise/training/fine_tune_all_conv_resnet50.h5')
    base_model_1 = Model(input=model.input, output=model.layers[-2].output)

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(AveragePooling2D((7, 7), name='avg_pool', input_shape=(2048, 7, 7)))
    base_model = Sequential()
    base_model.add(base_model_1)
    base_model.add(top_model)
    base_model.add(Flatten())
    print(base_model.summary())

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
    # rescale=1./255)
    # generate valid first
    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
    bottleneck_features_validation = base_model.predict_generator(generator, nb_validation_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50.npy'), 'w'), bottleneck_features_validation)

    # generate train
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)

    bottleneck_features_train = base_model.predict_generator(generator, nb_train_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50.npy'), 'w'), bottleneck_features_train)


def train_top_model():

    train_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50.npy')))
    validation_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50.npy')))

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (resc
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
    # ale=1./255)
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
    train_labels = np_utils.to_categorical(generator.classes, n_out)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
    validation_labels = np_utils.to_categorical(generator.classes, n_out)

    model = Sequential()
    model.add(Dense(n_out, activation='softmax',input_shape=train_data.shape[1:]))

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #               metrics=['accuracy', 'categorical_crossentropy'])
    #
    # model.fit(train_data, train_labels,
    #           nb_epoch=nb_epoch, batch_size=64,
    #           validation_data=(validation_data, validation_labels))

    lr_list = [0.001 * 0.1 ** (x) for x in range(3)]
    model.load_weights(classification_model_path)
    for lr in lr_list:
        print('lr: %.5f'%lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=64,
                  validation_data=(validation_data, validation_labels))
    # it's better to save the model...
    model.save_weights(classification_model_path)
    return model


def combine_top_fish_detection_network(model):
    renset_model = get_model_resnet50_fully_connected_no_pooling()
    renset_model.load_weights('./exp_dir/fish_localise/training/fine_tune_all_conv_resnet50.h5')
    base_model_1 = Model(input=renset_model.input, output=renset_model.layers[-2].output)

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(AveragePooling2D((7, 7), name='avg_pool', input_shape=(2048, 7, 7)))

    base_model = Sequential()
    base_model.add(base_model_1)
    base_model.add(top_model)
    base_model.add(Flatten())
    base_model.add(model)
    classification_combine_model_path = './exp_dir/fish_classification/classification_bottleneck_from_detection_combined.h5'
    base_model.save(classification_combine_model_path)

if False:
    save_bottlebeck_features()
model = train_top_model()
combine_top_fish_detection_network(model)

#
# 61264/61264 [==============================] - 2s - loss: 0.4464 - acc: 0.8379 - categorical_crossentropy: 0.4464 - val_loss: 0.4406 - val_acc: 0.8497 - val_categorical_crossentropy: 0.4406
# Epoch 99/100
# 61264/61264 [==============================] - 1s - loss: 0.4464 - acc: 0.8380 - categorical_crossentropy: 0.4464 - val_loss: 0.4406 - val_acc: 0.8497 - val_categorical_crossentropy: 0.4406
# Epoch 100/100
# 61264/61264 [==============================] - 1s - loss: 0.4464 - acc: 0.8380 - categorical_crossentropy: 0.4464 - val_loss: 0.4406 - val_acc: 0.8497 - val_categorical_crossentropy: 0.4406
