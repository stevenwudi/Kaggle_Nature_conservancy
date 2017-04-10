'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.utils import np_utils
from keras import optimizers

# path to the model weights file.
exp_dir_path = './exp_dir/fish_classification'
# dimensions of our images.
img_width, img_height = 200, 200
train_data_dir = os.path.join(exp_dir_path, 'train_no_enhancement')
validation_data_dir = os.path.join(exp_dir_path, 'valid_no_enhancement')

nb_train_samples = 7015
nb_validation_samples = 752
nb_epoch = 20
n_out = 8

resnet50_data_mean = [103.939, 116.779, 123.68]

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1.,
    featurewise_center=True,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)  # (rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=[img_width, img_height],
    batch_size=64,
    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
valid_datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)  # (rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=[img_width, img_height],
    batch_size=64,
    class_mode='categorical')


def save_bottlebeck_features():
    # This base_model is trained on fish detection, we assume that the convolutional features
    # will be more relevant

    base_model_1 = ResNet50(include_top=False, weights='imagenet', input_shape=(3, img_width, img_height))
    base_model = Model(input=base_model_1.input, output=base_model_1.layers[-1].output)
    #print(base_model.summary())

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
    # generate valid first
    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=64,
            class_mode='categorical',
            shuffle=False)

    bottleneck_features_validation = base_model.predict_generator(generator, nb_validation_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50_no_enhancement_avgpool.npy'), 'w'), bottleneck_features_validation)

    # generate train
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=64,
            class_mode='categorical',
            shuffle=False)

    bottleneck_features_train = base_model.predict_generator(generator, nb_train_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50_no_enhancement_avgpool.npy'), 'w'), bottleneck_features_train)


def train_top_model():

    train_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50_no_enhancement_avgpool.npy')))
    validation_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50_no_enhancement_avgpool.npy')))

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
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
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(n_out, activation='softmax'))

    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy', 'categorical_crossentropy'])

    lr_list = [0.001 * 0.1 ** (x) for x in range(3)]
    for lr in lr_list:
        print('lr: %.5f'%lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        model.fit(train_data,
                  train_labels,
                  nb_epoch=nb_epoch*10,
                  batch_size=64,
                  class_weight='auto',
                  validation_data=(validation_data, validation_labels))
    # # it's better to save the model...
    model.save_weights(os.path.join(exp_dir_path, 'bottleneck_fc_model_no_enhancement_avgpool.h5'))
    return model

    # Epoch 200/200
    # 7015/7015 [==============================] - 0s - loss: 0.0672 - acc: 0.9930 - categorical_crossentropy: 0.0672 - val_loss: 0.2966 - val_acc: 0.9122 - val_categorical_crossentropy: 0.2966


def fine_tune_top_network():
    # first round of model
    resn50 = ResNet50(include_top=False, weights='imagenet', input_shape=(3, img_width, img_height))
    print('RESNET50 Model loaded...............')
    # build a classifier model to put on top of the convolutional model
    old_top_model = Sequential()
    old_top_model.add(Flatten(input_shape=(2048, 1, 1)))
    old_top_model.add(Dense(n_out, activation='softmax'))
    old_top_model.load_weights(os.path.join(exp_dir_path, 'bottleneck_fc_model_no_enhancement_avgpool.h5'))
    dense_weights = old_top_model.layers[-1].weights[0].get_value()
    print(dense_weights.shape)
    print('TOP Model loaded...............')

    last = resn50.layers[-1].output
    x = Flatten()(last)
    predictions = Dense(n_out, activation='softmax')(x)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    model = Model(resn50.input, output=predictions)
    model.layers[-1].weights[0].set_value(old_top_model.layers[-1].weights[0].get_value())
    model.layers[-1].weights[1].set_value(old_top_model.layers[-1].weights[1].get_value())
    print('TOP Model loaded for new model...............')
    for layer in model.layers:
        layer.trainable = True

    # build a classifier model to put on top of the convolutional model
    lr_list = [0.001]
    for lr in lr_list:
        print('lr: %.5f'%lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        # fine-tune the model
        model.fit_generator(
            generator=train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=20,
            validation_data=validation_generator,
            class_weight='auto',
            nb_val_samples=64,)

        model.save(os.path.join(exp_dir_path, 'bottleneck_top_model_avgpool.h5'))
    print('Done')
    # Epoch
    # 19 / 20
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 276
    # s - loss: 0.0483 - acc: 0.9840 - categorical_crossentropy: 0.0483 - val_loss: 0.0166 - val_acc: 1.0000 - val_categorical_crossentropy: 0.0166
    # Epoch
    # 20 / 20
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 276
    # s - loss: 0.0416 - acc: 0.9866 - categorical_crossentropy: 0.0416 - val_loss: 0.3037 - val_acc: 0.9688 - val_categorical_crossentropy: 0.3037


def further_fine_tune():
    model = load_model(os.path.join(exp_dir_path, 'bottleneck_top_model_avgpool.h5'))
    lr_list = [0.001 * 0.1 ** (x) for x in range(3)]
    for lr in lr_list:
        print('lr: %.5f' % lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        # fine-tune the model
        model.fit_generator(
            generator=train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=50,
            validation_data=validation_generator,
            class_weight='auto',
            nb_val_samples=64, )

        model.save(os.path.join(exp_dir_path, 'bottleneck_top_model_avgpool_further_fine_tune.h5'))
    print('Done')
    # Epoch
    # 49 / 50
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 272
    # s - loss: 0.0062 - acc: 0.9983 - categorical_crossentropy: 0.0062 - val_loss: 0.0284 - val_acc: 0.9844 - val_categorical_crossentropy: 0.0284
    # Epoch
    # 50 / 50
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 272
    # s - loss: 0.0081 - acc: 0.9971 - categorical_crossentropy: 0.0081 - val_loss: 0.2461 - val_acc: 0.9531 - val_categorical_crossentropy: 0.2461

import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def early_stopping():
    from keras.callbacks import ReduceLROnPlateau
    history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=10e-6)

    model = load_model(os.path.join(exp_dir_path, 'bottleneck_top_model_avgpool_further_fine_tune.h5'))

    model.compile(optimizer=optimizers.SGD(lr=10e-4, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])
    # fine-tune the model
    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=50,
        validation_data=validation_generator,
        class_weight='auto',
        nb_val_samples=64,
        callbacks=[history, reduce_lr],
    )
    print history.losses
    model.save(os.path.join(exp_dir_path, 'bottleneck_top_model_avgpool_further_fine_tune.h5'))
    print('Done')


if False:
    save_bottlebeck_features()
    model = train_top_model()
    fine_tune_top_network()
    print("futher fine tune")
    further_fine_tune()

print("early_stopping")
early_stopping()