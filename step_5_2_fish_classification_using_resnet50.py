'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, AveragePooling2D, Dropout
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
    rotation_range=360,
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
    base_model = Model(input=base_model_1.input, output=base_model_1.layers[-2].output)
    #print(base_model.summary())

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
    # generate valid first
    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
    bottleneck_features_validation = base_model.predict_generator(generator, nb_validation_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50_no_enhancement.npy'), 'w'), bottleneck_features_validation)

    # generate train
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)

    bottleneck_features_train = base_model.predict_generator(generator, nb_train_samples)
    np.save(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50_no_enhancement.npy'), 'w'), bottleneck_features_train)


def train_top_model():

    train_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_train_resnet50_no_enhancement.npy')))
    validation_data = np.load(open(os.path.join(exp_dir_path, 'bottleneck_features_validation_resnet50_no_enhancement.npy')))

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
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_out, activation='softmax'))

    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy', 'categorical_crossentropy'])
    # model.fit(train_data, train_labels,
    #           nb_epoch=nb_epoch, batch_size=64,
    #           validation_data=(validation_data, validation_labels))

    lr_list = [0.001 * 0.1 ** (x) for x in range(3)]
    for lr in lr_list:
        print('lr: %.5f'%lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=64,
                  validation_data=(validation_data, validation_labels))
    # # it's better to save the model...
    model.save_weights(os.path.join(exp_dir_path, 'bottleneck_fc_model_no_enhancement.h5'))
    return model

    # Epoch
    # 14 / 15
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 2
    # loss: 0.0140 - acc: 0.9960 - categorical_crossentropy: 0.0140 - val_loss: 0.2658 - val_acc: 0.9441 - val_categorical_crossentropy: 0.2658
    # Epoch
    # 15 / 15
    # 7015 / 7015[ == == == == == == == == == == == == == == ==] - 2
    # loss: 0.0125 - acc: 0.9969 - categorical_crossentropy: 0.0125 - val_loss: 0.2656 - val_acc: 0.9441 - val_categorical_crossentropy: 0.2656


def fine_tune_top_network():
    # first round of model
    if False:
        renset_model = ResNet50(include_top=False, weights='imagenet', input_shape=(3, img_width, img_height))

        last = renset_model.layers[-2].output
        x = Flatten()(last)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(n_out, activation='softmax')(x)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in renset_model.layers:
            layer.trainable = False
        model = Model(renset_model.input, output=predictions)

        # build a classifier model to put on top of the convolutional model

        lr_list = [0.001]
        for lr in lr_list:
            print('lr: %.5f'%lr)
            model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy',
                          metrics=['accuracy', 'categorical_crossentropy'])
            # fine-tune the model
            model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=20,
                validation_data=validation_generator,
                nb_val_samples=nb_validation_samples)

            model.save(os.path.join(exp_dir_path, 'bottleneck_top_model_no_enhancement.h5'))
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True
    else:
        # retrain the model
        model = load_model(os.path.join(exp_dir_path, 'fine_tuned_top_model_no_enhancement.h5'))
        for layer in model.layers:
            layer.trainable = True

    lr_list = [0.0001 * 0.1 ** (x) for x in range(3)]
    for lr in lr_list:
        print('lr: %.5f' % lr)
        model.compile(optimizer=optimizers.SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_crossentropy'])
        # fine-tune the model
        model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples,
            class_weight='auto')
        model.save(os.path.join(exp_dir_path, 'fine_tuned_top_model_no_enhancement.h5'))


if False:
    save_bottlebeck_features()
    model = train_top_model()
fine_tune_top_network()
