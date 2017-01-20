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
exp_dir_path = './exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model_resnet50.h5')
# dimensions of our images.
img_width, img_height = 180, 180

train_data_dir = os.path.join(exp_dir_path, 'train_binary')
validation_data_dir = os.path.join(exp_dir_path, 'valid_binary')
train_data_dir_positive = os.path.join(exp_dir_path, 'train')
nb_train_samples = 32716
nb_validation_samples = 3615
nb_epoch = 50
n_out = 2
resnet50_data_mean = [103.939, 116.779, 123.68]


def hard_negative_mining():
    # the begining part is the same as using bottlenck for training

    train_data = np.load(open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_train_resnet50.npy')))

    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)  # (rescale=1./255)
    datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    train_labels = np_utils.to_categorical(train_generator.classes, n_out)

    valid_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

    validation_data = np.load(
        open(os.path.join(exp_dir_path, 'training', 'bottleneck_features_validation_resnet50.npy')))
    validation_labels = np_utils.to_categorical(valid_generator.classes, n_out)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(n_out, activation='softmax'))
    model.load_weights(top_model_weights_path)

    pred = model.predict(validation_data)
    # because we only mine the hard negatives
    idx_neg = (valid_generator.classes == 1)

    diff = np.sum(np.abs(validation_labels[idx_neg] - pred[idx_neg]), axis=1)
    neg_start_index = len(pred) - len(diff)
    index = neg_start_index + np.asarray(sorted(range(len(diff)), key=lambda k: diff[k], reverse=True))

    # now we only include  half of the hard negatives from training data
    pred = model.predict(train_data)
    # because we only mine the hard negatives
    idx_neg = (train_generator.classes == 1)
    diff = np.sum(np.abs(train_labels[idx_neg] - pred[idx_neg]), axis=1)
    neg_start_index = len(pred) - len(diff)
    index = neg_start_index + np.asarray(sorted(range(len(diff)), key=lambda k: diff[k], reverse=True))

    dir_hard_negative = os.path.join(exp_dir_path, 'train_hard_negatives')
    if not os.path.exists(dir_hard_negative):
        os.mkdir(dir_hard_negative)
        os.mkdir(os.path.join(dir_hard_negative, 'ALL'))
        os.mkdir(os.path.join(dir_hard_negative, 'BACKGROUND'))

    # first copy all the fish into the directory
    from distutils.dir_util import copy_tree
    toDirectory = os.path.join(dir_hard_negative, 'ALL')
    fromDirectory = os.path.join(train_generator.directory, 'ALL')
    copy_tree(fromDirectory, toDirectory)
    fromDirectory = os.path.join(valid_generator.directory, 'ALL')
    copy_tree(fromDirectory, toDirectory)

    # now copy the valid negative and a quarter of the hard train negatives
    toDirectory = os.path.join(dir_hard_negative, 'BACKGROUND')
    fromDirectory = os.path.join(valid_generator.directory, 'BACKGROUND')
    copy_tree(fromDirectory, toDirectory)
    from shutil import copyfile
    for idx in index[:len(index)/2]:
        src = os.path.join(train_generator.directory, train_generator.filenames[idx])
        dst = os.path.join(dir_hard_negative, train_generator.filenames[idx])
        copyfile(src, dst)

    # but also, because we have an under-representative fish like Opah,
    # for better fish detection, we need to scale them up accordingly.
    # the less it has, the more we insert them
    fish_num = []
    fish_type = os.listdir(train_data_dir_positive)
    for dir_pos in fish_type:
        fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))
    # after check we have 4471 positive and 17520 background
    # let's sample 3000 more fish? why not?
    fish_num = np.asarray(fish_num)
    acc_prob = np.add.accumulate(float(fish_num.max()) / fish_num)
    acc_dict = {x: 0 for x in fish_type}
    for rand_sample in np.random.rand(3000):
        bucket = np.asarray(np.where(acc_prob> rand_sample*acc_prob.max())).min()
        acc_dict[fish_type[bucket]] += 1
        # we put this type of fish into the training for bootstrapping
        # we randomly choose the replica of the image again...
        file_number = np.random.randint(0, len(os.listdir(os.path.join(train_data_dir_positive, fish_type[bucket]))))
        image_name = os.listdir(os.path.join(train_data_dir_positive, fish_type[bucket]))[file_number]
        src = os.path.join(train_data_dir_positive, fish_type[bucket],image_name)
        # we generate a random name for this replica
        image_copy_name = fish_type[bucket] + image_name[:-4] + str(np.random.randint(0,1e10)) + image_name[-4:]
        dst = os.path.join(dir_hard_negative, 'ALL', image_copy_name)
        copyfile(src, dst)

    print(acc_dict)

hard_negative_mining()

