'''
We combine the detected fish which could have different cropping boundingbox than the manually labelled
boundingbox. Of course, we would hope that they have the same boundingbox, but it's not likely
'''
import os
import numpy as np
from distutils.dir_util import copy_tree
import shutil

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# path to the model weights file.
gt_dir = './exp_dir/fish_localise'
detection_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'

# path to save
classification_dir = './exp_dir/fish_classification'


gt_train = os.path.join(gt_dir, 'train')
gt_valid = os.path.join(gt_dir, 'valid')
detection_train_data_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'
detection_validation_data_dir = './exp_dir/fish_localise/train_classification/valid/true_pos_dir'
classification_train_dir = os.path.join(classification_dir, 'train')
classification_valid_dir = os.path.join(classification_dir, 'valid')

if not os.path.exists(classification_train_dir):
    os.mkdir(classification_train_dir)
if not os.path.exists(classification_valid_dir):
    os.mkdir(classification_valid_dir)

# first copy all the fish into the directory
copy_tree(gt_train, classification_train_dir)
copy_tree(gt_valid, classification_valid_dir)
# but the background directory should be deleted because of the historical reason
shutil.rmtree(os.path.join(classification_train_dir, 'BACKGROUND'))
shutil.rmtree(os.path.join(classification_valid_dir, 'BACKGROUND'))

copy_tree(detection_train_data_dir, classification_train_dir)
copy_tree(detection_validation_data_dir, classification_valid_dir)

# Note that for NoF, Di Wu manually added some example...ITS A HACK!
print('HACK HACK HACK, DI WU ADDED NOF FROM FALSE POSITIVE FISH DETECTION@@@')
nof_dir = './exp_dir/fish_localise/train_classification/manually_examined_false_pos'
nof_dst_train_dir = os.path.join(classification_dir, 'train', 'Nof')
nof_dst_valid_dir = os.path.join(classification_dir, 'valid', 'Nof')
if not os.path.exists(nof_dst_valid_dir):
    os.mkdir(nof_dst_valid_dir)
nof_list_dir = os.listdir(nof_dir)
nof_list = np.random.permutation(len(nof_list_dir))

for i in nof_list[:int(len(nof_list_dir)*0.9)]:
    dst_file = os.path.join(nof_dst_train_dir, nof_list_dir[i])
    shutil.copyfile(os.path.join(nof_dir, nof_list_dir[i]), dst_file)

for i in nof_list[int(len(nof_list_dir)*0.9):]:
    dst_file = os.path.join(nof_dst_valid_dir, nof_list_dir[i])
    shutil.copyfile(os.path.join(nof_dir, nof_list_dir[i]), dst_file)

# but also, because we have an under-representative fish like Opah,
# for better fish detection, we need to scale them up accordingly.
# the less it has, the more we insert them
fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'train')
fish_type = os.listdir(train_data_dir_positive)

for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))

# after check we have 7080 fish
fish_num = np.asarray(fish_num)
acc_prob = np.add.accumulate(fish_num*1.0 / np.sum(fish_num))
acc_dict = {x: 0 for x in fish_type}

# we generate almost equal size of number of images for each type, total number will be 96956
total_num = int(1/acc_prob.min()*np.sum(fish_num))

# but we actually stop at sum(fish_num) = 61264
for rand_sample in np.random.rand(total_num):
    bucket = np.asarray(np.where(acc_prob > rand_sample * acc_prob.max())).min()
    acc_dict[fish_type[bucket]] += 1
    # we put this type of fish into the training for bootstrapping
    # we randomly choose the replica of the image again...
    file_number = np.random.randint(0, len(os.listdir(os.path.join(train_data_dir_positive, fish_type[bucket]))))
    image_name = os.listdir(os.path.join(train_data_dir_positive, fish_type[bucket]))[file_number]
    src = os.path.join(train_data_dir_positive, fish_type[bucket], image_name)
    img = load_img(src)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    # we generate a random name for this replica
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=os.path.join(train_data_dir_positive, fish_type[bucket]),
                              save_prefix='GENERATED_', save_format='jpg'):
        i += 1
        if i > 1:
            break  # otherwise the generator would loop indefinitely

print(acc_dict)
# [7911, 3901, 9560, 5192, 3092, 13447, 7180, 10981]