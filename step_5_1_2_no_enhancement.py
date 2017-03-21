'''
We combine the detected fish which could have different cropping boundingbox than the manually labelled
boundingbox. Of course, we would hope that they have the same boundingbox, but it's not likely
'''
import os
import numpy as np
from distutils.dir_util import copy_tree
import shutil
# path to the model weights file.
gt_dir = './exp_dir/fish_localise'
detection_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'

# path to save
classification_dir = './exp_dir/fish_classification'

gt_train = os.path.join(gt_dir, 'train')
gt_valid = os.path.join(gt_dir, 'valid')
detection_train_data_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'
detection_validation_data_dir = './exp_dir/fish_localise/train_classification/valid/true_pos_dir'
classification_train_dir = os.path.join(classification_dir, 'train_no_enhancement')
classification_valid_dir = os.path.join(classification_dir, 'valid_no_enhancement')

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
nof_dst_train_dir = os.path.join(classification_dir, 'train_no_enhancement', 'Nof')
nof_dst_valid_dir = os.path.join(classification_dir, 'valid_no_enhancement', 'Nof')
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
train_data_dir_positive = os.path.join(classification_dir, 'train_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)

for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))

print(fish_num, np.sum(fish_num))
# after check we have 7015 fish

fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'valid_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)

for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))

print(fish_num, np.sum(fish_num))
# 752 for valid

