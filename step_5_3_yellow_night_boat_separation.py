'''
We combine the detected fish which could have different cropping boundingbox than the manually labelled
boundingbox. Of course, we would hope that they have the same boundingbox, but it's not likely
'''
import os
import numpy as np
import shutil
from shutil import copyfile
import cPickle
import cv2


# loading a GMM for colormean classfication
with open('./exp_dir/night_boat_classifier.pkl', 'rb') as fid:
    gmix_color_mean = cPickle.load(fid)

full_img_dir = '.././train'
gt_dir = './exp_dir/fish_localise'
detection_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'

# path to save
classification_dir = './exp_dir/fish_classification'

gt_train = os.path.join(gt_dir, 'train')
gt_valid = os.path.join(gt_dir, 'valid')
detection_train_data_dir = './exp_dir/fish_localise/train_classification/train/true_pos_dir'
detection_validation_data_dir = './exp_dir/fish_localise/train_classification/valid/true_pos_dir'

# set saving directory
classification_train_day_dir = os.path.join(classification_dir, 'train_day_no_enhancement')
classification_valid_day_dir = os.path.join(classification_dir, 'valid_day_no_enhancement')
classification_train_night_dir = os.path.join(classification_dir, 'train_night_no_enhancement')
classification_valid_night_dir = os.path.join(classification_dir, 'valid_night_no_enhancement')
if not os.path.exists(classification_train_day_dir):
    os.mkdir(classification_train_day_dir)
if not os.path.exists(classification_valid_day_dir):
    os.mkdir(classification_valid_day_dir)
if not os.path.exists(classification_train_night_dir):
    os.mkdir(classification_train_night_dir)
if not os.path.exists(classification_valid_night_dir):
    os.mkdir(classification_valid_night_dir)

# first copy all the fish into the directory
if True:
    fish_type = os.listdir(gt_train)
    for f_dir in fish_type:
        if not f_dir == 'BACKGROUND':
            img_list = os.listdir(os.path.join(gt_train, f_dir))
            if not os.path.exists(os.path.join(classification_train_day_dir, f_dir)):
                os.mkdir(os.path.join(classification_train_day_dir, f_dir))
            if not os.path.exists(os.path.join(classification_train_night_dir, f_dir)):
                os.mkdir(os.path.join(classification_train_night_dir, f_dir))
            for img in img_list:
                img_name = img[:9]+img[-4:]
                img_full = os.path.join(full_img_dir, f_dir, img_name)
                if not os.path.isfile(img_full):
                    print ("img %s doest not exists!!!!"%img_full)
                else:
                    # classify whether it is a night boat or a day boat
                    im_whole = cv2.imread(img_full)
                    color_mean = im_whole.mean(axis=0).mean(axis=0)
                    label_color_mean = gmix_color_mean.predict(color_mean)
                    if label_color_mean==0:
                        copyfile(os.path.join(gt_train,f_dir, img),
                                 os.path.join(classification_train_day_dir, f_dir, img))
                    else:
                        copyfile(os.path.join(gt_train,f_dir, img),
                                 os.path.join(classification_train_night_dir, f_dir, img))
if True:
    fish_type = os.listdir(gt_valid)
    for f_dir in fish_type:
        if not f_dir == 'BACKGROUND':
            if not os.path.exists(os.path.join(classification_valid_day_dir, f_dir)):
                os.mkdir(os.path.join(classification_valid_day_dir, f_dir))
            if not os.path.exists(os.path.join(classification_valid_night_dir, f_dir)):
                os.mkdir(os.path.join(classification_valid_night_dir, f_dir))
            img_list = os.listdir(os.path.join(gt_valid, f_dir))
            for img in img_list:
                img_name = img[:9] + img[-4:]
                img_full = os.path.join(full_img_dir, f_dir, img_name)
                if not os.path.isfile(img_full):
                    print ("img %s doest not exists!!!!" % img_full)
                else:
                    # classify whether it is a night boat or a day boat
                    im_whole = cv2.imread(img_full)
                    color_mean = im_whole.mean(axis=0).mean(axis=0)
                    label_color_mean = gmix_color_mean.predict(color_mean)
                    if label_color_mean == 0:
                        copyfile(os.path.join(gt_valid, f_dir, img),
                                 os.path.join(classification_valid_day_dir, f_dir, img))
                    else:
                        copyfile(os.path.join(gt_valid, f_dir, img),
                                 os.path.join(classification_valid_night_dir, f_dir, img))

# Note that for NoF, Di Wu manually added some example...ITS A HACK!
print('HACK HACK HACK, DI WU ADDED NOF FROM FALSE POSITIVE FISH DETECTION@@@')
nof_dir = './exp_dir/fish_localise/train_classification/manually_examined_false_pos'

nof_dst_train_day_dir = os.path.join(classification_dir, 'train_day_no_enhancement', 'Nof')
nof_dst_valid_day_dir = os.path.join(classification_dir, 'valid_day_no_enhancement', 'Nof')
nof_dst_train_night_dir = os.path.join(classification_dir, 'train_night_no_enhancement', 'Nof')
nof_dst_valid_night_dir = os.path.join(classification_dir, 'valid_night_no_enhancement', 'Nof')
if not os.path.exists(nof_dst_train_day_dir):
    os.mkdir(nof_dst_train_day_dir)
if not os.path.exists(nof_dst_valid_day_dir):
    os.mkdir(nof_dst_valid_day_dir)
if not os.path.exists(nof_dst_train_night_dir):
    os.mkdir(nof_dst_train_night_dir)
if not os.path.exists(nof_dst_valid_night_dir):
    os.mkdir(nof_dst_valid_night_dir)

nof_list_dir = os.listdir(nof_dir)
nof_list = np.random.permutation(len(nof_list_dir))

for i in nof_list[:int(len(nof_list_dir) * 0.9)]:
    f_dir = nof_list_dir[i].split('_')[0]
    if f_dir=='Nof':
        f_dir = 'NoF'
    img_name = ('_').join(nof_list_dir[i].split('_')[1:])
    img_full = os.path.join(full_img_dir, f_dir, img_name)
    if not os.path.isfile(img_full):
        print ("img %s doest not exists!!!!" % img_full)
    else:
        # classify whether it is a night boat or a day boat
        im_whole = cv2.imread(img_full)
        color_mean = im_whole.mean(axis=0).mean(axis=0)
        label_color_mean = gmix_color_mean.predict(color_mean)
        if label_color_mean == 0:
            copyfile(os.path.join(nof_dir, nof_list_dir[i]),
                     os.path.join(nof_dst_train_day_dir, nof_list_dir[i]))
        else:
            copyfile(os.path.join(nof_dir, nof_list_dir[i]),
                     os.path.join(nof_dst_train_night_dir, nof_list_dir[i]))


for i in nof_list[int(len(nof_list_dir) * 0.9):]:
    f_dir = nof_list_dir[i].split('_')[0]
    if f_dir=='Nof':
        f_dir = 'NoF'
    img_name = ('_').join(nof_list_dir[i].split('_')[1:])
    img_full = os.path.join(full_img_dir, f_dir, img_name)
    if not os.path.isfile(img_full):
        print ("img %s doest not exists!!!!" % img_full)
    else:
        # classify whether it is a night boat or a day boat
        im_whole = cv2.imread(img_full)
        color_mean = im_whole.mean(axis=0).mean(axis=0)
        label_color_mean = gmix_color_mean.predict(color_mean)
        if label_color_mean == 0:
            copyfile(os.path.join(nof_dir, nof_list_dir[i]),
                     os.path.join(nof_dst_valid_day_dir, nof_list_dir[i]))
        else:
            copyfile(os.path.join(nof_dir, nof_list_dir[i]),
                     os.path.join(nof_dst_valid_night_dir, nof_list_dir[i]))


################# Some statistics printing
fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'train_day_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)
for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))
print(fish_num, np.sum(fish_num))

fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'valid_day_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)
for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))
print(fish_num, np.sum(fish_num))

fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'train_night_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)
for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))
print(fish_num, np.sum(fish_num))

fish_num = []
train_data_dir_positive = os.path.join(classification_dir, 'valid_night_no_enhancement')
fish_type = os.listdir(train_data_dir_positive)
for dir_pos in fish_type:
    fish_num.append(len(os.listdir(os.path.join(train_data_dir_positive, dir_pos))))
print(fish_num, np.sum(fish_num))


# ([219, 76, 483, 65, 90, 1958, 258, 531], 3680)
# ([25, 9, 54, 9, 7, 210, 23, 54], 391)
# ([110, 38, 183, 111, 4, 353, 23, 227], 1049)
# ([12, 3, 20, 12, 0, 45, 4, 22], 118)