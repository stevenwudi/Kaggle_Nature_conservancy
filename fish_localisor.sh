#!/usr/bin/env bash

export IMG_DIR="/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train"
export PYTHONPATH="${PYTHONPATH}:."


TRAIN_CSV=data/trainLabels.csv

POINT1=data/points1.json
POINT2=data/points2.json
SLOT=data/slot.json

SAMPLE_SUB=data/sample_submission.csv

NAME=fish_localise
EXP_DIR=exp_dir/fish_localise

echo --name ${NAME} \
--train-dir-url ${IMG_DIR} \
--load-arch-url architecture \
--exp-dir-url ${EXP_DIR} \
--target-name crop1 \
--arch vgg_convolutional \
--crop-h 224 \
--crop-w 224 \
--sloth-annotations-url boundingbox_annotation \
--aug-params crop1_buckets\
--fish-types 8 \
--global-saver-url global \
--glr 0.0001 \
--glr-burnout 15  \
--momentum 0.9 \
--glr-decay 0.9955 \
--fc-l2-reg 0.05 \
--conv-l2-reg 0.0005 \
--dropout 0 \
--dropout-coeff 0.5 \
--debug_plot 0 \
--n-samples-valid 1 \
--mb-size 64 \
--buckets 20 \
--monitor-freq 50 \
--early_stopping_wait 10 \
--trainable 0 \
--valid-seed 1000 \
--process-recipe-name vgg_keras_recipe \
--show_images 0\
--collect_training_validation_images 0 \
--load-arch-path /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5 \
--init_model 1 \
--test_img_dir \
/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_stg1 \
--test-dir-url \
/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_stg1 \




echo --test-csv-url ${SAMPLE_SUB} --name ${NAME} \
--test-dir-url ${IMG_DIR} --train-dir-url ${IMG_DIR} \
--train-csv-url ${TRAIN_CSV} --glr 0.01 --mb-size 64 --crop-h 224 --crop-w 224 \
--method momentum --arch gscp_smaller --monitor-freq 100 --n-samples-valid 1 --loss-freq 5 --do-pca 1 --pca-scale 0.01 \
--fc-l2-reg 0.05 --conv-l2-reg 0.0005 --do-mean 1 --aug-params crop1_buckets --glr-burnout 15 --glr-decay 0.9955 \
--valid-seed 7300 --slot-annotations-url ${SLOT} --show-images 30 --valid-freq 1 \
--process-recipe-name fetch_rob_crop_recipe --point1-annotations-url ${POINT1} \
--point2-annotations-url ${POINT2} --buckets 60 --target-name crop1 --mode crop1 --global-saver-url global \
--exp-dir-url ${EXP_DIR} --real-valid-shuffle --valid-partial-batches --train-pool-size 1 --n-epochs 501


/home/stevenwudi/tensorflow/bin/python /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch deepsenseio_whale_localisor --crop-h 224 --crop-w 224 --process-recipe-name fetch_rob_crop_recipe --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr 0.01 --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --valid-seed 1000
#--load-arch-path exp_dir/fish_localise/training/epoch_1750.h5

##3 VGG
/home/stevenwudi/tensorflow/bin/python /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --process-recipe-name fetch_rob_crop_recipe --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr 0.01 --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0


# First step is to collect training images for fish localisation
/home/stevenwudi/tensorflow/bin/python /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0 --valid-seed 1000 --process-recipe-name vgg_keras_recipe --glr 0.0001 --show_images 0 --collect_training_validation_images 1 --collect_training_validation_stats 1

/home/stevenwudi/tensorflow/bin/python /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_classification --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0 --valid-seed 1000 --process-recipe-name vgg_keras_recipe --glr 0.0001 --show_images 0 --collect_training_validation_images 0 --load-arch-path /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5 --fish_redetection 0 --iou_meta_parameter_selection 0 --extract_test_fish 1 --init_model 1 --test-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_stg1

im_name =
img_03014.jpg
0.00117056
img_07147.jpg
0.991622
img_00763.jpg
0.497406
img_07859.jpg
0.0696555
img_04726.jpg
0.120357
img_06541.jpg
0.893519
img_00667.jpg
0.326806
img_01716.jpg
0.311201
img_01782.jpg
0.979848
img_01541.jpg
0.993726
img_01505.jpg
0.00908641
img_01204.jpg
0.389952
img_04453.jpg
0.942408
img_02865.jpg
0.0872455
img_06585.jpg
0.297395
img_05475.jpg
0.911858
img_05059.jpg
0.999971
img_06947.jpg
1.06344e-05
img_03456.jpg
0.997345
img_06272.jpg
0.542166
img_04503.jpg
0.105628
img_03924.jpg
0.00161671
img_00196.jpg
0.978968