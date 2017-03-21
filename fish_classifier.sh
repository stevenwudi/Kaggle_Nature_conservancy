#!/usr/bin/env bash


## (4) The fourth step is to train a fish classifier
python main.py
--train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train
--name
fish_classification
--target-name
final_no_conn
--crop-h
224
--crop-w
224
--sloth-annotations-url
boundingbox_annotation
--aug-params
crop1_buckets
--fish-types
7
--mb-size
64
--monitor-freq
10
--valid-seed
1000
--process-recipe-name
vgg_keras_recipe
--glr
0.0001
--show_images
0
--collect_training_validation_images
0
--fish_redetection
0
--iou_meta_parameter_selection
0
--extract_test_fish
1
--exp-dir-url
exp_dir/fish_classification
--test_dir_url
../test_stg1
--init_model
1
--classify_test
0
--load-arch-path
./exp_dir/fish_classification/classification_bottleneck_from_detection_combined.h5