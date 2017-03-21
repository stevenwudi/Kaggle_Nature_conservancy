#!/usr/bin/env bash

export IMG_DIR="/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train"
export PYTHONPATH="${PYTHONPATH}:."

# First step is to collect training images for fish localisation
/home/stevenwudi/tensorflow/bin/python /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0 --valid-seed 1000 --process-recipe-name vgg_keras_recipe --glr 0.0001 --show_images 0 --collect_training_validation_images 1
