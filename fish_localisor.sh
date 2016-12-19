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
--arch deepsenseio_whale_localisor \
--crop-h 224 \
--crop-w 224 \
--process-recipe-name fetch_rob_crop_recipe \
--sloth-annotations-url boundingbox_annotation \
--aug-params crop1_buckets\
--fish-types 8 \
--global-saver-url global


echo --test-csv-url ${SAMPLE_SUB} --name ${NAME} \
--test-dir-url ${IMG_DIR} --train-dir-url ${IMG_DIR} \
--train-csv-url ${TRAIN_CSV} --glr 0.01 --mb-size 64 --crop-h 224 --crop-w 224 \
--method momentum --arch gscp_smaller --monitor-freq 100 --n-samples-valid 1 --loss-freq 5 --do-pca 1 --pca-scale 0.01 \
--fc-l2-reg 0.05 --conv-l2-reg 0.0005 --do-mean 1 --aug-params crop1_buckets --glr-burnout 15 --glr-decay 0.9955 \
--valid-seed 7300 --slot-annotations-url ${SLOT} --show-images 30 --valid-freq 1 \
--process-recipe-name fetch_rob_crop_recipe --point1-annotations-url ${POINT1} \
--point2-annotations-url ${POINT2} --buckets 60 --target-name crop1 --mode crop1 --global-saver-url global \
--exp-dir-url ${EXP_DIR} --real-valid-shuffle --valid-partial-batches --train-pool-size 1 --n-epochs 501



