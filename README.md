# Kaggle_Nature_conservancy_dwu
Kaggle competition for Nature Conservancy: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

## (1) First step is to collect training images for fish localisation
- `python main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0 --valid-seed 1000 --process-recipe-name vgg_keras_recipe --glr 0.0001 --show_images 0 --collect_training_validation_images 1 --collect_training_validation_stats 1`

## (2) The second step is to train a regressor for fish and background
- `python step_2_bottleneck_resnet50.py` - train a resnet50 bottleneck network
- `python step_2_2_hard_negative_mining_vgg.py` - select hard negatives
- `python step_3_fine_tune_resnet50_keep_valid` - fine tune the trained network` 
- `python step_3_fine_tune_top_resnet50.py` - fine tune the trained network` for all training iamges

- `python main.py --train-dir-url /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train --name fish_localise --exp-dir-url exp_dir/fish_localise --target-name crop1 --arch vgg_convolutional --crop-h 224 --crop-w 224 --sloth-annotations-url boundingbox_annotation --aug-params crop1_buckets --fish-types 7 --global-saver-url global --glr-burnout 15 --glr-decay 0.9999 --fc-l2-reg 0.05 --conv-l2-reg 0.0005 --dropout 0 --dropout-coeff 0.5 --debug_plot 0 --n-samples-valid 1 --mb-size 64 --buckets 60 --monitor-freq 10 --trainable 0 --valid-seed 1000 --process-recipe-name vgg_keras_recipe --glr 0.0001 --show_images 0 --collect_training_validation_images 0 --fish_redetection 1 --load-arch-path /home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5 --init_model 1`
- Generate response maps, we will use it for metaparameter tunning for fish detection (e.g. 

- `python step_5_fish_detection.py` -fish detection using the trained network and non-maximum suppresion?


## (3) The third step is to train a aligner for fish
TODO

## (4) The fourth step is to train a fish classifier
TODO