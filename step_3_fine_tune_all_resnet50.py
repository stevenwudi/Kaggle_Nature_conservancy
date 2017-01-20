import os
import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from architecture.resnet50_fcn import convert_resnet50_to_fcn_model

exp_dir_path = './exp_dir/fish_localise'
# dimensions of our images.
image_size = (200, 200)
# number of image scene classes
n_out = 2
train_data_dir = os.path.join(exp_dir_path, 'train_hard_negatives')
nb_train_samples = 7471+17520
nb_epoch = 100
resnet50_data_mean = [103.939, 116.779, 123.68]

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1.,
    featurewise_center=True,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360,
    horizontal_flip=True)

train_datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=64,
        class_mode='categorical')

# Wudi is just too lazy to do proper document... The next time is the first time
# to do the training
#model = load_model('./exp_dir/fish_localise/training/fine_tune_model_resnet50.h5')
model_path = './exp_dir/fish_localise/training/fine_tune_model_resnet50_retrained.h5'
model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    verbose=2)

model.save(model_path)
save_mode_path = './exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5'
convert_resnet50_to_fcn_model(final_conv_area=7, model_path=model_path, save_mode_path=save_mode_path)


# Epoch 98/100
# 579s - loss: 0.0756 - acc: 0.9911
# Epoch 99/100
# 579s - loss: 0.0624 - acc: 0.9920
# Epoch 100/100
# 579s - loss: 0.0670 - acc: 0.9912

# Epoch 48/50
# 575s - loss: 0.0190 - acc: 0.9929
# Epoch 49/50
# 575s - loss: 0.0244 - acc: 0.9917
# Epoch 50/50
# 574s - loss: 0.0196 - acc: 0.9925
# (100352, 2)
# (2, 2048, 7, 7)
# FCN Model saved as ./exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5

