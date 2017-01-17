"""
Note that:
(1)in order to perform fine-tuning, all layers should start with properly trained weights:
for instance you should not slap a randomly initialized fully-connected network on top of
a pre-trained convolutional base. This is because the large gradient updates triggered by the
randomly initialized weights would wreck the learned weights in the convolutional base.
In our case this is why we first train the top-level classifier, and only then start fine-tuning
convolutional weights alongside it.
(2) we choose to only fine-tune the last convolutional block rather than the entire network
in order to prevent overfitting, since the entire network would have a very large entropic
capacity and thus a strong tendency to overfit. The features learned by low-level convolutional
blocks are more general, less abstract than those found higher-up, so it is sensible to keep the
first few blocks fixed (more general features) and only fine-tune the last one (more specialized
features).
(3) fine-tuning should be done with a very slow learning rate, and typically with the SGD optimizer
 rather than an adaptative learning rate optimizer such as RMSProp. This is to make sure that the
  magnitude of the updates stays very small, so as not to wreck the previously learned features.
"""
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model

top_dir = "/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy"
exp_dir_path = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise'
top_model_weights_path = os.path.join('/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise', 'training', 'bottleneck_fc_model_resnet50.h5')
# dimensions of our images.
image_size = (200, 200)
# number of image scene classes
n_out = 2
train_data_dir = os.path.join(exp_dir_path, 'train_hard_negatives_keep_valid', 'train')
validation_data_dir = os.path.join(exp_dir_path, 'train_hard_negatives_keep_valid', 'valid')
nb_train_samples = 4036 + 7170
nb_validation_samples = 435 + 3180
nb_epoch = 20
resnet50_data_mean = [103.939, 116.779, 123.68]

resn50 = ResNet50(include_top=False, weights='imagenet', input_shape=(3, image_size[0], image_size[1]))
base_model = Model(input=resn50.input, output=resn50.layers[-2].output)

print('RESNET50 Model loaded...............')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(2048, 2, 2)))
top_model.add(Dense(n_out, activation='softmax'))
# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Sequential()
model.add(base_model)
model.add(AveragePooling2D((3, 3), name='avg_pool', input_shape=base_model.output_shape[1:]))
model.add(top_model)
print('TOP Model loaded...............')
model.summary()

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

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
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_datagen.mean = np.array(resnet50_data_mean, dtype=np.float32).reshape(3, 1, 1)  # (rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical')

model.load_weights(os.path.join("/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise",
                                'training', 'fine_tune_all_conv_resnet50.h5'))
print("Reload the model")
# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    verbose=1)

model.save_weights(os.path.join("/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/Kaggle_Nature_conservancy/exp_dir/fish_localise",
                                'training', 'fine_tune_all_conv_resnet50.h5'))
#
# Epoch 18/20
# 11206/11206 [==============================] - 516s - loss: 0.9551 - acc: 0.9190 - val_loss: 0.1284 - val_acc: 0.9873
# Epoch 19/20
# 11206/11206 [==============================] - 516s - loss: 0.9465 - acc: 0.9201 - val_loss: 0.1106 - val_acc: 0.9895
# Epoch 20/20
# 11206/11206 [==============================] - 515s - loss: 0.9431 - acc: 0.9202 - val_loss: 0.1233 - val_acc: 0.9873
