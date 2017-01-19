import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from architecture.resnet50_fcn import get_model_resnet50_fully_connected_no_pooling, \
    convert_resnet50_to_fcn_model

exp_dir_path = './exp_dir/fish_localise'
# dimensions of our images.
image_size = (200, 200)
# number of image scene classes
n_out = 2
train_data_dir = os.path.join(exp_dir_path, 'train_hard_negatives_keep_valid', 'train')
validation_data_dir = os.path.join(exp_dir_path, 'train_hard_negatives_keep_valid', 'valid')
nb_train_samples = 4036 + 7170
nb_validation_samples = 435 + 3180
nb_epoch = 100
resnet50_data_mean = [103.939, 116.779, 123.68]

model = get_model_resnet50_fully_connected_no_pooling()

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

#fine-tune the model

model.load_weights(os.path.join("./exp_dir/fish_localise", 'training', 'fine_tune_all_conv_resnet50.h5'))

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    verbose=1)

model.save_weights(os.path.join("./exp_dir/fish_localise", 'training', 'fine_tune_all_conv_resnet50.h5'))
#model.load_weights(os.path.join("./exp_dir/fish_localise", 'training', 'fine_tune_all_conv_resnet50.h5'))
model.save('./exp_dir/fish_localise/training/fine_tune_model_resnet50.h5')
convert_resnet50_to_fcn_model()

# Epoch 19/20
# 11206/11206 [==============================] - 527s - loss: 0.1217 - acc: 0.9580 - val_loss: 0.0295 - val_acc: 0.9898
# Epoch 20/20
# 11206/11206 [==============================] - 527s - loss: 0.1163 - acc: 0.9584 - val_loss: 0.0298 - val_acc: 0.9917
