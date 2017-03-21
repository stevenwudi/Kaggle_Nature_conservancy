import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pickle
from classes.DataLoaderClass import my_portable_hash
from architecture.vgg_fcn import get_model_vgg_fully_connect, get_model_vgg_fcn
from keras.preprocessing.image import img_to_array, load_img

exp_dir_path = '../exp_dir/fish_localise'
top_model_weights_path = os.path.join(exp_dir_path, 'training', 'bottleneck_fc_model.h5')
image_size = (180, 180)
# number of image scene classes
n_out = 2
train_data_dir = os.path.join(exp_dir_path, 'train_binary')
validation_data_dir = os.path.join(exp_dir_path, 'valid_binary')
nb_train_samples = 32716
nb_validation_samples = 3615
nb_epoch = 500

h = my_portable_hash([os.listdir(os.path.join(exp_dir_path, 'train_binary', 'ALL')), os.listdir(os.path.join(exp_dir_path, 'train_binary', 'BACKGROUND'))])
mean_shape_path = 'mean_shape_{}'.format(h)
obj = pickle.load(open(os.path.join('../global/objs', mean_shape_path), 'rb'))


test_datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
test_datagen.mean = np.array(obj[0], dtype=np.float32).reshape(3, 1, 1)  # (rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical')

img = load_img(os.path.join(validation_generator.directory, validation_generator.filenames[0]))  # this is a PIL image
x = img_to_array(img)
im_180 = cv2.resize(x.transpose(1,2,0), (180, 180))
x_new = im_180.transpose(2,0,1)
# this is a Numpy array with shape (3, 150, 150)
x_new = x_new.reshape((1,) + x_new.shape)
x_new = x_new - obj[0][None, :,  None, None]

vgg_fully_connect = get_model_vgg_fully_connect()
out = vgg_fully_connect.predict(x_new)
print(out)


vgg_fcn = get_model_vgg_fcn()
out_fcn = vgg_fcn.predict(x_new)
print(out_fcn)


im_180 = cv2.resize(x.transpose(1,2,0), (250, 250))
x_new = im_180.transpose(2,0,1)
# this is a Numpy array with shape (3, 150, 150)
x_new = x_new.reshape((1,) + x_new.shape)
x_new = x_new - obj[0][None, :,  None, None]

out_fcn = vgg_fcn.predict(x_new)

