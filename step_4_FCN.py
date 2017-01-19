import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
import scipy
from architecture.vgg_fcn import softmax
from architecture.resnet50_fcn import load_resnet50_model, get_model_resnet50_fcn_no_pooling,get_fcn_resnet50_last_layer_model

# number of image scene classes
resnet50_data_mean = [103.939, 116.779, 123.68]
# now we test our fully convolutional network
model = get_model_resnet50_fcn_no_pooling()
#model = load_resnet50_model()

total_scale = 7
scale_prop = 1.1
minus_scale = total_scale/2
scale_list = [scale_prop ** (x-minus_scale) for x in range(total_scale)]

if False:
    alb_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train/ALB'
    img_list = os.listdir(alb_directory)
    for im_num in range(len(alb_directory)):
        print("test image: " + img_list[im_num])
        img = load_img(os.path.join(alb_directory, img_list[im_num]))  # this is a PIL image
        img_origin = img.copy()
        # we sample 4 times different scale
        out_list = []
        for i in range(total_scale):
            basewidth = int(float(img_origin.size[0]) * scale_list[i])
            hsize = int((float(img_origin.size[1]) * scale_list[i]))
            img = img.resize((basewidth, hsize))
            x = img_to_array(img)  #
            print("test image is of shape: "+str(x.shape))  #this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)
            x_new = x - np.asarray(resnet50_data_mean)[None, :, None, None]
            # predict the model output
            out = model.predict(x_new)
            out = softmax(out)
            out_list.append(out[0, 0, :, :])

        # we average the ouput
        max_list = [np.max(x) for x in out_list]
        out_shape =out_list[0].shape
        resize_shape = [x for x in img_origin.size[::-1]]
        out_mean = np.mean(np.asarray([scipy.misc.imresize(x, resize_shape) * m for x, m in zip(out_list, max_list)]),
                           axis=0)
        max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)

        ax = plt.figure(1)
        plt.clf()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplot(1, 2, 1)
        plt.imshow(img_origin)
        plt.scatter(x=[max_col], y=[max_row], c='r', s=30)
        plt.title('scale response is: ' + str(['{:.3f}'.format(i) for i in max_list]))

        plt.subplot(1, 2, 2)
        im = plt.imshow(out_mean * 1.0 / 255)
        from matplotlib import colors
        norm = colors.Normalize(vmin=0, vmax=1)
        im.set_norm(norm)
        norm = colors.Normalize(vmin=0, vmax=1)
        plt.colorbar(ticks=np.linspace(0, 1.0, 10, endpoint=True))
        plt.title('resnet oringial size maximum response is %.2f' % (out_list[0].max()))
        plt.draw()
        plt.waitforbuttonpress(0.1)
        plt.savefig('./exp_dir/fish_localise/imgs/resnet50_fish_detect_train/'+img_list[im_num], bbox_inches='tight')
###############################################


test_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_stg1'
#test_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/test_pesudo'
img_list = os.listdir(test_directory)

#for im_num in range(len(img_list[5:30])):
softmax_flag = True
for im_num in range(len(img_list)):
    print("test image: "+img_list[im_num])
    img = load_img(os.path.join(test_directory, img_list[im_num]))  # this is a PIL image
    img_origin = img.copy()
    # we sample 4 times different scale
    out_list = []
    for i in range(total_scale):
        basewidth = int(float(img_origin.size[0]) * scale_list[i])
        hsize = int((float(img_origin.size[1]) * scale_list[i]))
        img = img.resize((basewidth, hsize))
        x = img_to_array(img)  #
        print("test image is of shape: "+str(x.shape))  #this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)
        x_new = x - np.asarray(resnet50_data_mean)[None, :,  None, None]
        # predict the model output
        out = model.predict(x_new)
        print(out.shape)
        if softmax_flag:
            out = softmax(out)
        out_list.append(out[0, 0, :, :])

    # visualise the fish detection result
    # plt.subplot(total_scale+1, 2, i*2+1)
    # plt.imshow(x[0].transpose(1,2,0) /255.)
    # plt.subplot(total_scale+1, 2, i*2+2)
    # plt.imshow(out[0,0,:,])
    # plt.colorbar()
    # we average the ouput
    max_list = [np.max(x) for x in out_list]
    resize_shape = [x for x in img_origin.size[::-1]]
    out_mean = np.mean(np.asarray([scipy.misc.imresize(x, resize_shape)* m for x, m in zip(out_list, max_list)]), axis=0)
    #out_mean = np.mean(np.asarray([scipy.misc.imresize(x, resize_shape) for x in out_list]), axis=0)
    max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)
    if softmax_flag:
        ax = plt.figure(2)
    else:
        ax = plt.figure(1)
    # maximize the figure
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(img_origin)
    plt.scatter(x=[max_col], y=[max_row], c='r', s=30)
    plt.title('scale response is: ' + str(['{:.3f}'.format(i) for i in max_list]))

    plt.subplot(1, 2, 2)
    if softmax_flag:
        im = plt.imshow(out_mean * 1.0 / 255)
        from matplotlib import colors
        norm = colors.Normalize(vmin=0, vmax=1)
        im.set_norm(norm)
        norm = colors.Normalize(vmin=0, vmax=1)
        plt.colorbar(ticks=np.linspace(0, 1.0, 10, endpoint=True))
    else:
        im = plt.imshow(out_mean * 1.0 / out_mean.max())
        # because out_list[2] is the original size in the above definition.
        # we used the activation for an indication of the existence of the fish

    plt.title("test image: "+img_list[im_num]+'. Resnet oringial size maximum response is %.2f' % (out_list[minus_scale].max()))
    #plt.title('resnet oringial size maximum response is %.2f' % (out_list[len(out_list)/2].max()))
    plt.draw()
    plt.waitforbuttonpress(1)
    if softmax_flag:
        plt.savefig('./exp_dir/fish_localise/imgs/resnet50_fish_detection_softmax/' + img_list[im_num], bbox_inches='tight')
    else:
        plt.savefig('./exp_dir/fish_localise/imgs/resnet50_fish_detection/'+img_list[im_num], bbox_inches='tight')

print('Finish visualising')