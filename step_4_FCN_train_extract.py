import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy
from architecture.vgg_fcn import softmax
from architecture.resnet50_fcn import load_resnet50_model, get_model_resnet50_fcn_no_pooling,get_fcn_resnet50_last_layer_model
from architecture.post_processing import generate_attention_map, generate_boundingbox_from_response_map
from matplotlib.patches import Rectangle


# number of image scene classes
resnet50_data_mean = [103.939, 116.779, 123.68]
# now we test our fully convolutional network
#model = get_model_resnet50_fcn_no_pooling()
model_path = './exp_dir/fish_localise/training/fish_detection_resnet50_none_input.h5'
model = load_resnet50_model(model_path)

total_scale = 9
scale_prop = 1.1
minus_scale = total_scale/2
scale_list = [scale_prop ** (x) for x in range(total_scale)]
norm = colors.Normalize(vmin=0, vmax=1)

alb_directory = '/home/stevenwudi/Documents/Python_Project/Kaggle_The_Nature_Conversancy_Fisheries_Monitoring/train/ALB'
img_list = os.listdir(alb_directory)
start_num = 50
for im_name in img_list[start_num:start_num+50]:
    print("test image: "+im_name)
    img = load_img(os.path.join(alb_directory, im_name))  # this is a PIL image
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
        out = softmax(out)
        out_list.append(out[0, 0, :, :])

    # ########################## visualise the fish detection result
    # First row we plot the original image and final fused response maps
    # Second row we plot the smallest scale, original scale and largest scale response maps
    # we average the ouput
    max_list = [np.max(x) for x in out_list]
    resize_shape = [x for x in img_origin.size[::-1]]
    resized_response = [scipy.misc.imresize(x, resize_shape) * m for x, m in zip(out_list, max_list)]
    out_mean = np.mean(np.asarray(resized_response), axis=0)
    max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)

    attention_map = generate_attention_map(out_mean.shape)
    out_mean_attention = np.multiply(attention_map, out_mean)
    max_row_new, max_col_new = np.unravel_index(np.argmax(out_mean_attention), out_mean_attention.shape)

    # now from the response map we generate the bounding box.
    show_map, chosen_region, top, left, bottom, right = \
        generate_boundingbox_from_response_map(out_mean_attention,
                                           max_row_new, max_col_new)

    fusion_response = out_mean_attention[max_row_new, max_col_new] / 255.

    rect = Rectangle(
        xy=(left, top),
        width=right-left,
        height=bottom-top,
        facecolor='none',
        edgecolor='r',
        )

    #### visualisation
    plt.clf()
    plt.subplot(2, 3, 1)
    im = plt.imshow(out_mean * 1.0 / 255)
    im.set_norm(norm)
    plt.title('Scale response:  ' +
          str(['{:.3f}'.format(i) for i in [l[max_row_new, max_col_new] / 255. for l in resized_response]]) +
          '.\n  Corresponding: ' + str(['{:.3f}'.format(i) for i in scale_list]))

    plt.subplot(2, 3, 2)
    im = plt.imshow(out_mean_attention * 1.0 / 255)
    im.set_norm(norm)
    #plt.colorbar(ticks=np.linspace(0, 1.0, 10, endpoint=True))
    if fusion_response<0.3:
        plt.title('New maximum response is %.2f, NO FISH!!!' % (fusion_response))
    else:
        plt.title('New maximum response is %.2f' % (fusion_response))

    rect_axes_1 = plt.subplot(2, 3, 3)
    plt.imshow(img_origin)
    plt.scatter(x=[max_col], y=[max_row],  color='b', s=80, alpha=.5)
    plt.scatter(x=[max_col_new], y=[max_row_new], color='r', s=30, marker='^', alpha=1)
    rect_axes_1.add_patch(rect)
    plt.title("test image: " + im_name)

    # plot rectangle acquisition process
    plt.subplot(2, 3, 4)
    plt.imshow(show_map)

    plt.subplot(2, 3, 5)
    plt.imshow(chosen_region)

    plt.subplot(2, 3, 6)
    img = np.asarray(img_origin)
    plt.imshow(img[top:bottom, left:right, :])

    plt.draw()
    plt.waitforbuttonpress(1)
    plt.savefig('./exp_dir/fish_localise/imgs/resnet50_fish_detect_train/'+im_name, bbox_inches='tight')

print('Finish visualising')