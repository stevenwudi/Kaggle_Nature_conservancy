import numpy as np
import sys


def generate_attention_map(img_size, left_right_percentage=0.16, top_percentage=0.3, attenutate=0.1, plot=False):
    """
    This method takes in an img_size and output a heuristic attention map
    Given the heuristic observation that
    (1) our fish detection network doest not take context into consideration
    (2) some detection will simply appear at the very left, top, right of the image
        which actually is never a true positive.
    :param default_percentage: The
    :param img_size:
    :return:
    """
    atttention_map = np.ones(img_size)
    left_right_margin = img_size[1] * left_right_percentage
    hanning_win = np.hanning(left_right_margin)
    left_hanning = ((hanning_win[:left_right_margin/2]) / (1/(1-attenutate))) + attenutate
    left_coeff = np.tile(left_hanning, (img_size[0], 1))

    right_hanning = ((hanning_win[left_right_margin/2:]) / (1/(1-attenutate))) + attenutate
    right_coeff = np.tile(right_hanning, (img_size[0], 1))

    top_margin = img_size[0] * top_percentage
    hanning_win = np.hanning(top_margin)
    top_hanning =((hanning_win[:top_margin/2]) / (1/(1-attenutate))) + attenutate
    top_coeff = np.tile(top_hanning, (img_size[1], 1))
    top_coeff = top_coeff.transpose(1,0)

    atttention_map[:, :left_coeff.shape[1]] *= left_coeff
    atttention_map[:, -right_coeff.shape[1]:] *= right_coeff
    atttention_map[:top_coeff.shape[0],:] *= top_coeff

    if plot:
        from matplotlib import pyplot as plt
        from matplotlib import colors
        plt.figure()
        im = plt.imshow(atttention_map)
        norm = colors.Normalize(vmin=0, vmax=1)
        im.set_norm(norm)
        plt.colorbar(ticks=np.linspace(0, 1.0, 10, endpoint=True))

    return atttention_map


def generate_boundingbox_from_response_map(out_mean_attention,
                                           max_row_new, max_col_new,
                                           mul_factor=0.4,
                                           expand_ratio=1.4):
    """

    :param out_mean_attention:
    :param max_row_new:
    :param max_col_new:
    :param mul_factor:  A coefficient for binary mask
    :param expand_ratio:  A ratio for width and height expand to have a complete fish
                            bounding box
    :return:
    """
    from skimage import measure
    response_binary = out_mean_attention > (out_mean_attention.max() * mul_factor)

    L, nums = measure.label(response_binary, return_num=True)
    #print("Number of components:", np.max(L))

    label = L[max_row_new, max_col_new]
    show_map = np.multiply(response_binary, L) * 255.0 / nums
    chosen_region = L==label

    # we then find the bounding box according to the chosen region
    cols = chosen_region.max(axis=0)
    left = np.asarray(np.nonzero(cols)).min()
    right = np.asarray(np.nonzero(cols)).max()
    rows = chosen_region.max(axis=1)
    top = np.asarray(np.nonzero(rows)).min()
    bottom = np.asarray(np.nonzero(rows)).max()

    center = [(right+left)/2., (bottom + top)/2.]
    width = int((right-left) * expand_ratio)
    height = int((bottom - top) * expand_ratio)

    left = int(max(0, center[0] - width/2.))
    top = int(max(0, center[1] - height/2.))
    right = int(min(L.shape[1], center[0] + width/2.))
    bottom = int(min(L.shape[0], center[1] + height/2.))

    return show_map, chosen_region, top, left, bottom, right

def generate_boundingbox_from_response_map_square(out_mean_attention,
                                           max_row_new, max_col_new,
                                           mul_factor=0.5, threshold=255*0.5,
                                           expand_ratio=1.4):
    """

    :param out_mean_attention:
    :param max_row_new:
    :param max_col_new:
    :param mul_factor:  A coefficient for binary mask
    :param expand_ratio:  A ratio for width and height expand to have a complete fish
                            bounding box
    :return:
    """
    from skimage import measure
    #response_binary = out_mean_attention > (out_mean_attention.max() * mul_factor)
    response_binary = out_mean_attention > threshold
    L, nums = measure.label(response_binary, return_num=True)
    #print("Number of components:", np.max(L))

    label = L[max_row_new, max_col_new]
    show_map = np.multiply(response_binary, L) * 255.0 / nums
    chosen_region = L==label

    # we then find the bounding box according to the chosen region
    cols = chosen_region.max(axis=0)
    left = np.asarray(np.nonzero(cols)).min()
    right = np.asarray(np.nonzero(cols)).max()
    rows = chosen_region.max(axis=1)
    top = np.asarray(np.nonzero(rows)).min()
    bottom = np.asarray(np.nonzero(rows)).max()

    center = [max_col_new, max_row_new]
    width = int((right-left) * expand_ratio)
    height = int((bottom - top) * expand_ratio)
    width = max(width, height)
    height = max(width, height)

    left = int(max(0, center[0] - width/2.))
    top = int(max(0, center[1] - height/2.))
    right = int(min(L.shape[1], center[0] + width/2.))
    bottom = int(min(L.shape[0], center[1] + height/2.))

    return show_map, chosen_region, top, left, bottom, right


def bb_intersection_over_union(rect_pred, rect_gt, detection_area_ratio=0.5):
    # Intersection over Union (IoU)
    # determine the (x, y)-coordinates of the intersection rectangle
    # if both of the boxes are empty, hence a NoF fish, we return iou==1
    if rect_pred._width==0 and rect_pred._height==0 and rect_gt._width == 0 and rect_gt._height == 0:
        return 1

    boxA = [rect_pred.xy[0], rect_pred.xy[1], rect_pred.xy[0] + rect_pred._width, rect_pred.xy[1] + rect_pred._height]
    boxB = [rect_gt.xy[0], rect_gt.xy[1], rect_gt.xy[0] + rect_gt._width, rect_gt.xy[1] + rect_gt._height]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, (xB - xA)) * max(0, (yB - yA))

    # compute the area of both the prediction and ground-truth
    # rectangles
    # Di Wu deliberately downplay the effect of detection box area
    # in the hope for grid search, we can have larger boundingbox that
    # can have a more complete fish
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) * detection_area_ratio
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()