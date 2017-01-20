import numpy as np


def generate_attention_map(img_size, left_right_percentage=0.16, top_percentage=0.4, attenutate=0.1, plot=False):
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

