import numpy as np
import skimage
import json
import hashlib
from math import floor
from bunch import Bunch
import scipy


float32 = 'float32'
floatX = float32

whales_4 = {
    'zoom_range': (1.0, 1.5),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}

magik_z = {
    'zoom_range': (1.0, 1.3),
    'rotation_range': (-8, 8),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}

crop1_buckets = {
    'zoom_range': (1.0 / 1.2, 1.2),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': True,
    'allow_stretch': False,
}


def my_portable_hash(l):
    class NumpyAwareEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return str(obj)
            return json.JSONEncoder(self, obj)
    s = json.dumps(l, cls=NumpyAwareEncoder)
    print 'before hash', s
    return hashlib.sha224(s).hexdigest()[:20]


def fetch_path_local(path):
    # This returns images in HxWxC format, dtyep = uint8 probably
    img = scipy.misc.imread(path)
    # for from keras.preprocessing import image, image will be 3* H 8 W

    if len(img.shape) == 2:
        # Some images are in grayscale
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = img.repeat(3, axis=2)

    if img.shape[2] > 3:
        # Some images have more than 3-channels. Doing same thing as fbcunn
        img = img[:, :, :3]
    return img


def unpack(a, *args):
    res = []
    for b in args:
        res.append(a[b])
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
    r = {'zoom_x': zoom_x, 'zoom_y': zoom_y,
         'rotation': rotation, 'shear': shear,
         'translation': translation, 'flip': flip}
    return Bunch(tform=build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip), r=r)


def build_center_uncenter_transforms2(width, height):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([width, height]) / 2.0
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    return tform_center, tform_uncenter


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    # TODO: why this -0.5 here?
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_centering_transform(image_shape, target_shape=(50, 50)):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def transformation(img, spec, perturb):
    # inter_size = spec['inter_size']
    mean = spec['mean']
    std = spec['std']
    img = img.astype(dtype=floatX)
    # img /= 255.0

    def apply_mean_std(img):
        if mean is not None:
            assert (len(mean) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] -= mean[channel]

        if std is not None:
            assert (len(std) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] /= std[channel]
        return img

    if perturb:
        img = perturb(img, spec['augmentation_params'], target_shape=(spec['target_h'], spec['target_w']))

    # imgs.append(img)
    # img = np.copy(img)

    # PCA
    if spec['pca_data'] is not None:
        evs, U = spec['pca_data']
        ls = evs.astype(float) * np.random.normal(scale=spec['pca_scale'], size=evs.shape[0])
        noise = U.dot(ls).reshape((1, 1, evs.shape[0]))
        # print evs, ls, U
        # print 'noise', noise
        img += noise

    img = apply_mean_std(img)

    def f(img):
        img = np.rollaxis(img, 2)
        return img

    # The img was H x W x C before
    return f(img)


def find_bucket(s, buckets, wsp):
    if wsp < 0 or wsp >= s:
        return -1
    res = int(floor((wsp * buckets) / s))
    assert (res >= 0 and res < buckets)
    return res



