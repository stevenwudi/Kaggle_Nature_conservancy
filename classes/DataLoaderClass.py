import traceback
import numpy as np
import skimage
from math import floor
from bunch import Bunch
from skimage.transform import warp, SimilarityTransform, AffineTransform, estimate_transform
from keras.preprocessing import image

from FishClass import FishClass

float32 = 'float32'
floatX = float32


def fetch_path_local(path):
    # This returns images in HxWxC format, dtyep = uint8 probably
    img = image.load_img(path)
    img = image.img_to_array(img)

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


def perturb(img, augmentation_params, target_shape=(50, 50), rng=np.random):
    assert(img.shape[2] < 10)
    w = (img.shape[0], img.shape[1])
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    tform_centering = build_centering_transform(w, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape[0:2])
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return warp(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant').astype('float32')


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


def fetch_rob_crop_recipe(recipe, global_spec):
    try:
        img = fetch_path_local(recipe.path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        pre_h, pre_w = 256, 256
        target_h, target_w = global_spec['target_h'], global_spec['target_w']
        TARGETS = global_spec['TARGETS']
        buckets = global_spec['buckets']

        sx = (pre_w - target_w + 1) / 2
        sy = (pre_h - target_h + 1) / 2
        #print sx, sy

        #target_h, target_w = 256, 256
        #print 'img_size', img.shape
        true_dist = np.zeros(shape=(FishClass.get_y_shape(TARGETS),), dtype=floatX)

        tform_res = AffineTransform(scale=(pre_w / float(img_w), pre_h / float(img_h)))
        tform_res += SimilarityTransform(translation=(-sx, -sy))

        tform_augment, r = unpack(random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
                                  'tform', 'r')

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_res += tform_center + tform_augment + tform_uncenter

        img = warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img = transformation(img, global_spec, perturb=False)

        if recipe.fetch_true_dist:
            # Constructing true_dist

            def go_indygo(name, a, b, v):
                inter = FishClass.get_interval(name, TARGETS)
                if a != -1 and b != -1 and inter is not None:
                    idx = inter[0] + a
                    if idx < inter[1]:
                        true_dist[idx] = v

            # slot
            slot_resize_scale = 0.25
            slot_annotation = recipe.annotations['slot'][0]
            ann1_x, ann1_y, ann_w, ann_h = (slot_annotation['x'] * slot_resize_scale,
                                          slot_annotation['y'] * slot_resize_scale,
                                          slot_annotation['width'] * slot_resize_scale,
                                          slot_annotation['height'] * slot_resize_scale)
            ann2_x = ann1_x + ann_w
            ann2_y = ann1_y + ann_h
            slot_mul = 1 / 4.0
            slot_1 = tform_res((ann1_x, ann1_y))[0]
            slot_2 = tform_res((ann2_x, ann2_y))[0]

            slot_bucket1_x = find_bucket(target_w, buckets, slot_1[0])
            slot_bucket1_y = find_bucket(target_h, buckets, slot_1[1])
            slot_bucket2_x = find_bucket(target_w, buckets, slot_2[0])
            slot_bucket2_y = find_bucket(target_h, buckets, slot_2[1])
            #print slot_1, slot_2
            #print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            #print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('slot_point1_x', slot_bucket1_x, slot_bucket1_y, slot_mul)
            go_indygo('slot_point1_y', slot_bucket1_y, slot_bucket1_x, slot_mul)
            go_indygo('slot_point2_x', slot_bucket2_x, slot_bucket2_y, slot_mul)
            go_indygo('slot_point2_y', slot_bucket2_y, slot_bucket2_x, slot_mul)

            # indygo
            indygo_resize_scale = 0.25
            point_1 = recipe.annotations['point1'][0]
            point_2 = recipe.annotations['point2'][0]
            point_1['x'] *= indygo_resize_scale
            point_1['y'] *= indygo_resize_scale


            point_2['x'] *= indygo_resize_scale
            point_2['y'] *= indygo_resize_scale

            indygo_mul = 1 / 4.0
            indygo_res1 = tform_res((point_1['x'], point_1['y']))[0]
            indygo_res2 = tform_res((point_2['x'], point_2['y']))[0]
            indygo_bucket1_x = find_bucket(target_w, buckets, indygo_res1[0])
            indygo_bucket1_y = find_bucket(target_h, buckets, indygo_res1[1])
            indygo_bucket2_x = find_bucket(target_w, buckets, indygo_res2[0])
            indygo_bucket2_y = find_bucket(target_h, buckets, indygo_res2[1])
            go_indygo('indygo_point1_x', indygo_bucket1_x, indygo_bucket1_y, indygo_mul)
            go_indygo('indygo_point1_y', indygo_bucket1_y, indygo_bucket1_x, indygo_mul)
            go_indygo('indygo_point2_x', indygo_bucket2_x, indygo_bucket2_y, indygo_mul)
            go_indygo('indygo_point2_y', indygo_bucket2_y, indygo_bucket2_x, indygo_mul)
            ##

        info = {
            'tform_res': tform_res,
            'r': r
            }

        #print 'img_shape', img.shape

        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
    except Exception as e:
        print traceback.format_exc()
        raise

