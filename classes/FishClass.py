import sys
import os
import copy
import numpy as np
import random
from bunch import Bunch
import json
import time
from collections import defaultdict
import theano
import keras
import scipy

from classes.TrainerClass import BaseTrainer
from classes.SaverClass import ExperimentSaver, Tee, load_obj, Saver
import classes.DataLoaderClass
from classes.DataLoaderClass import unpack, floatX, fetch_path_local, random_perturbation_transform, \
    build_center_uncenter_transforms2, transformation, find_bucket

from classes.TrainerClass import ProcessFunc, MinibatchOutputDirector2, create_standard_iterator, repeat, epoch_header, \
    EpochData, elapsed_time_ms, elapsed_time_mins
import traceback
from skimage.transform import warp, SimilarityTransform, AffineTransform
from keras.preprocessing.image import load_img, img_to_array

from architecture.vgg_fcn import softmax
from architecture.post_processing import generate_attention_map, generate_boundingbox_from_response_map, \
    bb_intersection_over_union, printProgress

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


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
        # print sx, sy

        # target_h, target_w = 256, 256
        # print 'img_size', img.shape
        true_dist = np.zeros(shape=(FishClass.get_y_shape(TARGETS),), dtype=floatX)

        tform_res = AffineTransform(scale=(pre_w / float(img_w), pre_h / float(img_h)))
        tform_res += SimilarityTransform(translation=(-sx, -sy))

        tform_augment, r = unpack(
            random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
            'tform', 'r')

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_res += tform_center + tform_augment + tform_uncenter

        img_warp = warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        img_transformed = transformation(img_warp, global_spec, perturb=False)

        if recipe.fetch_true_dist:
            # Constructing true_dist

            def go_indygo(name, a, b, v):
                inter = FishClass.get_interval(name, TARGETS)
                if a != -1 and b != -1 and inter is not None:
                    idx = inter[0] + a
                    if idx < inter[1]:
                        true_dist[idx] = v

            # sloth annotation
            slot_annotation = recipe.annotations
            ann1_x, ann1_y, ann_w, ann_h = (slot_annotation['x'], slot_annotation['y'],
                                            slot_annotation['width'],
                                            slot_annotation['height'])
            ann2_x = ann1_x + ann_w
            ann2_y = ann1_y + ann_h
            slot_1 = tform_res((ann1_x, ann1_y))[0]
            slot_2 = tform_res((ann2_x, ann2_y))[0]

            sloth_bucket1_x = find_bucket(target_w, buckets, slot_1[0])
            sloth_bucket1_y = find_bucket(target_h, buckets, slot_1[1])
            sloth_bucket2_x = find_bucket(target_w, buckets, slot_2[0])
            sloth_bucket2_y = find_bucket(target_h, buckets, slot_2[1])
            # print slot_1, slot_2
            # print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            # print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('sloth_point1_x', sloth_bucket1_x, sloth_bucket1_y, 1)
            go_indygo('sloth_point1_y', sloth_bucket1_y, sloth_bucket1_x, 1)
            go_indygo('sloth_point2_x', sloth_bucket2_x, sloth_bucket2_y, 1)
            go_indygo('sloth_point2_y', sloth_bucket2_y, sloth_bucket2_x, 1)

            # indygo ??? What for?

        info = {
            'tform_res': tform_res,
            'r': r
        }

        # print 'img_shape', img.shape
        if global_spec['debug_plot']:
            from matplotlib import pyplot as plt
            from matplotlib.patches import Rectangle
            plt.figure()
            plt.clf()
            axes1 = plt.subplot(221)
            gt_rect = Rectangle(
                xy=(slot_annotation['x'], slot_annotation['y']),
                width=slot_annotation['width'],
                height=slot_annotation['height'],
                facecolor='none',
                edgecolor='r',
            )
            axes1.add_patch(gt_rect)
            plt.imshow(img)
            plt.title('original image')

            axes2 = plt.subplot(222)
            if r['flip']:
                top_x = slot_2[0]
            else:
                top_x = slot_1[0]
            gt_rect_warp = Rectangle(
                xy=(top_x, slot_1[1]),
                width=(np.abs(slot_2[0] - slot_1[0])),
                height=(np.abs(slot_2[1] - slot_1[1])),
                facecolor='none',
                edgecolor='r',
            )
            axes2.add_patch(gt_rect_warp)
            plt.imshow(img_warp)
            plt.title('transformed image')

            axes2 = plt.subplot(223)
            if r['flip']:
                top_x = slot_2[0]
            else:
                top_x = slot_1[0]
            gt_rect_warp = Rectangle(
                xy=(top_x, slot_1[1]),
                width=(np.abs(slot_2[0] - slot_1[0])),
                height=(np.abs(slot_2[1] - slot_1[1])),
                facecolor='none',
                edgecolor='r',
            )
            axes2.add_patch(gt_rect_warp)
            plt.imshow(img_warp)

            plt.rc('grid', linestyle="-", color='black')
            plt.grid(True)
            plt.xticks([i * target_w / buckets for i in range(0, buckets)])
            plt.yticks([i * target_h / buckets for i in range(0, buckets)])

            ratio_bucket = 1.0 * target_w / buckets
            if r['flip']:
                points = [(slot_bucket2_x * ratio_bucket, slot_bucket1_y * ratio_bucket),
                          (slot_bucket1_x * ratio_bucket, slot_bucket2_y * ratio_bucket)]
            else:
                points = [(slot_bucket1_x * ratio_bucket, slot_bucket1_y * ratio_bucket),
                          (slot_bucket2_x * ratio_bucket, slot_bucket2_y * ratio_bucket)]
            x = map(lambda x: x[0], points)
            y = map(lambda x: x[1], points)
            plt.scatter(x, y, color='r', s=20)
            plt.title('transformed image')

            plt.show()

        return Bunch(x=img_transformed, y=true_dist, recipe=recipe, info=info)

    except Exception as e:
        print traceback.format_exc()
        raise


def vgg_keras_recipe(recipe, global_spec):
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
        # print sx, sy

        # target_h, target_w = 256, 256
        # print 'img_size', img.shape
        true_dist = np.zeros(shape=(FishClass.get_y_shape(TARGETS),), dtype=floatX)

        tform_res = AffineTransform(scale=(pre_w / float(img_w), pre_h / float(img_h)))
        tform_res += SimilarityTransform(translation=(-sx, -sy))

        tform_augment, r = unpack(
            random_perturbation_transform(rng=np.random, **global_spec['augmentation_params']),
            'tform', 'r')

        tform_center, tform_uncenter = build_center_uncenter_transforms2(target_w, target_h)
        tform_res += tform_center + tform_augment + tform_uncenter
        img_warp = warp(img, AffineTransform(tform_res._inv_matrix), output_shape=(target_h, target_w))
        # set the image to keras formality
        img_warp = img_warp * 255.
        # 'RGB'->'BGR'
        img_warp = img_warp[:, :, ::-1]
        # Zero-center by mean pixel
        img_warp[:, :, 0] -= 103.939
        img_warp[:, :, 1] -= 116.779
        img_warp[:, :, 2] -= 123.68
        img_transformed = np.rollaxis(img_warp, 2)

        if recipe.fetch_true_dist:
            # Constructing true_dist

            def go_indygo(name, a, b, v):
                inter = FishClass.get_interval(name, TARGETS)
                if a != -1 and b != -1 and inter is not None:
                    idx = inter[0] + a
                    if idx < inter[1]:
                        true_dist[idx] = v

            # sloth annotation
            slot_annotation = recipe.annotations
            ann1_x, ann1_y, ann_w, ann_h = (slot_annotation['x'], slot_annotation['y'],
                                            slot_annotation['width'],
                                            slot_annotation['height'])
            ann2_x = ann1_x + ann_w
            ann2_y = ann1_y + ann_h
            slot_1 = tform_res((ann1_x, ann1_y))[0]
            slot_2 = tform_res((ann2_x, ann2_y))[0]

            sloth_bucket1_x = find_bucket(target_w, buckets, slot_1[0])
            sloth_bucket1_y = find_bucket(target_h, buckets, slot_1[1])
            sloth_bucket2_x = find_bucket(target_w, buckets, slot_2[0])
            sloth_bucket2_y = find_bucket(target_h, buckets, slot_2[1])
            # print slot_1, slot_2
            # print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            # print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('sloth_point1_x', sloth_bucket1_x, sloth_bucket1_y, 1)
            go_indygo('sloth_point1_y', sloth_bucket1_y, sloth_bucket1_x, 1)
            go_indygo('sloth_point2_x', sloth_bucket2_x, sloth_bucket2_y, 1)
            go_indygo('sloth_point2_y', sloth_bucket2_y, sloth_bucket2_x, 1)

            # indygo ??? What for?

        info = {
            'tform_res': tform_res,
            'r': r
        }

        # print 'img_shape', img.shape
        if global_spec['debug_plot']:
            from matplotlib import pyplot as plt
            from matplotlib.patches import Rectangle
            plt.figure()
            plt.clf()
            axes1 = plt.subplot(221)
            gt_rect = Rectangle(
                xy=(slot_annotation['x'], slot_annotation['y']),
                width=slot_annotation['width'],
                height=slot_annotation['height'],
                facecolor='none',
                edgecolor='r',
            )
            axes1.add_patch(gt_rect)
            plt.imshow(img)
            plt.title('original image')

            axes2 = plt.subplot(222)
            if r['flip']:
                top_x = slot_2[0]
            else:
                top_x = slot_1[0]
            gt_rect_warp = Rectangle(
                xy=(top_x, slot_1[1]),
                width=(np.abs(slot_2[0] - slot_1[0])),
                height=(np.abs(slot_2[1] - slot_1[1])),
                facecolor='none',
                edgecolor='r',
            )
            axes2.add_patch(gt_rect_warp)
            plt.imshow(img_warp)
            plt.title('transformed image')

            axes2 = plt.subplot(223)
            if r['flip']:
                top_x = slot_2[0]
            else:
                top_x = slot_1[0]
            gt_rect_warp = Rectangle(
                xy=(top_x, slot_1[1]),
                width=(np.abs(slot_2[0] - slot_1[0])),
                height=(np.abs(slot_2[1] - slot_1[1])),
                facecolor='none',
                edgecolor='r',
            )
            axes2.add_patch(gt_rect_warp)
            plt.imshow(img_warp)

            plt.rc('grid', linestyle="-", color='black')
            plt.grid(True)
            plt.xticks([i * target_w / buckets for i in range(0, buckets)])
            plt.yticks([i * target_h / buckets for i in range(0, buckets)])

            ratio_bucket = 1.0 * target_w / buckets
            if r['flip']:
                points = [(slot_bucket2_x * ratio_bucket, slot_bucket1_y * ratio_bucket),
                          (slot_bucket1_x * ratio_bucket, slot_bucket2_y * ratio_bucket)]
            else:
                points = [(slot_bucket1_x * ratio_bucket, slot_bucket1_y * ratio_bucket),
                          (slot_bucket2_x * ratio_bucket, slot_bucket2_y * ratio_bucket)]
            x = map(lambda x: x[0], points)
            y = map(lambda x: x[1], points)
            plt.scatter(x, y, color='r', s=20)
            plt.title('transformed image')

            plt.show()

        return Bunch(x=img_transformed, y=true_dist, recipe=recipe, info=info)

    except Exception as e:
        print traceback.format_exc()
        raise


class FishClass(BaseTrainer):
    @classmethod
    def norm_name(cls, key):
        if key[-4:] == '.jpg':
            key = key[-13:-4]
        return key

    @classmethod
    def get_interval(cls, name, TARGETS):
        sum = 0
        for (a, b) in TARGETS:
            if a == name:
                return (sum, sum + b)
            sum += b
        return None

    @classmethod
    def get_target_suffixes(cls, TARGETS):
        return map(lambda a: a[0], TARGETS)

    def initialize(self):
        # type: () -> object
        np.random.seed(None)
        seed = np.random.randint(0, 1000000000)
        if self.args.seed is not None:
            seed = self.args.seed
        self.exp.set_seed(seed)
        print 'Seed', seed
        self.seed = seed
        self.numpy_rng = np.random.RandomState(seed)

        if self.args.valid_seed is None:
            self.valid_seed = random.randint(0, 10000)
        else:
            self.valid_seed = self.args.valid_seed

        self.exp.set_valid_seed(self.valid_seed)

        self.global_saver = Saver(self.args.global_saver_path)

        if self.exp_dir_path:
            self.saver = ExperimentSaver(self.exp_dir_path)
            a = self.saver.open_file(None, self.args.log_name)
            self.log_file, filepath = a.file, a.filepath
            self.tee_stdout = Tee(sys.stdout, self.log_file)
            self.tee_stderr = Tee(sys.stderr, self.log_file)

            sys.stdout = self.tee_stdout
            sys.stderr = self.tee_stderr

    def set_targets(self):
        MANY_TARGETS = {
            'final':
                [
                    ('class', 7),
                    ('new_conn', 2),
                ],
            'final_no_conn':
                [
                    ('class', 7),
                ],
            'crop1':
                [
                    ('sloth_point1_x', self.args.buckets),
                    ('sloth_point1_y', self.args.buckets),
                    ('sloth_point2_x', self.args.buckets),
                    ('sloth_point2_y', self.args.buckets),
                ],
            'crop2':
                [
                    ('class', 7),
                    ('conn', 2),
                    ('indygo_point1_x', self.args.buckets),
                    ('indygo_point1_y', self.args.buckets),
                    ('indygo_point2_x', self.args.buckets),
                    ('indygo_point2_y', self.args.buckets),
                ]
        }
        self.TARGETS = MANY_TARGETS[self.args.target_name]
        print self.get_intervals(self.TARGETS)
        self.Y_SHAPE = sum(self.get_n_outs(self.TARGETS))

    def construct_model(self, arch_name, **kwargs):
        module_name = 'architecture.' + arch_name
        mod = __import__(module_name, fromlist=[''])
        return getattr(mod, 'create_model')(**kwargs)

    def init_model(self):
        if self.args.load_arch_path is not None:
            print('..loading arch and model')
            self.model = keras.models.load_model(self.args.load_arch_path)
            # this is a bad hack
            # self.model.optimizer.lr.set_value(0.1)
        else:
            params = {
                'channels': self.args.channels,
                'image_size': (self.args.crop_h, self.args.crop_w),
                'n_outs': self.get_n_outs(self.TARGETS),
                'conv_l2_reg': self.args.conv_l2_reg,
                'fc_l2_reg': self.args.fc_l2_reg,
                'dropout': self.args.dropout,
                'dropout_coeff': self.args.dropout_coeff,
                'glr_decay': self.args.glr_decay,
                'momentum': self.args.momentum,
                'glr': self.args.glr,
                'trainable': self.args.trainable,
            }
            print 'Set up new model'
            self.model = self.construct_model(self.args.arch, **params)

            print 'Saving arch'
            self.exp.set_nof_params(self.model.count_params())
            self.exp.set_weights_desc(self.model.summary())

            if self.args.load_params_path is not None:
                print '..loading params', self.args.load_params_path
                self.model.load_state_new(self.args.load_params_path)

    def go(self, exp, args, exp_dir_path):

        self.args = args
        self.exp = exp
        self.n_classes = self.args.n_classes
        self.exp_dir_path = exp_dir_path
        print(' '.join(sys.argv))
        print('ARGS', args)
        exp.set_name(self.args.name)
        # initialise the seeding process
        self.initialize()

        # set the model targets
        self.set_targets()
        self.ts = self.create_timeseries_and_figures()
        # initialise the model! Important step
        if self.args.init_model:
            self.init_model()
        self.process_recipe = getattr(classes.FishClass, self.args.process_recipe_name)
        self.do_training()

        return 0

    def read_annotations(self):

        annotations = defaultdict(dict)
        fish_folders = ['OTHER', 'ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']

        def f_annotation_largest(l, anno_name, fld, fn):
            '''
            We read the boundingbox that contains the largest area.
            The reason is that since there is only one fish type on the boat, we can temporally assume that
            there is only one fish on the boat.
            TODO: try DetectNet; Deep Neural Network for Object Detection in DIGITS
            https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/
            :param l:
            :param anno_name:
            :return:
            '''
            count = 0
            for el in l:
                if 'filename' in el:
                    key = el['filename']
                key = self.norm_name(self.norm_name(key))
                key += '.jpg'

                if 'annotations' not in el or len(el['annotations']) < 1:
                    print("No annotation for " + fld + '/' + key)
                else:
                    # if there is more than one labeled fish on the boat
                    fish_idx = 0
                    if len(el['annotations']) > 1:
                        area_array = np.zeros(len(el['annotations']))
                        for i in range(len(el['annotations'])):
                            area_array[i] = el['annotations'][i]['height'] * el['annotations'][i]['width']
                        fish_idx = np.argmax(area_array)

                    annotations[key][anno_name] = el['annotations'][fish_idx]
                    annotations[key][anno_name]['fish_class'] = fn
                    count += 1
            return count

        def clean_anno(l):
            for i, el in enumerate(l):
                if 'filename' in el:
                    key = el['filename']
                key = self.norm_name(self.norm_name(key))
                l[i]['filename'] = key + '.jpg'
            return l

        if self.args.sloth_annotations_url is not None:
            for fn, fld in enumerate(fish_folders):
                json_path = os.path.join(self.args.sloth_annotations_url, fld.lower() + "_labels.json")
                sloth_annotations_list = json.load(open(json_path, 'r'))
                count = f_annotation_largest(sloth_annotations_list, 'sloth', fld, fn)
                print('loading class ' + fld + ', number of annotation is: ' + str(count))

                ## we comment the section below which is used to clean the annoation
                # l = clean_anno(sloth_annotations_list)
                # with open(os.path.join('../clean_boundingbox', fld.lower()+"_labels.json"), 'w') as outfile:
                #     json.dump(l, outfile)

        print("Finish loading images, total number of images is: " + str(len(annotations)))
        return annotations

    def collect_training_validation_images(self):
        # for every training images, we collect tagged fish with 4 negative background images
        # that does not overlap with any tagged images
        train_dir = os.path.join(self.exp_dir_path, 'train')
        valid_dir = os.path.join(self.exp_dir_path, 'valid')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        dir_list_name = self.strclass_to_class_idx.keys()
        dir_list_name.append('BACKGROUND')
        for k in dir_list_name:
            if not os.path.exists(os.path.join(train_dir, k)):
                os.mkdir(os.path.join(train_dir, k))

        if not os.path.exists(valid_dir):
            os.mkdir(valid_dir)
        for k in dir_list_name:
            if not os.path.exists(os.path.join(valid_dir, k)):
                os.mkdir(os.path.join(valid_dir, k))

        annotations = defaultdict(dict)

        def f_annotation_all(l, fld):
            '''
            We read the boundingbox that contains the all the annotated area.
            :param l:
            :param anno_name:
            :return:
            '''
            count = 0
            for el in l:
                if 'filename' in el:
                    key = el['filename']
                key = self.norm_name(self.norm_name(key))
                key += '.jpg'

                if 'annotations' not in el or len(el['annotations']) < 1:
                    print("No annotation for " + fld + '/' + key)
                else:
                    # if there is more than one labeled fish on the boat
                    if len(el['annotations']) > 1:
                        print("More than one annotation for " + fld + '/' + key)
                        count += len(el['annotations'])
                    else:
                        count += 1
                    annotations[key] = el['annotations']

            return count

        count_all = 0
        if self.args.sloth_annotations_url is not None:
            for fn, fld in enumerate(self.strclass_to_class_idx.keys()):
                json_path = os.path.join(self.args.sloth_annotations_url, fld.lower() + "_labels.json")
                sloth_annotations_list = json.load(open(json_path, 'r'))
                count = f_annotation_all(sloth_annotations_list, fld)
                count_all += count
                print('loading class ' + fld + ', number of annotation is: ' + str(count))
        print("Finish loading annotation, total number of images is: " + str(count_all))

        for recipe in self.train_recipes:
            self.save_crop_image(recipe=recipe, annotations=annotations, dir=train_dir)

        for recipe in self.valid_recipes:
            self.save_crop_image(recipe=recipe, annotations=annotations, dir=valid_dir)

        return 1

    def save_crop_image(self, recipe, annotations, dir):
        from collections import namedtuple
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        def area(a, b):  # returns None if rectangles don't intersect
            dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
            dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
            if (dx >= 0) and (dy >= 0):
                return dx * dy

        img = fetch_path_local(recipe.path)
        rect_list = []
        height_list = []
        width_list = []
        for i, anno in enumerate(annotations[recipe.name]):
            # some annotations are outside the image
            anno['x'] = min(max(0, anno['x']), img.shape[1])
            anno['y'] = min(max(0, anno['y']), img.shape[0])
            img_fish = img[int(anno['y']):int(anno['y'] + anno['height']),
                       int(anno['x']):int(anno['x'] + anno['width']), :]
            img_class = self.class_idx_to_strclass[recipe.annotations['fish_class']]
            img_path_name = os.path.join(dir, img_class, recipe.name[:-4] + "_" + str(i) + '.jpg')
            scipy.misc.imsave(img_path_name, img_fish)
            rect_list.append(Rectangle(int(anno['y']), int(anno['x']),
                                       int(anno['y'] + anno['height']), int(anno['x'] + anno['width'])))
            height_list.append(anno['height'])
            width_list.append(anno['width'])
        # now generating negative examples, negetive example size is the mean of positive
        # need to consider the random seed
        rng = random.Random()
        rng.seed(self.valid_seed)
        height_mean = np.mean(height_list)
        width_mean = np.mean(width_list)
        neg_rect_count = 0
        try_count = 0

        while neg_rect_count < self.args.collect_training_pos_neg_ratio:
            try_count += 1
            if try_count > 10000:
                print("fail to generate negative example for image %s" % recipe.path)
                break
            # generate negative examples:
            # if the positive examples are very large...:
            if int(height_mean) > int(img.shape[0] - height_mean) or int(width_mean) > int(img.shape[1] - width_mean):
                print(
                "fail to generate negative example for image %s because the positive image is too large." % recipe.path)
                break
            xmin = rng.randint(int(height_mean), int(img.shape[0] - height_mean))
            ymin = rng.randint(int(width_mean), int(img.shape[1] - width_mean))
            rect_negative = Rectangle(xmin, ymin, xmin + height_mean, ymin + width_mean)

            overlap = False
            for rect in rect_list:
                # if there is overlap
                if area(rect, rect_negative):
                    overlap = True

            if overlap:
                continue
            else:
                img_background = img[int(rect_negative.xmin):int(rect_negative.xmax),
                                 int(rect_negative.ymin):int(rect_negative.ymax), :]
                img_class = self.class_idx_to_strclass[recipe.annotations['fish_class']]
                img_path_name = os.path.join(dir, 'BACKGROUND',
                                             img_class + "_" + recipe.name[:-4] + "_" + str(neg_rect_count) + '.jpg')
                scipy.misc.imsave(img_path_name, img_background)
                neg_rect_count += 1

    def collect_training_validation_stats(self):
        from distutils.dir_util import copy_tree
        import scipy
        # for every training images, we collect tagged fish with 4 negative background images
        # that does not overlap with any tagged images
        train_dir = os.path.join(self.exp_dir_path, 'train_binary')
        valid_dir = os.path.join(self.exp_dir_path, 'valid_binary')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(valid_dir):
            os.mkdir(valid_dir)
        # we create another folder to collect all positive fish images, it's simply redundant, but makes the keras
        # interface much easier to implement....
        if not os.path.exists(os.path.join(train_dir, 'ALL')):
            os.mkdir(os.path.join(train_dir, 'ALL'))
        if not os.path.exists(os.path.join(valid_dir, 'ALL')):
            os.mkdir(os.path.join(valid_dir, 'ALL'))
        if not os.path.exists(os.path.join(train_dir, 'BACKGROUND')):
            os.mkdir(os.path.join(train_dir, 'BACKGROUND'))
        if not os.path.exists(os.path.join(valid_dir, 'BACKGROUND')):
            os.mkdir(os.path.join(valid_dir, 'BACKGROUND'))

        dir_list_name = self.strclass_to_class_idx.keys()

        h = classes.DataLoaderClass.my_portable_hash(
            [os.listdir(os.path.join(train_dir, 'ALL')), os.listdir(os.path.join(train_dir, 'BACKGROUND'))])
        name = 'mean_shape_{}'.format(h)
        print 'mean_shape_ filename', name
        res = self.global_saver.load_obj(name)
        if res is None or self.args.invalid_cache:
            print '..recomputing mean, shape'

            # copy subdirectory example
            toDirectory = os.path.join(train_dir, 'ALL')
            for k in dir_list_name:
                fromDirectory = os.path.join(self.exp_dir_path, 'train', k)
                copy_tree(fromDirectory, toDirectory)

            # copy background
            fromDirectory = os.path.join(self.exp_dir_path, 'train', 'BACKGROUND')
            toDirectory = os.path.join(train_dir, 'BACKGROUND')
            copy_tree(fromDirectory, toDirectory)

            toDirectory = os.path.join(valid_dir, 'ALL')
            for k in dir_list_name:
                fromDirectory = os.path.join(self.exp_dir_path, 'valid', k)
                copy_tree(fromDirectory, toDirectory)

            # copy background
            fromDirectory = os.path.join(self.exp_dir_path, 'valid', 'BACKGROUND')
            toDirectory = os.path.join(valid_dir, 'BACKGROUND')
            copy_tree(fromDirectory, toDirectory)

            img_shap_list = []
            train_mean = np.array([0, 0, 0], dtype=np.float32)
            total_img_num = len(os.listdir(os.path.join(train_dir, 'ALL'))) + len(
                os.listdir(os.path.join(train_dir, 'BACKGROUND')))

            for im_name in os.listdir(os.path.join(train_dir, 'ALL')):
                img = scipy.misc.imread(os.path.join(train_dir, 'ALL', im_name))
                img_shap_list.append(img.shape[:2])
                train_mean += img.mean(axis=0).mean(axis=0) / total_img_num

            for im_name in os.listdir(os.path.join(train_dir, 'BACKGROUND')):
                img = scipy.misc.imread(os.path.join(train_dir, 'BACKGROUND', im_name))
                img_shap_list.append(img.shape[:2])
                train_mean += img.mean(axis=0).mean(axis=0) / total_img_num

            # average shape is 149 * 211 -- we can save it's a square of 180
            img_shape_mean = int(np.asarray(img_shap_list).mean())

            print('train_mean: ' + str(train_mean) + ' img_shape_mean: ' + str(img_shape_mean))
            mean_data_path = self.global_saver.save_obj((train_mean, img_shape_mean), name)
            self.exp.set_mean_data_url(mean_data_path)

        return 1

    def create_recipes_old(self, valid_seed, train_part=0.9):
        rng = random.Random()
        rng.seed(valid_seed)
        recipes = []
        fish_ids = list(self.annotations.keys())
        count = defaultdict(int)
        self.strclass_to_class_idx = {'OTHER': 0, 'ALB': 1, 'BET': 2, 'DOL': 3, 'LAG': 4, 'SHARK': 5, 'YFT': 6}
        self.class_idx_to_strclass = {v: k for k, v in self.strclass_to_class_idx.iteritems()}

        for fish_id in fish_ids:
            fish_class = self.annotations[fish_id]['sloth']['fish_class']
            count[fish_class] += 1
            class_name = self.class_idx_to_strclass[fish_class]
            annotations = self.annotations[fish_id]['sloth']
            recipe = Bunch(name=fish_id,
                           path=os.path.join(self.args.train_dir_path, class_name, fish_id),
                           annotations=annotations,
                           class_idx=fish_class,
                           fetch_true_dist=True
                           )
            recipes.append(recipe)

        rng.shuffle(recipes)
        n = len(recipes)
        split_point = int(train_part * n)
        train_recipes = recipes[:split_point]
        valid_recipes = recipes[split_point:]

        print 'VALID EXAMPLES'
        print valid_recipes[0]
        print('it should be name: img_04238.jpg because of the valid seed is 1000! Important!!!')
        print('it should be name: img_04238.jpg because of the valid seed is 1000! Important!!!')

        return Bunch(train_recipes=train_recipes,
                     valid_recipes=valid_recipes,
                     strclass_to_class_idx=self.strclass_to_class_idx)

    def create_iterator_factory(self):
        train_spec = Bunch({
            'equalize': self.args.equalize,
            'indygo_equalize': self.args.indygo_equalize,
            'target_h': self.args.crop_h,
            'target_w': self.args.crop_w,
            'target_channels': self.args.channels,
            'cropping_type': 'random',
            'mean': None,
            'std': None,
            'pca_data': None,
            'augmentation_params': getattr(classes.DataLoaderClass, self.args.aug_params),
            'margin': self.args.margin,
            'diag': self.args.diag,
            'buckets': self.args.buckets,
            'TARGETS': self.TARGETS,
            'debug_plot': self.args.debug_plot,
        })

        train_recipes, strclass_to_class_idx = unpack(self.create_recipes_old(valid_seed=self.valid_seed,
                                                                              train_part=self.args.train_part),
                                                      'train_recipes', 'strclass_to_class_idx')
        print len(train_recipes)

        SAMPLING_SPLIT = 200
        if self.args.do_pca:
            if self.args.pca_data_path is None:
                print 'train_spec', train_spec
                h = classes.DataLoaderClass.my_portable_hash([train_spec, SAMPLING_SPLIT])
                name = 'pca_data_{}'.format(h)
                print 'pca_data filename', name
                pca_data = self.global_saver.load_obj(name)
                if pca_data is None or self.args.invalid_cache:
                    print '..recomputing pca_data'
                    pca_data = self.pca_it(train_spec, train_recipes[0:SAMPLING_SPLIT], self.process_recipe)
                    pca_data_path = self.global_saver.save_obj(pca_data, name)
                    self.exp.set_pca_data_url(pca_data_path)

                else:
                    print '..using old pca_data'
            else:
                pca_data = load_obj(self.args.pca_data_path)
            print 'pca_data', pca_data

        if self.args.do_mean:
            mean, std = self.get_mean_std(train_spec, train_recipes[SAMPLING_SPLIT:2 * SAMPLING_SPLIT],
                                          train_recipes[2 * SAMPLING_SPLIT: 3 * SAMPLING_SPLIT], )
            self.mean = mean
            self.std = std
            print 'MEAN', mean, 'STD', std
            train_spec['mean'] = mean
            train_spec['std'] = std
        else:
            self.mean = np.zeros((3,), dtype=floatX)
            self.std = np.ones((3,), dtype=floatX)

        valid_spec = copy.copy(train_spec)
        test_spec = copy.copy(train_spec)
        process_recipe = self.process_recipe
        Y_SHAPE = self.Y_SHAPE

        class IteratorFactory(object):
            TEST_SPEC = test_spec
            TRAIN_SPEC = train_spec
            VALID_SPEC = valid_spec

            def get_strclass_to_class_idx(self):
                return strclass_to_class_idx

            def get_iterator(self, recipes, mb_size, spec, buffer_size=15, pool_size=5, chunk_mul=3,
                             output_partial_batches=False):
                print 'Create iterator!!!! pool_size = ', pool_size
                process_func = ProcessFunc(process_recipe, spec)
                output_director = MinibatchOutputDirector2(mb_size,
                                                           x_shape=(
                                                           spec['target_channels'], spec['target_h'], spec['target_w']),
                                                           y_shape=(Y_SHAPE,),
                                                           output_partial_batches=output_partial_batches)

                return create_standard_iterator(
                    process_func,
                    recipes,
                    output_director,
                    pool_size=pool_size,
                    buffer_size=buffer_size,
                    chunk_size=chunk_mul * mb_size)

            def get_train_iterator(self, train_recipes, mb_size, pool_size=6, buffer_size=45):
                print 'Create train iterator!!!! pool_size = ', pool_size
                return self.get_iterator(train_recipes, mb_size, self.TRAIN_SPEC, pool_size=pool_size,
                                         buffer_size=buffer_size, chunk_mul=2)

            def get_valid_iterator(self, valid_recipes, mb_size, n_samples_valid, pool_size=6, buffer_size=30,
                                   real_valid_shuffle=False):
                valid_recipes_repeated = repeat(valid_recipes, n_samples_valid, mb_size)
                if real_valid_shuffle:
                    random.shuffle(valid_recipes_repeated)
                return self.get_iterator(valid_recipes_repeated, mb_size, self.VALID_SPEC, pool_size=pool_size,
                                         buffer_size=buffer_size)

            def get_test_iterator(self, test_recipes, mb_size, n_samples_test, buffer_size=5, real_test_shuffle=False,
                                  pool_size=5):
                test_recipes_repeated = repeat(test_recipes, n_samples_test, mb_size)
                if real_test_shuffle:
                    random.shuffle(test_recipes_repeated)
                print len(test_recipes_repeated)

                return self.get_iterator(test_recipes_repeated, mb_size, self.TEST_SPEC, buffer_size=buffer_size,
                                         output_partial_batches=True, pool_size=pool_size)

            def get_test_iterator2(self, test_recipes, mb_size, buffer_size=5, pool_size=5):
                return self.get_iterator(test_recipes, mb_size, self.TEST_SPEC, buffer_size=buffer_size,
                                         output_partial_batches=True, pool_size=pool_size)

        return IteratorFactory()

    def fish_redetection(self):

        all_img_dir = os.path.join(self.exp_dir_path, 'imgs/resnet50_fish_detect_train')
        response_img_dir = os.path.join(self.exp_dir_path, 'response_maps')
        if not os.path.exists(response_img_dir):
            os.mkdir(response_img_dir)
        # some meta parameters
        self.resnet50_data_mean = [103.939, 116.779, 123.68]
        self.total_scale = 9
        self.scale_prop = 1.1
        self.scale_list = [self.scale_prop ** (x) for x in range(self.total_scale)]
        self.color_norm = colors.Normalize(vmin=0, vmax=1)

        time_list = []
        iou_list = []
        for i, recipe in enumerate(self.train_recipes):
            time_start_batch = time.time()
            iou_list.append(self.extract_salient_fish(recipe=recipe, all_img_dir=all_img_dir, response_img_dir=response_img_dir))
            time_list.append(time.time() - time_start_batch)
            eta = len(self.train_recipes) * np.array(time_list).mean()
            printProgress(i, len(self.train_recipes+self.valid_recipes), prefix='Progress:',
                          suffix='IOU error: %0.5f, ETA: %0.2f sec.' % (np.array(iou_list).mean(), eta),
                          barLength=50)

        for i, recipe in enumerate(self.valid_recipes):
            self.extract_salient_fish(recipe=recipe, all_img_dir=all_img_dir, response_img_dir=response_img_dir)
            time_start_batch = time.time()
            iou_list.append(
                self.extract_salient_fish(recipe=recipe, all_img_dir=all_img_dir, response_img_dir=response_img_dir))
            time_list.append(time.time() - time_start_batch)
            eta = len(self.valid_recipes) * np.array(time_list).mean()
            printProgress(i, len(self.train_recipes+self.valid_recipes), prefix='Progress:',
                          suffix='IOU: %0.5f, ETA: %0.2f sec.' % (np.array(iou_list).mean(), eta),
                          barLength=50)

        ### create a pseudo recipe for NoF class... and feed it into fish detection pipeline
        return True

    def extract_salient_fish(self, recipe,
                             all_img_dir='imgs/resnet50_fish_detect_train',
                             response_img_dir='response_maps'):
        img = load_img(recipe.path)
        img_origin = img.copy()
        # we sample different scale
        out_list = []
        for i in range(self.total_scale):
            basewidth = int(float(img_origin.size[0]) * self.scale_list[i])
            hsize = int((float(img_origin.size[1]) * self.scale_list[i]))
            img = img.resize((basewidth, hsize))
            x = img_to_array(img)  #
            #print("test image is of shape: " + str(x.shape))  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)
            x_new = x - np.asarray(self.resnet50_data_mean)[None, :, None, None]
            # predict the model output
            out = self.model.predict(x_new)
            #print(out.shape)
            out = softmax(out)
            out_list.append(out[0, 0, :, :])

        # ########################## visualise the fish detection result
        # max_list store the maximum response for different scales
        max_list = [np.max(x) for x in out_list]
        resize_shape = [x for x in img_origin.size[::-1]]
        resized_response = [scipy.misc.imresize(x, resize_shape) * m for x, m in zip(out_list, max_list)]
        out_mean = np.mean(np.asarray(resized_response), axis=0)
        max_row, max_col = np.unravel_index(np.argmax(out_mean), out_mean.shape)

        # attention map add heuristic low confidence near the boundaries
        attention_map = generate_attention_map(out_mean.shape)
        out_mean_attention = np.multiply(attention_map, out_mean)
        max_row_new, max_col_new = np.unravel_index(np.argmax(out_mean_attention), out_mean_attention.shape)

        # now from the response map we generate the bounding box.
        show_map, chosen_region, top, left, bottom, right = \
            generate_boundingbox_from_response_map(out_mean_attention,
                                                   max_row_new, max_col_new)

        fusion_response = out_mean_attention[max_row_new, max_col_new] / 255.

        rect_pred = Rectangle(
            xy=(left, top),
            width=right - left,
            height=bottom - top,
            facecolor='none',
            edgecolor='r',
        )

        # We get the ground truth from the annotation using sloth
        img = np.asarray(img_origin)
        for i, anno in enumerate([self.annotations[recipe.name]['sloth']]):
            # some annotations are outside the image
            anno['x'] = min(max(0, anno['x']), img.shape[1])
            anno['y'] = min(max(0, anno['y']), img.shape[0])
            img_class = self.class_idx_to_strclass[recipe.annotations['fish_class']]
            rect_gt = Rectangle(xy=(int(anno['x']), int(anno['y'])),
                                width=int(anno['width']),
                                height=int(anno['height']),
                                facecolor='none',
                                edgecolor='g')

        # get the Intersection over Union (IoU)
        iou = bb_intersection_over_union(rect_pred, rect_gt)

        # visualisation
        plt.clf()
        plt.subplot(2, 3, 1)
        im = plt.imshow(out_mean * 1.0 / 255)
        im.set_norm(self.color_norm)
        plt.title('Scale response:  ' +
                  str(['{:.3f}'.format(i) for i in [l[max_row_new, max_col_new] / 255. for l in resized_response]]) +
                  '.\n  Corresponding: ' + str(['{:.3f}'.format(i) for i in self.scale_list]))

        plt.subplot(2, 3, 2)
        im = plt.imshow(out_mean_attention * 1.0 / 255)
        im.set_norm(self.color_norm)
        # plt.colorbar(ticks=np.linspace(0, 1.0, 10, endpoint=True))
        if fusion_response < 0.3:
            plt.title('New maximum response is %.2f, NO FISH!!!' % (fusion_response))
        else:
            plt.title('New maximum response is %.2f' % (fusion_response))

        rect_axes_1 = plt.subplot(2, 3, 3)
        plt.imshow(img_origin)
        plt.scatter(x=[max_col], y=[max_row], color='b', s=80, alpha=.5)
        plt.scatter(x=[max_col_new], y=[max_row_new], color='r', s=30, marker='^', alpha=1)
        rect_axes_1.add_patch(rect_pred)
        rect_axes_1.add_patch(rect_gt)
        rect_axes_1.text(rect_pred.xy[0], rect_pred.xy[1], 'IoU: %.3f'%(iou), fontsize=15, color='r')
        plt.title("Red: pred, Green: gt. test image: " + recipe.name)

        # plot rectangle acquisition process
        plt.subplot(2, 3, 4)
        plt.imshow(show_map)

        plt.subplot(2, 3, 5)
        plt.imshow(chosen_region)

        plt.subplot(2, 3, 6)
        plt.imshow(img[top:bottom, left:right, :])
        plt.title("Annotated fish type: " + img_class)

        # plt.draw()
        # plt.waitforbuttonpress(1)

        upper_dir = os.path.join(all_img_dir, img_class)
        response_map_dir = os.path.join(response_img_dir, img_class)
        if not os.path.exists(upper_dir):
            os.mkdir(upper_dir)
        if not os.path.exists(response_map_dir):
            os.mkdir(response_map_dir)

        plt.savefig(os.path.join(upper_dir, recipe.name), bbox_inches='tight')
        # 	1. Save response map for IoU metadata optimization
        np.save(os.path.join(response_map_dir, recipe.name+'.npy'), out_mean_attention)

        #if iou < 0.1:
            # there could be two possibilities
            # (1) the prediction contains a fish but the biggest annotation doest not correspond to this detection
            # (2) the prediction simply is not a fish, and we needs to put it back to the background category
            # regardless we need to save it to a negative folder and then examin it manully
        #print('Finish saving figure: '+os.path.join(upper_dir, recipe.name))
        return iou

    def do_training(self):
        '''
        This is the main function for training
        :return:
        '''
        self.annotations = self.read_annotations()
        iterator_factory = self.create_iterator_factory()
        strclass_to_class_idx = iterator_factory.get_strclass_to_class_idx()
        self.strclass_to_class_idx = strclass_to_class_idx

        if self.args.no_train is False:
            self.train_recipes, self.valid_recipes = unpack(
                self.create_recipes_old(valid_seed=self.valid_seed, train_part=self.args.train_part),
                'train_recipes', 'valid_recipes')
            if self.args.collect_training_validation_images > 0:
                # we collect the training and the validation images and put them in the corresponding folders
                # this is going to take a while
                self.collect_training_validation_images()
                print('Finish collecting images.')
            if self.args.collect_training_validation_stats > 0:
                flag = self.collect_training_validation_stats()
                return flag
            if self.args.fish_redetection > 0:
                flag = self.fish_redetection()
                return flag

    def do_train_epoch(self, epoch_params, train_iterator, model):

        self.ts.act_glr.add(epoch_params['glr'])
        mb_size = self.args.mb_size
        epoch_timer = time.time()
        load_timer = time.time()
        whole_batch_timer = time.time()
        train_losses, train_costs = [], []

        for mb_idx, item in enumerate(train_iterator):
            mb_x, mb_y = item['mb_x'], item['mb_y']
            load_time_per_example = float(elapsed_time_ms(load_timer)) / mb_size
            # print 'load time per example', load_time_per_example
            self.ts.train_per_example_load_time_ms.add(load_time_per_example)

            # if mb_idx % self.args.monitor_freq == 0 and mb_idx:
            # for the moment, we don't monitor the network training...bad habit...
            call_timer = time.time()

            # we distribute the output into the list
            output = []
            idx_out = 0
            for _, n_out in self.TARGETS:
                output.append(mb_y[:, idx_out:idx_out + n_out])
                idx_out == n_out

            loss_batch = model.train_on_batch(mb_x, output)
            call_time_per_example = float(elapsed_time_ms(call_timer)) / mb_size
            self.ts.train_per_example_proc_time_ms.add(call_time_per_example)
            self.ts.train_cost.add(loss_batch[0])

            for i, suff in enumerate(self.get_target_suffixes(self.TARGETS)):
                loss = loss_batch[i + 1]
                ts = getattr(self.ts, 'train_loss_' + suff)
                ts.add(np.mean(loss))
                train_losses.append(np.mean(loss))

            # if mb_idx % 10 == 0:
            #     self.exp.update_ping()

            load_timer = time.time()
            rest_time_per_example = float(
                elapsed_time_ms(whole_batch_timer)) / mb_size - call_time_per_example - load_time_per_example
            self.ts.train_per_example_rest_time_ms.add(rest_time_per_example)
            whole_batch_timer = time.time()

            # if mb_idx % self.args.monitor_freq == 0 and mb_idx >= self.args.monitor_freq:
            #     for name, value in zip(model.metrics_names, loss_batch):
            #         print(name+' = '+str(value))

        self.ts.train_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))

        return Bunch(train_losses=train_losses)

    def do_valid_epoch(self, valid_iterator, model):

        valid_losses = defaultdict(list)
        epoch_timer = time.time()

        p_y_given_x_all = np.zeros(shape=(self.args.mb_size, self.Y_SHAPE), dtype=floatX)

        for mb_idx, item in enumerate(valid_iterator):
            mb_x, mb_y = item['mb_x'], item['mb_y']
            # predict the valid data
            res = model.predict_on_batch(mb_x)
            p_y_given_x_all[:, :] = np.concatenate(res, axis=1)

            for suff, interval in zip(self.get_target_suffixes(self.TARGETS), self.get_intervals(self.TARGETS)):
                # valid_loss will be of dimension equals to # batch
                valid_loss = -(mb_y[:, interval[0]:interval[1]] * np.log(
                    np.finfo(np.float32).eps + p_y_given_x_all[:, interval[0]:interval[1]])).sum(axis=1)
                valid_losses[suff].append(valid_loss)

        for suff in self.get_target_suffixes(self.TARGETS):
            ts = getattr(self.ts, 'val_loss_' + suff)
            # print 'STUFF: ', suff
            # print 'LEN: ', len(valid_losses[suff])
            ts.add(np.mean(valid_losses[suff]))

        self.ts.valid_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))

        return valid_losses[self.get_target_suffixes(self.TARGETS)[0]]


if __name__ == '__main__':
    sys.exit(0)
