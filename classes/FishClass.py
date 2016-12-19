import sys
import os
import copy
import numpy as np
import random
from termcolor import colored
from bunch import Bunch
import json
from collections import defaultdict


from classes.TrainerClass import BaseTrainer
from classes.SaverClass import ExperimentSaver, Tee, load_obj, Saver
import classes.DataLoaderClass
from classes.DataLoaderClass import unpack, floatX, fetch_path_local, random_perturbation_transform, build_center_uncenter_transforms2, transformation, find_bucket

from classes.TrainerClass import ProcessFunc, MinibatchOutputDirector2, create_standard_iterator, repeat
from utils.theano_utils import set_theano_fast_compile, set_theano_fast_run
import traceback
from skimage.transform import warp, SimilarityTransform, AffineTransform


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

            # sloth annotation
            slot_resize_scale = 0.25
            slot_annotation = recipe.annotations
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
            # print slot_1, slot_2
            # print 'slot1_bucket', slot_bucket1_x, slot_bucket1_y
            # print 'slot2_bucket', slot_bucket2_x, slot_bucket2_y
            go_indygo('slot_point1_x', slot_bucket1_x, slot_bucket1_y, slot_mul)
            go_indygo('slot_point1_y', slot_bucket1_y, slot_bucket1_x, slot_mul)
            go_indygo('slot_point2_x', slot_bucket2_x, slot_bucket2_y, slot_mul)
            go_indygo('slot_point2_y', slot_bucket2_y, slot_bucket2_x, slot_mul)

            # indygo ??? What for?

        info = {
            'tform_res': tform_res,
            'r': r
        }

        # print 'img_shape', img.shape

        return Bunch(x=img, y=true_dist, recipe=recipe, info=info)
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

    def initialize(self):
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
        if self.args.debug:
            print colored('Running --debug, fast compile', 'red')
            set_theano_fast_compile()
        else:
            print colored('Running fast run', 'red')

            set_theano_fast_run()

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
                    ('indygo_point1_x', self.args.buckets),
                    ('indygo_point1_y', self.args.buckets),
                    ('indygo_point2_x', self.args.buckets),
                    ('indygo_point2_y', self.args.buckets),
                    ('slot_point1_x', self.args.buckets),
                    ('slot_point1_y', self.args.buckets),
                    ('slot_point2_x', self.args.buckets),
                    ('slot_point2_y', self.args.buckets),
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
            print '..loading arch'
            self.model = self.saver.load_path(self.args.load_arch_path)
        else:
            params = {
                'channels': self.args.channels,
                'image_size': (self.args.crop_h, self.args.crop_w),
                'n_outs': self.get_n_outs(self.TARGETS),
                'conv_l2_reg': self.args.conv_l2_reg,
                'fc_l2_reg': self.args.fc_l2_reg,
                'dropout': self.args.dropout,
            }

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
        print 'ARGS', args

        self.n_classes = self.args.n_classes
        self.exp_dir_path = exp_dir_path

        print ' '.join(sys.argv)
        print args

        exp.set_name(self.args.name)

        self.initialize()
        self.set_targets()
        self.ts = self.create_timeseries_and_figures()
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
            for i,el in enumerate(l):
                if 'filename' in el:
                    key = el['filename']
                key = self.norm_name(self.norm_name(key))
                l[i]['filename'] = key +'.jpg'
            return l

        if self.args.sloth_annotations_url is not None:
            for fn, fld in enumerate(fish_folders):
                json_path = os.path.join(self.args.sloth_annotations_url, fld.lower()+"_labels.json")
                sloth_annotations_list = json.load(open(json_path, 'r'))
                count = f_annotation_largest(sloth_annotations_list, 'sloth', fld, fn)
                print('loading class ' + fld + ', number of annotation is: ' + str(count))

                ### we comment the section below which is used to clean the annoation
                # l = clean_anno(slot_annotations_list)
                # with open(os.path.join('../clean_boundingbox', fld.lower()+"_labels.json"), 'w') as outfile:
                #     json.dump(l, outfile)

        print("Finish loading images, total number of images is: " + str(len(annotations)))
        return annotations

    def create_recipes_old(self, valid_seed, train_part=0.9):
        rng = random.Random()
        rng.seed(valid_seed)
        recipes = []
        fish_ids = list(self.annotations.keys())
        count = defaultdict(int)
        strclass_to_class_idx = {'OTHER': 0, 'ALB': 1, 'BET': 2, 'DOL': 3, 'LAG': 4, 'SHARK': 5, 'YFT': 6}
        class_idx_to_strclass = {v: k for k, v in strclass_to_class_idx.iteritems()}

        for fish_id in fish_ids:
            fish_class = self.annotations[fish_id]['sloth']['fish_class']
            count[fish_class] += 1
            class_name = class_idx_to_strclass[fish_class]
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

        return Bunch(train_recipes=train_recipes,
                     valid_recipes=valid_recipes,
                     strclass_to_class_idx=strclass_to_class_idx)

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
            'TARGETS': self.TARGETS
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
                                                           x_shape=(spec['target_channels'], spec['target_h'], spec['target_w']),
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

    def do_training(self):

        self.annotations = self.read_annotations()

        iterator_factory = self.create_iterator_factory()

        ### TODO: start here to actually train the network
        ### Finished: data loading and augmentation
        print 'Params info before training'
        

if __name__ == '__main__':
    sys.exit(0)