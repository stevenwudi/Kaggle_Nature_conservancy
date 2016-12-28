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
from classes.TrainerClass import BaseTrainer
from classes.SaverClass import ExperimentSaver, Tee, load_obj, Saver
import classes.DataLoaderClass
from classes.DataLoaderClass import unpack, floatX, fetch_path_local, random_perturbation_transform, \
    build_center_uncenter_transforms2, transformation, find_bucket

from classes.TrainerClass import ProcessFunc, MinibatchOutputDirector2, create_standard_iterator, repeat, epoch_header, \
    timestamp_str, EpochData, elapsed_time_ms, elapsed_time_mins
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
            plt.xticks([i*target_w/buckets for i in range(0,buckets)])
            plt.yticks([i*target_h/buckets for i in range(0,buckets)])

            ratio_bucket = 1.0* target_w/buckets
            if r['flip']:
                points = [(slot_bucket2_x * ratio_bucket, slot_bucket1_y * ratio_bucket),
                          (slot_bucket1_x * ratio_bucket, slot_bucket2_y * ratio_bucket)]
            else:
                points = [(slot_bucket1_x * ratio_bucket, slot_bucket1_y * ratio_bucket), (slot_bucket2_x * ratio_bucket,slot_bucket2_y * ratio_bucket)]
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
                'dropout_coeff': self.args.dropout_coeff,
                'glr_decay': self.args.glr_decay,
                'momentum': self.args.momentum,
                'glr': self.args.glr,
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
        # Finished: data loading and augmentation
        print 'Params info before training'
        iterator_factory = self.create_iterator_factory()
        strclass_to_class_idx = iterator_factory.get_strclass_to_class_idx()
        self.strclass_to_class_idx = strclass_to_class_idx

        act_params = Bunch()
        act_params.act_glr = self.args.glr

        if self.args.no_train is False:
            self.train_recipes, self.valid_recipes = unpack(self.create_recipes_old(valid_seed=self.valid_seed,
                                                                                    train_part=self.args.train_part),
                                                            'train_recipes', 'valid_recipes')

            if self.args.show_images > 0:
                show_images_spec = copy.copy(iterator_factory.TRAIN_SPEC)
                show_images_iterator = iterator_factory.get_iterator(self.train_recipes, 10, show_images_spec)
                self.show_images(show_images_iterator, self.args.show_images)

            try:
                for epoch_idx in xrange(self.args.n_epochs):
                    random.shuffle(self.train_recipes)
                    random.shuffle(self.valid_recipes)

                    print epoch_header(epoch_idx)
                    epoch_train_recipes = self.train_recipes
                    random.shuffle(epoch_train_recipes)
                    epoch_valid_recipes = self.valid_recipes

                    epoch_params = {
                        'glr': act_params.act_glr,
                        'l2_reg_global': self.args.l2_reg_global,
                        'mb_size': self.args.mb_size
                    }

                    if epoch_idx >= self.args.glr_burnout:
                        act_params.act_glr *= self.args.glr_decay

                    n_train_batches = len(epoch_train_recipes) // epoch_params['mb_size']
                    n_valid_batches = len(epoch_valid_recipes) // epoch_params['mb_size']
                    train_iterator = iterator_factory.get_train_iterator(epoch_train_recipes, epoch_params['mb_size'],
                                                                         pool_size=self.args.train_pool_size,
                                                                         buffer_size=self.args.train_buffer_size)

                    # We should delay creating the iterator because, it will start loading images
                    print 'valid_size', len(epoch_valid_recipes)

                    real_valid_len = len(epoch_valid_recipes)
                    for i in xrange(real_valid_len):
                        epoch_valid_recipes[i].idx = i

                    to_add = (epoch_params['mb_size'] - len(epoch_valid_recipes) % epoch_params['mb_size']) % \
                             epoch_params['mb_size']

                    if to_add and self.args.valid_partial_batches:
                        epoch_valid_recipes_fixed = epoch_valid_recipes + to_add * [epoch_valid_recipes[0]]
                    else:
                        epoch_valid_recipes_fixed = epoch_valid_recipes

                    print 'real_size', real_valid_len

                    valid_iterator = iterator_factory.get_valid_iterator(epoch_valid_recipes_fixed,
                                                                         epoch_params['mb_size'],
                                                                         self.args.n_samples_valid,
                                                                         pool_size=self.args.valid_pool_size,
                                                                         real_valid_shuffle=self.args.real_valid_shuffle)

                    # one_train_epoch
                    if self.args.no_train_update is False:
                        print('n_train_batches %d, n_valid_batches; %d' % (n_train_batches, n_valid_batches))
                        train_losses = unpack(self.do_train_epoch(epoch_params, train_iterator, self.model),
                                              'train_losses')
                    else:
                        train_losses = [-1.0]
                        train_costs = [-1.0]

                    #######################################################################
                    print 'Epoch', epoch_idx, 'train_losses', np.mean(train_losses)
                    if False:
                        # validation
                        if epoch_idx % self.args.valid_freq == 0 and epoch_idx > 0:
                            valid_losses = self.do_valid_epoch(epoch_idx, epoch_params, valid_iterator, self.args.n_samples_valid, real_valid_len=real_valid_len)
                        else:
                            valid_losses = [0]

                        # stats + saving
                        timestamp_str_ = timestamp_str()
                        file_name = 'epoch_' + str(epoch_idx)

                        if epoch_idx % self.args.SAVE_FREQ == 0:
                            model_path = self.save_model(self.model, file_name)
                        else:
                            model_path = None

                        epoch_data = EpochData(train_loss=np.mean(train_losses),
                                               valid_loss=np.mean(valid_losses),
                                               train_cost=np.mean(train_costs),
                                               train_costs=train_costs,
                                               train_losses=train_losses,
                                               valid_losses=valid_losses,
                                               epoch_params=epoch_params,
                                               model_path=self.url_translator.path_to_url(model_path))
                        self.exp.add_epoch_data(epoch_data.encode())

                        print 'Epoch', epoch_idx, 'train_losses', np.mean(train_losses), 'valid_losses', np.mean(valid_losses)
                        print epoch_params

            except KeyboardInterrupt as e:
                print 'Early break.'

            print '..Training ended.'
        else:
            print '..No Training!!'

    def do_train_epoch(self, epoch_params, train_iterator, model):

        self.ts.act_glr.add(epoch_params['glr'])
        mb_size = epoch_params['mb_size']
        epoch_timer = time.time()
        load_timer = time.time()
        whole_batch_timer = time.time()
        train_losses, train_costs = [], []

        for mb_idx, item in enumerate(train_iterator):
            mb_x, mb_y = item['mb_x'], item['mb_y']
            load_time_per_example = float(elapsed_time_ms(load_timer)) / mb_size
            #print 'load time per example', load_time_per_example
            self.ts.train_per_example_load_time_ms.add(load_time_per_example)

            # if mb_idx % self.args.monitor_freq == 0 and mb_idx:
            # for the moment, we don't monitor the network training...bad habit...
            call_timer = time.time()

            # we distribute the output into the list
            output = []
            idx_out = 0
            for _, n_out in self.TARGETS:
                output.append(mb_y[:, idx_out:idx_out+n_out])
                idx_out == n_out

            loss_batch = model.train_on_batch(mb_x, output)
            call_time_per_example = float(elapsed_time_ms(call_timer)) / mb_size
            self.ts.train_per_example_proc_time_ms.add(call_time_per_example)
            self.ts.train_cost.add(loss_batch[0])

            for i, suff in enumerate(self.get_target_suffixes(self.TARGETS)):
                loss = loss_batch[i+1]
                ts = getattr(self.ts, 'train_loss_' + suff)
                ts.add(np.mean(loss))

            if mb_idx % 10 == 0:
                self.exp.update_ping()

            load_timer = time.time()

            rest_time_per_example = float(elapsed_time_ms(whole_batch_timer)) / mb_size - call_time_per_example - load_time_per_example
            self.ts.train_per_example_rest_time_ms.add(rest_time_per_example)
            whole_batch_timer = time.time()

            if mb_idx % self.args.monitor_freq == 0 and mb_idx >= self.args.monitor_freq:
                for name, value in zip(model.metrics_names, loss_batch):
                    print(name+' = '+str(value))

        self.ts.train_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))

        return Bunch(train_losses=train_losses)

    def do_valid_epoch(self, epoch_idx, epoch_params, valid_iterator, n_samples, real_valid_len):
        print 'read_valid_l', real_valid_len
        lines_done = 0
        lines_done_2 = 0
        valid_id = ml_utils.id_generator(5)
        valid_submit_file = self.saver.open_file(None,
                                                 'valid_submit_{epoch_idx}_{valid_id}.csv'.format(epoch_idx=epoch_idx,
                                                                                                  valid_id=valid_id)).file
        valid_all_samples_submit_file = self.saver.open_file(None,
                                                             'valid_all_samples_submit_{}.csv'.format(valid_id)).file

        valid_losses = defaultdict(list)
        valid_top5_acc = defaultdict(list)
        mb_idx = 0
        epoch_timer = start_timer()
        verbose_valid = self.args.verbose_valid

        losses = defaultdict(list)
        losses_list = []
        results = None

        full_valid = []
        try:
            p_y_given_x_ans = np.zeros(shape=(1, epoch_params['mb_size'], self.Y_SHAPE), dtype=floatX)
            p_y_given_x_all = np.zeros(shape=(epoch_params['mb_size'], self.Y_SHAPE, n_samples), dtype=floatX)
            vidx = 0
            while True:
                self.command_receiver.handle_commands(ct)
                mb_size = epoch_params['mb_size']
                recipes = None
                infos = []
                mb_xs = []
                for samples_idx in xrange(n_samples):
                    item = valid_iterator.next()
                    mb_x, mb_y = item['mb_x'], item['mb_y']
                    mb_xs.append(mb_x)
                    results = item['batch']
                    infos.append(map(lambda a: a.info, results))
                    current_mb_size = len(results)
                    mb_y_corr = mb_y

                    self.x_sh.set_value(mb_x)
                    self.y_sh.set_value(mb_y)

                    res = valid_function(epoch_params['l2_reg_global'], epoch_params['mb_size'])

                    p_y_given_x = res['p_y_given_x']

                    p_y_given_x_all[:, :, samples_idx] = p_y_given_x

                p_y_given_x_ans[0, ...] = np.mean(p_y_given_x_all, axis=2)

                for j in xrange(mb_x.shape[0]):
                    if lines_done < real_valid_len:
                        name = results[j].recipe.name

                        if self.args.write_valid_preds_all:
                            for sample_idx in xrange(n_samples):
                                preds = p_y_given_x_all[j, :, sample_idx]
                                full_valid.append(Bunch(
                                    sample_idx=sample_idx,
                                    name=name,
                                    info=infos[sample_idx][j],
                                    preds=self.permute_preds(self.strclass_to_class_idx, preds[:447])
                                ))

                                p = '{name}_{sample_idx}.jpg'.format(name=name, sample_idx=sample_idx)

                                path = self.saver.get_path('full_valid_imgs', p)
                                img = np.rollaxis(mb_xs[sample_idx][j, ...], axis=0, start=3)

                                img = self.rev_img(img, self.mean, self.std)

                                loss = -(mb_y_corr[j, :447] * np.log(preds[:447])).sum()
                                print 'show_images: saving to ', path, 'loss', loss
                                plot_image_to_file2(img, path)

                                self.write_preds(self.strclass_to_class_idx, preds[:447], name,
                                                 valid_all_samples_submit_file)

                            preds = p_y_given_x_ans[0, j, ...]
                            self.write_preds(self.strclass_to_class_idx, preds[:447], name, valid_submit_file)

                        lines_done += 1

                for suff, interval in zip(self.get_target_suffixes(self.TARGETS), self.get_intervals(self.TARGETS)):
                    temp_lines_done_2 = lines_done_2
                    valid_loss = ml_utils.categorical_crossentropy(p_y_given_x_ans[0, :, interval[0]:interval[1]],
                                                                   mb_y_corr[:, interval[0]:interval[1]])

                    if suff == 'class':
                        top5_accuracy = ml_utils.get_top_k_accuracy(p_y_given_x_ans[0, :, interval[0]:interval[1]],
                                                                    np.argmax(mb_y_corr[:, interval[0]:interval[1]],
                                                                              axis=1), k=5)
                        print 'partial top5', top5_accuracy
                        valid_top5_acc[suff].append(top5_accuracy)

                    if suff == 'class':
                        print np.mean(valid_loss)

                    for j in xrange(mb_x.shape[0]):
                        if temp_lines_done_2 < real_valid_len:
                            valid_losses[suff].append(valid_loss[j])
                            temp_lines_done_2 += 1
                        else:
                            break

                lines_done_2 = temp_lines_done_2

                if mb_idx % 10 == 0:
                    self.exp.update_ping()
                mb_idx += 1

        except StopIteration:
            pass

        print valid_losses['class']

        for suff in self.get_target_suffixes(self.TARGETS):
            ts = getattr(self.ts, 'val_loss_' + suff)
            print 'suff', suff
            print 'LEEEEEEEEEN', len(valid_losses[suff])
            ts.add(np.mean(valid_losses[suff]))
            print 'top5 accuracy', np.mean(valid_top5_acc[suff])

        losses_list = sorted(losses_list, key=lambda b: b.loss)
        for b in losses_list:
            print b.loss, b.recipe.name

        self.ts.valid_epoch_time_minutes.add(elapsed_time_mins(epoch_timer))
        valid_submit_file.close()
        valid_all_samples_submit_file.close()

        if self.args.write_valid_preds_all:
            self.saver.save_obj(full_valid, 'full_valid.3c')

        return valid_losses[self.get_target_suffixes(self.TARGETS)[0]]


if __name__ == '__main__':
    sys.exit(0)
