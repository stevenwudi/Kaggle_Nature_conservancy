import sys
import os
import numpy as np
import random
from termcolor import colored

import json
from collections import defaultdict

from TrainerClass import BaseTrainer
from SaverClass import ExperimentSaver, Tee
import DataLoaderClass
from utils.theano_utils import set_theano_fast_compile, set_theano_fast_run


class FishClass(BaseTrainer):
    @classmethod
    def norm_name(cls, key):
        if key[-4:] == '.jpg':
            key = key[-13:-4]
        return key

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

        self.process_recipe = getattr(DataLoaderClass, self.args.process_recipe_name)

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

        if self.args.slot_annotations_url is not None:
            for fn, fld in enumerate(fish_folders):
                json_path = os.path.join(self.args.slot_annotations_url, fld.lower()+"_labels.json")
                slot_annotations_list = json.load(open(json_path, 'r'))
                count = f_annotation_largest(slot_annotations_list, 'slot', fld, fn)
                print('loading class ' + fld + ', number of annotation is: ' + str(count))

                # l = clean_anno(slot_annotations_list)
                # with open(os.path.join('../clean_boundingbox', fld.lower()+"_labels.json"), 'w') as outfile:
                #     json.dump(l, outfile)

        print("Finish loading images, total number of images is: " + str(len(annotations)))
        return annotations

    def do_training(self):

        self.annotations = self.read_annotations()

        print 'Params info before training'


if __name__ == '__main__':
    sys.exit(0)
