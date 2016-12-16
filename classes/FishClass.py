import sys
import numpy as np
import random
from termcolor import colored

from TrainerClass import BaseTrainer
from SaverClass import ExperimentSaver, Tee
import DataLoaderClass
from utils.theano_utils import set_theano_fast_compile, set_theano_fast_run


class Trainer(BaseTrainer):
    @classmethod
    def get_n_outs(cls, TARGETS):
        return map(lambda a: a[1], TARGETS)

    @classmethod
    def get_intervals(cls, TARGETS):
        res = []
        s = 0
        for _, n_out in TARGETS:
            res.append((s, s + n_out))
            s += n_out
        return res

    @classmethod
    def get_y_shape(cls, TARGETS):
        return sum(map(lambda a: a[1], TARGETS))


class FishClass(Trainer):
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

            self.exp.set_log_url(self.url_translator.path_to_url(filepath))

    def set_targets(self):
        MANY_TARGETS = {
            'final':
                [
                    ('class', 447),
                    ('new_conn', 2),
                ],
            'final_no_conn':
                [
                    ('class', 447),
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
                    ('class', 447),
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
        module_name = 'architectures.' + arch_name
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
        self.init_shared(self.args.mb_size)
        self.init_model()

        self.process_recipe = getattr(DataLoaderClass, self.args.process_recipe_name)

        self.do_training()

        return 0

    def do_training(self):

        pass




