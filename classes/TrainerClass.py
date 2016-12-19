import argparse
import sys
import re
import numpy as np
import copy
from collections import OrderedDict
from mock import Mock
from bunch import Bunch


def ArgparsePair(value):
    w = value.split('%')
    return int(w[0]), int(w[1])


class BaseUrlTranslator(object):
    def url_to_path(self, url):
        return url

    def path_to_url(self, path):
        return path


class TimeSeries(object):
    def __getstate__(self):
        return {'t': self.t}

    def __savestate__(self, d):
        self.t = d['t']

    def __init__(self):
        self.t = []
        self.t_x = []

        self.add_observers = []

    def add(self, y, x=None):
        if x is None:
            x = (0 if len(self.t_x) == 0 else self.t_x[-1]) + 1

        self.t.append(y)
        self.t_x.append(x)

        for add_observer in self.add_observers:
            add_observer.notify_add(self, y, x)

    def add_add_observer(self, observer):
        self.add_observers.append(observer)

    def size(self):
        return len(self.t)

    def last_mean(self, n=None):
        if n is None:
            n = self.size()

        if n > self.size():
            raise RuntimeError()

        return np.mean(self.t[-n:])

    def get_items(self):
        return self.t

    def last_n(self, n):
        if n == -1:
            return self.t
        if n > self.size():
            raise RuntimeError()

        return self.t[-n:]

    def last_x(self):
        return self.t_x[-1]


class LogTimeseriesObserver(object):
    def __init__(self, name, add_freq, fun=np.mean):
        self.add_freq = add_freq
        self.fun = fun
        self.name = name

    def notify_add(self, ts, y, x):
        if ts.size() % self.add_freq == 0 and ts.size():
            xx = x
            yy = self.fun(ts.last_n(self.add_freq))
            print 'LogTimerseries {name} x={x}, y={y}'.format(name=self.name,
                                                              x=xx, y=yy)


class BaseTrainer(object):
    def __init__(self):
        pass

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

    @classmethod
    def create_parser(cls):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')

        parser.add_argument('--name', type=str, default=None, help='TODO')

        parser.add_argument('--load-arch-url', type=str, default=None, help='TODO')
        parser.add_argument('--load-params-url', type=str, default=None, help='TODO')
        parser.add_argument('--mode', type=str, default=None, help='TODO')

        # annos
        parser.add_argument('--slot-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--auto-slot-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--auto-indygo-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--ryj-conn-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--symetria-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--new-conn-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--widacryj-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--point1-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--point2-annotations-url', type=str, default=None, help='TODO')

        parser.add_argument('--target-name', type=str, default=None, help='TODO')

        # paths
        parser.add_argument('--train-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--test-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--train-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--test-csv-url', type=str, default=None, help='TODO')
        parser.add_argument('--mean-data-url', type=str, default=None, help='TODO')
        parser.add_argument('--pca-data-url', type=str, default=None, help='TODO')
        parser.add_argument('--global-saver-url', type=str, default='global', help='TODO')

        # dataloading
        parser.add_argument('--valid-pool-size', type=int, default=6, help='TODO')
        parser.add_argument('--train-pool-size', type=int, default=4, help='TODO')
        parser.add_argument('--test-pool-size', type=int, default=4, help='TODO')
        parser.add_argument('--train-buffer-size', type=int, default=100, help='TODO')
        parser.add_argument('--mb-size', type=int, default=1, help='TODO')
        parser.add_argument('--no-train-update', action='store_true', help='TODO')
        parser.add_argument('--n-epochs', type=int, default=1000000, help='TODO')

        parser.add_argument('--equalize', action='store_true', help='TODO')
        parser.add_argument('--indygo-equalize', action='store_true', help='TODO')

        parser.add_argument('--use-cpu', action='store_true', help='TODO')
        parser.add_argument('--no-est', action='store_true', help='TODO')

        parser.add_argument('--gen-crop1-train', action='store_true', help='TODO')
        parser.add_argument('--gen-crop1-test', action='store_true', help='TODO')
        parser.add_argument('--gen-crop2-train', action='store_true', help='TODO')
        parser.add_argument('--gen-crop2-test', action='store_true', help='TODO')
        parser.add_argument('--gen-submit', action='store_true', help='TODO')
        parser.add_argument('--gen-submit-mod', type=ArgparsePair, default=(0, 1), help='TODO')

        parser.add_argument('--write-valid-preds-all', action='store_true', help='TODO')
        parser.add_argument('--gen-valid-preds', action='store_true', help='TODO')
        parser.add_argument('--margin', type=int, default=40, help='TODO')

        parser.add_argument('--buckets', type=int, default=60, help='TODO')

        parser.add_argument('--diag', action='store_true', help='TODO')
        parser.add_argument('--real-valid-shuffle', action='store_true', help='TODO')
        parser.add_argument('--real-test-shuffle', action='store_true', help='TODO')

        parser.add_argument('--gen-saliency-map-url', type=str, default=None, help='TODO')
        parser.add_argument('--gen-train-saliency-map-n-random', type=int, default=None, help='TODO')
        parser.add_argument('--gen-valid-saliency-map-n-random', type=int, default=None, help='TODO')

        parser.add_argument('--gen-valid-annotations', type=ArgparsePair, default=None, help='TODO')
        parser.add_argument('--gen-train-valid-annotations', type=ArgparsePair, default=None, help='TODO')
        parser.add_argument('--gen-test-annotations', type=ArgparsePair, default=None, help='TODO')

        parser.add_argument('--verbose-valid', action='store_true', help='TODO')

        parser.add_argument('--invalid-cache', action='store_true', help='TODO')

        parser.add_argument('--pca-scale', type=float, default=None, help='TODO')
        parser.add_argument('--train-part', type=float, default=0.9, help='TODO')

        parser.add_argument('--report-case-th', type=float, default=None, help='TODO')
        parser.add_argument('--ssh-reverse-host', type=str, default=None, help='TODO')

        parser.add_argument('--adv-alpha', type=float, default=None, help='TODO')
        parser.add_argument('--adv-eps', type=float, default=None, help='TODO')
        parser.add_argument('--show-images', type=int, default=10, help='TODO')

        parser.add_argument('--valid-partial-batches', action='store_true', help='TODO')

        parser.add_argument('--FREQ1', type=int, default=80, help='TODO')
        parser.add_argument('--SAVE_FREQ', type=int, default=10, help='TODO')

        parser.add_argument('--do-pca', type=int, default=1, help='TODO')
        parser.add_argument('--do-mean', type=int, default=1, help='TODO')
        parser.add_argument('--do-dump', type=int, default=1, help='TODO')

        parser.add_argument('--n-classes', type=int, default=2, help='TODO')
        parser.add_argument('--loss-freq', type=int, default=1, help='TODO')
        parser.add_argument('--monitor-freq', type=int, default=9999999, help='TODO')
        parser.add_argument('--crop-h', type=int, default=448, help='TODO')
        parser.add_argument('--crop-w', type=int, default=448, help='TODO')
        parser.add_argument('--channels', type=int, default=3, help='TODO')
        parser.add_argument('--nof-best-crops', type=int, default=1, help='TODO')
        parser.add_argument('--n-samples-valid', type=int, default=None, help='TODO')
        parser.add_argument('--n-samples-test', type=int, default=None, help='TODO')

        parser.add_argument('--l2-reg-global', type=float, default=1.0, help='TODO')
        parser.add_argument('--glr', type=float, default=None, help='TODO')
        parser.add_argument('--valid-freq', type=int, default=5, help='TODO')
        parser.add_argument('--glr-burnout', type=int, default=19999999, help='TODO')
        parser.add_argument('--glr-decay', type=float, default=1.0, help='TODO')

        parser.add_argument('--arch', type=str, default=None, help='TODO')
        parser.add_argument('--log-name', type=str, default='log.txt', help='TODO')
        parser.add_argument('--no-train', action='store_true', help='TODO')
        parser.add_argument('--debug', action='store_true', help='TODO')
        parser.add_argument('--seed', type=int, default=None, help='TODO')
        parser.add_argument('--valid-seed', type=int, default=None, help='TODO')
        parser.add_argument('--method', type=str, default='rmsprop', help='TODO')
        parser.add_argument('--aug-params', type=str, default=None, help='TODO')
        parser.add_argument('--process-recipe-name', type=str, default=None, help='TODO')

        # hyperparams
        parser.add_argument('--dropout', type=float, default=None, help='TODO')
        parser.add_argument('--fc-l2-reg', type=float, default=None, help='TODO')
        parser.add_argument('--conv-l2-reg', type=float, default=None, help='TODO')
        parser.add_argument('--n-fc', type=float, default=None, help='TODO')
        parser.add_argument('--n-first', type=float, default=None, help='TODO')
        parser.add_argument('--make-some-noise', action='store_true', help='TODO')
        parser.add_argument('--eta', type=float, default=0.01, help='TODO')
        parser.add_argument('--gamma', type=float, default=0.55, help='TODO')

        # unknown
        parser.add_argument('--starting-time', type=int, default=0, help='TODO')
        parser.add_argument('--dummy-run', action='store_true', help='TODO')

        return parser

    def create_control_parser(self, default_owner):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
        parser.add_argument('--exp-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--exp-parent-dir-url', type=str, default=None, help='TODO')
        return parser

    def transform_urls_to_paths(self, args):
        regex = re.compile('.*_url$')
        keys = copy.copy(vars(args))
        for arg in keys:
            if regex.match(arg):
                new_arg = re.sub('_url$', '_path', arg)
                setattr(args, new_arg, getattr(args, arg))
        return args

    def get_url_translator(self):
        return BaseUrlTranslator()

    # The user have to define go function
    def go(self, exp, args, exp_dir_path):
        raise NotImplementedError()

    def create_timeseries_and_figures(self, optim_state={}):
        FREQ1 = self.args.FREQ1

        channels = [
            # train channels
            'train_cost',
            'train_loss',
            'train_slot_loss',
            'train_indygo_loss',
            'train_l2_reg_cost',
            'train_per_example_proc_time_ms',
            'train_per_example_load_time_ms',
            'train_per_example_rest_time_ms',

            # valid channels
            'valid_loss',
            'valid_slot_loss',
            'valid_indygo_loss',

            'l2_reg_global',
            'train_epoch_time_minutes',
            'valid_epoch_time_minutes',
            'act_glr'
        ]

        for suff, _ in self.TARGETS:
            channels.append('train_loss_' + suff)
            channels.append('val_loss_' + suff)

        figures_schema = OrderedDict([

            # (channel_name, name_on_plot, frequency_of_updates)
            ('train', [
                ('train_cost', 'cost', FREQ1),
                ('train_loss', 'loss', FREQ1),
                ('train_slot_loss', 'slot_loss', FREQ1),
                ('train_indygo_loss', 'indygo_loss', FREQ1),
                ('train_l2_reg_cost', 'l2_reg_cost', FREQ1)] +
             [('train_loss_' + suff, 'loss_' + suff, FREQ1) for suff, _ in self.TARGETS]
             ),

            ('valid', [
                ('valid_loss', 'loss', 1),
                ('valid_slot_loss', 'slot_loss', 1),
                ('valid_indygo_loss', 'indygo_loss', 1)] +
             [('val_loss_' + suff, 'loss_' + suff, 1) for suff, _ in self.TARGETS]),

            ('train + valid', [
                ('train_loss', 'train_loss', FREQ1),
            ]),

            ('perf', [
                ('train_per_example_proc_time_ms', 'train_per_example_proc_ms', FREQ1),
                ('train_per_example_load_time_ms', 'train_per_example_load_ms', FREQ1),
                ('train_per_example_rest_time_ms', 'train_per_example_rest_ms', FREQ1)
            ]),

            ('perf_2', [
                ('train_epoch_time_minutes', 'train_epoch_time_minutes', 1),
                ('valid_epoch_time_minutes', 'valid_epoch_time_minutes', 1)
            ]),

            ('act_glr', [
                ('act_glr', 'act_glr', 1)
            ])

        ])

        return self._create_timeseries_and_figures(channels, figures_schema)

    def _create_timeseries_and_figures(self, channels, figures_schema, *args, **kwargs):
        ts = Bunch()
        for ts_name in channels:
            ts.__setattr__(ts_name, TimeSeries())

        for figure_title, l in figures_schema.iteritems():

            for idx, (ts_name, line_name, mean_freq) in enumerate(l):
                observer = LogTimeseriesObserver(name=ts_name + ':' + line_name, add_freq=mean_freq)
                getattr(ts, ts_name).add_add_observer(observer)

        return ts

    def main(self, *args, **kwargs):
        parser = self.create_parser()
        control_parser = self.create_control_parser(default_owner='a')
        control_args, prog_argv = control_parser.parse_known_args(sys.argv[1:])
        control_args = self.transform_urls_to_paths(control_args)
        prog_args = self.transform_urls_to_paths(parser.parse_args(prog_argv))

        print vars(control_args)
        if control_args.exp_dir_path:
            exp_dir_path = control_args.exp_dir_path
        else:
            raise RuntimeError('exp_dir_path is not present!!!')

        exp = Mock()
        self.go(exp, prog_args, exp_dir_path)





