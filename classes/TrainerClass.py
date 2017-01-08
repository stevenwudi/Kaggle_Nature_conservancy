import argparse
import sys
import re
import os
import numpy as np
import copy
import time
import datetime
from collections import OrderedDict
from mock import Mock
from setproctitle import setproctitle
from bunch import Bunch
import classes.DataLoaderClass
from sklearn.covariance import empirical_covariance
from scipy.linalg import eigh
import multiprocessing as mp
from multiprocessing import Process
import traceback
from Queue import Empty
from classes.SaverClass import load_obj
from classes.DataLoaderClass import floatX
from bson import Binary
import cPickle


def epoch_header(epoch_idx):
    return (50 * '-') + ' epoch_idx = ' + str(epoch_idx) + ' ' + (50 * '-')


def timestamp_str():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S')


def ArgparsePair(value):
    w = value.split('%')
    return int(w[0]), int(w[1])


def trim(examples, mb_size):
    l = len(examples)
    ll = l - l % mb_size
    examples = examples[0:ll]
    return examples


def repeat(examples, n_samples, mb_size):
    examples = trim(examples, mb_size)
    nexamples = []
    for a in xrange(len(examples) / mb_size):
        for i in xrange(n_samples):
            nexamples.extend(examples[a * mb_size: (a + 1) * mb_size])
    return nexamples


def elapsed_time_ms(timer):
    return (time.time() - timer) * 1000


def elapsed_time_mins(timer):
    return (time.time() - timer) / 60.


# This has to be picklable
class EndMarker(object):
    pass


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


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
            print 'LogTimerseries {name} times={x}, y={y}'.format(name=self.name,
                                                              x=xx, y=yy)


class ProcessFunc(object):
    def __init__(self, process_func, *args, **kwargs):
        self.process_func = process_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, elem):
        setproctitle('cnn_worker_thread')
        recipe_result = self.process_func(elem, *self.args, **self.kwargs)
        return recipe_result


class MinibatchOutputDirector2(object):
    from classes.DataLoaderClass import floatX

    def __init__(self, mb_size, x_shape, y_shape, x_dtype=floatX, y_dtype=floatX, output_partial_batches=False):
        self.mb_size = mb_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.output_partial_batches = output_partial_batches

    def handle_begin(self):
        self._start_new_mb()

    def handle_result(self, res):
        self.current_batch.append(res)

        if 'x' in res:
            #print 'chu', self.current_mb_size, self.mb_size
            self.mb_x[self.current_mb_size] = res.x

        if 'y' in res:
            self.mb_y[self.current_mb_size] = res.y

        self.current_mb_size += 1

        if self.current_mb_size == self.mb_size:
            res = self._get_res()
            self._start_new_mb()
            return res
        else:
            return None

    def handle_end(self):
        print 'handle_end'
        if self.output_partial_batches:
            print 'OK', self.current_mb_size
            if len(self.current_batch) != 0:
                return self._get_res()
            else:
                return None
        else:
            print 'none'
            return None

    def _get_res(self):
        return Bunch(batch=self.current_batch,
                        mb_x=self.mb_x,
                        mb_y=self.mb_y)

    def _start_new_mb(self):
        self.current_mb_size = 0
        self.current_batch = []
        self.mb_x = np.zeros(shape=(self.mb_size,) + self.x_shape, dtype=self.x_dtype)
        self.mb_y = np.zeros(shape=(self.mb_size,) + self.y_shape, dtype=self.y_dtype)


class ExceptionMarker(object):
    def __init__(self, traceback):
        self.traceback = traceback

    def get_traceback(self):
        return self.traceback


class BufferedProcessor(object):
    def __init__(self, chunk_loader, buffer_size, add_to_queue_func, name):
        self.chunk_loader = chunk_loader
        self.buffer_size = buffer_size
        self.add_to_queue_func = add_to_queue_func
        self.name = name

    def get_iterator(self):
        def reader_process(chunk_loader, buffer, add_to_queue_func):
            # NOTICE:
            # We have to catch any exception raised in this process, and pass it to the parent
            # which is waiting on the Queue. Any better solution?

            try:
                setproctitle('cnn_buffered_processor' + self.name)
                idx = 0
                chunk_loader_iter = chunk_loader.get_iterator()

                while True:
                    try:
                        v = chunk_loader_iter.next()
                    except StopIteration:
                        break
                    add_to_queue_func(buffer, v)
                    idx += 1

                buffer.put(EndMarker())
            except Exception as e:
                buffer.put(ExceptionMarker(traceback.format_exc()))

        buffer = mp.Queue(maxsize=self.buffer_size)
        process = Process(target=reader_process, args=(self.chunk_loader, buffer, self.add_to_queue_func))
        process.start()
        TIMEOUT_IN_SECONDS = 600

        while True:
            #print 'BufferedProcessor', 'trying to get from the queue', buffer.qsize()
            try:
                v = buffer.get(timeout=TIMEOUT_IN_SECONDS)
            except Empty:
                print 'something is going wrong, could not get from buffer'
                raise

            if isinstance(v, EndMarker):
                break

            if isinstance(v, ExceptionMarker):
                raise RuntimeError(v.get_traceback())
            else:
                #print 'roz', buffer.qsize()
                yield v

        process.join()


class MultiprocessingChunkProcessor(object):
    def __init__(self, process_func, elements_to_process, output_director, chunk_size, pool_size=4, map_chunksize=4):
        """
        :param pre_func:
        :param process_func:
        :param post_func:
        :param examples:
        :param chunk_size: Yielded items will be of size 'chunk_size'.
        :param pool_size:
        :return:
        """
        self.process_func = process_func
        self.elements_to_process = elements_to_process
        self.output_director = output_director
        self.chunk_size = chunk_size
        self.pool_size = pool_size

    def get_iterator(self):
        pool = mp.Pool(self.pool_size)
        # pool = ThreadPool(pool_size)

        self.output_director.handle_begin()
        print 'Will try to pickle', type(self.process_func)
        for chunk in chunks(self.elements_to_process, self.chunk_size):
            chunk_results = pool.map(self.process_func, chunk, chunksize=4)
            for chunk_result in chunk_results:
                res = self.output_director.handle_result(chunk_result)
                if res is not None:
                    yield res

        print('MultiprocessingChunkProcessor_end')
        res = self.output_director.handle_end()
        if res is not None:
            yield res

        pool.close()
        pool.join()


def create_standard_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=100):

    def add_to_queue_func(buffer_queue, item):
        while buffer_queue.full():
            #print 'buffer is full, sleep for a while.', buffer_queue.qsize()
            time.sleep(5)
        #print 'put to buffer_queue'
        buffer_queue.put(item)

    return BufferedProcessor(
        MultiprocessingChunkProcessor(process_func, elements_to_process, output_director, chunk_size=chunk_size,
                                      pool_size=pool_size),
        buffer_size=buffer_size,
        add_to_queue_func=add_to_queue_func,
        name='get_valid_iterator').get_iterator()


class EpochData(object):
    def __init__(self, **kwargs):
        self.d = {}
        for key, value in kwargs.iteritems():
            self.set_field(key, value)

    def as_dict(self):
        return copy.copy(self.d)

    def set_field(self, field, value):
        self.d[field] = value

    def encode(self):
        return Binary(cPickle.dumps(self, protocol=2), 128)

    @classmethod
    def decode(cls, binary):
        return cPickle.loads(binary)


class BaseTrainer(object):

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
        parser.add_argument('--sloth-annotations-url', type=str, default=None, help='TODO')
        parser.add_argument('--auto-sloth-annotations-url', type=str, default=None, help='TODO')
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
        parser.add_argument('--fish-types', type=int, default=7, help='TODO')
        parser.add_argument('--valid-pool-size', type=int, default=4, help='TODO')
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

        # CNN learning related
        parser.add_argument('--load-arch-path', type=str, default=None, help='TODO')
        parser.add_argument('--n-classes', type=int, default=2, help='TODO')
        parser.add_argument('--loss-freq', type=int, default=1, help='TODO')
        parser.add_argument('--monitor-freq', type=int, default=9999999, help='TODO')
        parser.add_argument('--crop-h', type=int, default=448, help='TODO')
        parser.add_argument('--crop-w', type=int, default=448, help='TODO')
        parser.add_argument('--channels', type=int, default=3, help='TODO')
        parser.add_argument('--nof-best-crops', type=int, default=1, help='TODO')
        parser.add_argument('--n-samples-valid', type=int, default=None, help='TODO')
        parser.add_argument('--n-samples-test', type=int, default=None, help='TODO')
        parser.add_argument('--momentum', type=float, default=0.9, help="momentum value")

        parser.add_argument('--l2-reg-global', type=float, default=1.0, help='TODO')
        parser.add_argument('--glr', type=float, default=None, help='TODO')
        parser.add_argument('--valid-freq', type=int, default=1, help='TODO')
        parser.add_argument('--glr-burnout', type=int, default=19999999, help='TODO')
        parser.add_argument('--glr-decay', type=float, default=1.0, help='TODO')

        parser.add_argument('--arch', type=str, default=None, help='TODO')
        parser.add_argument('--log-name', type=str, default='log.txt', help='TODO')
        parser.add_argument('--no-train', action='store_true', help='TODO')
        parser.add_argument('--debug', action='store_true', help='TODO')
        parser.add_argument('--debug_plot', type=int, default=None, help='TODO')
        parser.add_argument('--seed', type=int, default=None, help='TODO')
        parser.add_argument('--valid-seed', type=int, default=None, help='TODO')
        parser.add_argument('--method', type=str, default='rmsprop', help='TODO')
        parser.add_argument('--aug-params', type=str, default=None, help='TODO')
        parser.add_argument('--process-recipe-name', type=str, default=None, help='TODO')

        # hyperparams
        parser.add_argument('--dropout', type=int, default=1, help='TODO')
        parser.add_argument('--dropout-coeff', type=float, default=0.5, help='TODO')
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

    def pca_it(self, spec, recipes, process_recipe):
        MB_SIZE = 10
        process_func = ProcessFunc(process_recipe, spec)
        output_director = MinibatchOutputDirector2(MB_SIZE,
                                                   x_shape=(spec['target_channels'], spec['target_h'], spec['target_w']),
                                                   y_shape=(self.Y_SHAPE,))

        iterator = create_standard_iterator(process_func, recipes, output_director, pool_size=6, buffer_size=40, chunk_size=MB_SIZE * 3)

        print 'computing eigenvalues ...'
        X = np.concatenate([batch['mb_x'][0, ...].reshape((3, -1)).T for batch in iterator])
        n = X.shape[0]
        limit = 125829120
        if n > limit:
            X = X[np.random.randint(n, size=limit), :]
        print X.shape
        cov = empirical_covariance(X)
        print cov
        evs, U = eigh(cov)
        print evs
        print U

        return evs, U

    def estimeate_mean_var(self, iterator):
        channels = self.args.channels
        mean = np.zeros((channels,), dtype=floatX)
        var = np.zeros((channels,), dtype=floatX)
        n_examples = 0
        h, w = None, None

        for mb_idx, item in enumerate(iterator):
            print 'MB_IDX', mb_idx

            mb_x = item['mb_x']
            h = mb_x.shape[2]
            w = mb_x.shape[3]
            for idx in xrange(mb_x.shape[0]):
                n_examples += 1

                for channel in xrange(channels):
                    mean[channel] += np.sum(mb_x[idx, channel, ...])
                    var[channel] += np.sum(mb_x[idx, channel, ...] ** 2)

        mean /= n_examples * h * w
        var /= n_examples * h * w
        return mean, var

    def get_mean_std(self, spec, recipes, test_recipes):
        spec = copy.copy(spec)

        if self.args.no_est:
            mean = [0] * self.args.channels
            std = [1] * self.args.channels
        else:
            if self.args.mean_data_path is None:

                h = classes.DataLoaderClass.my_portable_hash([spec, len(recipes)])
                name = 'mean_std_{}'.format(h)
                print 'mean_std filename', name
                res = self.global_saver.load_obj(name)

                if res is None or self.args.invalid_cache:
                    print '..recomputing mean, std'

                    MB_SIZE = 40
                    process_func = ProcessFunc(self.process_recipe, spec)
                    output_director = MinibatchOutputDirector2(MB_SIZE,
                                                               x_shape=(spec['target_channels'], spec['target_h'], spec['target_w']),
                                                               y_shape=(self.Y_SHAPE,))

                    iterator = create_standard_iterator(process_func, recipes, output_director, pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)

                    mean, _ = self.estimeate_mean_var(iterator)
                    spec.mean = mean
                    iterator = create_standard_iterator(process_func, recipes, output_director, pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)
                    mean2, std_kw = self.estimeate_mean_var(iterator)
                    std = np.sqrt(std_kw)
                    print 'mean2', mean2
                    spec.std = std
                    iterator = create_standard_iterator(process_func, test_recipes, output_director, pool_size=6, buffer_size=40, chunk_size=3 * MB_SIZE)
                    res = self.estimeate_mean_var(iterator)
                    print res
                    mean_data_path = self.global_saver.save_obj((mean, std), name)
                    self.exp.set_mean_data_url(mean_data_path)
                else:
                    print '..using cached mean, std'
                    mean, std = res[0], res[1]
            else:
                mean_data = load_obj(self.args.mean_data_path)
                if len(mean_data) == 2:
                    mean, std = mean_data[0], mean_data[1]
                elif len(mean_data) == 3:
                    # historical compability
                    mean = np.asarray(mean_data)
                    std = np.asarray([255.0, 255.0, 255.0])
                else:
                    raise RuntimeError()

        return mean, std

    def create_control_parser(self, default_owner):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
        parser.add_argument('--exp-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--exp-parent-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--load-arch-path', type=str, default=None, help='TODO')
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
                observer = LogTimeseriesObserver(name=ts_name + ': ', add_freq=mean_freq)
                getattr(ts, ts_name).add_add_observer(observer)

        return ts

    def show_images(self, train_iterator, n):
        import matplotlib.pyplot as plt

        def plot_image_to_file(img, filepath, interpolation='none', only_image=True):
            if only_image:
                from PIL import Image
                if img.dtype in ['float32']:
                    data = np.asarray(img * 255, dtype=np.uint8)
                elif img.dtype in ['int32', 'uint8']:
                    data = np.asarray(img, dtype=np.uint8)
                im = Image.fromarray(data, 'RGB')
                im.save(filepath)
                return filepath
            else:
                plt.imshow(img, interpolation=interpolation)
                plt.savefig(filepath)
                return filepath

        k = 0

        def apply_mean_std_inverse(img, mean, std, channels):
            if std is not None:
                assert (len(std) == channels)
                for channel in xrange(channels):
                    img[:, :, channel] *= std[channel]

            if mean is not None:
                assert (len(mean) == channels)
                for channel in xrange(channels):
                    img[:, :, channel] += mean[channel]

            return img

        for mb_idx, item in enumerate(train_iterator):
            print 'item', type(item)
            mb_x, mb_y = item['mb_x'], item['mb_y']
            batch = item['batch']
            mb_size = len(batch)

            for j in xrange(mb_size):
                img = np.rollaxis(mb_x[j, ...], axis=0, start=3)
                name = batch[j].recipe.name

                img = apply_mean_std_inverse(img, self.mean, self.std, self.args.channels)

                path_suffix = ''
                class_inter = self.get_interval('class', self.TARGETS)
                if class_inter is not None:
                    class_idx = np.argmax(mb_y[j, class_inter[0]:class_inter[1]])
                    path_suffix += '_class_idx_' + str(class_idx)

                crop2_inter = self.get_interval('crop2', self.TARGETS)
                if crop2_inter is not None:
                    crop2_idx = np.argmax(mb_y[j, crop2_inter[0]:crop2_inter[1]])
                    path_suffix += '_crop2_idx_' + str(crop2_idx)

                conn_inter = self.get_interval('conn', self.TARGETS)
                if conn_inter is not None:
                    ryj_conn_idx = np.argmax(mb_y[j, conn_inter[0]: conn_inter[1]])
                    if mb_y[j][ryj_conn_idx + conn_inter[0]] < 0.5:
                        ryj_conn_idx = -1
                else:
                    ryj_conn_idx = -2

                filename = 'img_{mb_idx}_{name}'.format(mb_idx=mb_idx, name=name) + path_suffix
                path = self.saver.get_path('imgs', filename)

                print 'show_images: saving to ', path
                plot_image_to_file(img, path)
                k += 1
                if k >= n:
                    return

    def main(self, *args, **kwargs):
        parser = self.create_parser()
        control_parser = self.create_control_parser(default_owner='diwu')
        control_args, prog_argv = control_parser.parse_known_args(sys.argv[1:])
        control_args = self.transform_urls_to_paths(control_args)
        prog_args = self.transform_urls_to_paths(parser.parse_args(prog_argv))

        print vars(control_args)
        prog_args.load_arch_path = control_args.load_arch_path
        if control_args.exp_dir_path:
            exp_dir_path = control_args.exp_dir_path
        else:
            raise RuntimeError('exp_dir_path is not present!!!')

        exp = Mock()
        self.go(exp, prog_args, exp_dir_path)





