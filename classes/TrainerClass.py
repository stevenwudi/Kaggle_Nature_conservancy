import argparse
import sys
import re
import copy
from mock import Mock


class BaseUrlTranslator(object):
    def url_to_path(self, url):
        return url

    def path_to_url(self, path):
        return path

class BaseTrainer(object):
    def __init__(self):
        pass

    @classmethod
    def create_parser(cls):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
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





