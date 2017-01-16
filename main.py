"""
This is the main file
"""
import sys

from classes.FishClass import FishClass

if __name__ == '__main__':
    fish_class = FishClass()
    paths_to_dump = ['']
    python_paths = ['']
    default_owner = 'kaggle_diwu'

    sys.exit(fish_class.main(default_owner=default_owner,
                             paths_to_dump=paths_to_dump,
                             python_paths=python_paths))

