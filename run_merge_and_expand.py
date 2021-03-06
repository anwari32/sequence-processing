"""
This script is prepared to do merging and expansion.
Merging is carried out merging promoter, splice sites, and poly-A.
"""
from data_dir import raw_data_polya_dir, raw_data_promoter_dir, raw_data_ss_dir, workspace_dir
from data_preparation import merge_csv, expand_complete_data, merge_dataset
import os

import sys
import getopt
if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "t:n:s", ["types=", "n_samples=", "skip_merge="])
    _types = []
    _n_samples = []
    _skip_merge = False
    for option, argument in opts:
        if option in ['-t', '--types']:
            _types = argument.strip().split(',')
        elif option in ['-n', '--n_samples']:
            _n_samples = [int(a) for a in argument.strip().split(',')]
        elif option in ['-s', '--skip_merge']:
            _skip_merge = bool(argument)
        else:
            print('Argument {} not recognized.'.format(option))
            print('-t, --types=["train", "validation", <other filename>]    Filename to be processed.')
            print('-n, --n_samples=[<integer>, <integer>, ...]              Number of samples in file.')
            print('-s, --skip_merge=<True, False>                           Set True to skip merging process.')
            sys.exit(2)
    #endfor
    #_types = ['train', 'validation']
    # _n_samples = [500, 1000, 2000, 3000]
    #_n_samples = [500, 1000]
    # Merge train.csv and validation.csv
    if not _skip_merge:
        for _type in _types:
            prom = os.path.join(raw_data_promoter_dir, '{}.kmer.csv'.format(_type))
            polya = os.path.join(raw_data_polya_dir, '{}.kmer.csv'.format(_type))
            for _n in _n_samples:
                ss = os.path.join(raw_data_ss_dir, "{}.{}.kmer.csv".format(_type, _n))
                target = os.path.join(workspace_dir, "{}.{}.kmer.all.csv".format(_type, _n))
                print("Merging data promoter, ss, and polya => {}: {}".format(target, merge_dataset(prom, ss, polya, target)))
    #endif
    # Expand train and validation files.
    for _type in _types:
        for _n in _n_samples:
            src = os.path.join(workspace_dir, "{}.{}.kmer.all.csv".format(_type, _n))
            dest = os.path.join(workspace_dir, "{}.{}.kmer.all.expanded.csv".format(_type, _n))
            print('Expanding {} => {}: '.format(src, dest, expand_complete_data(src, dest, length=510)))

