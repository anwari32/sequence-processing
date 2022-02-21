"""
This script is prepared to do merging and expansion.
Merging is carried out merging promoter, splice sites, and poly-A.
"""
from data_dir import raw_data_polya_dir, raw_data_promoter_dir, raw_data_ss_dir, workspace_dir
from data_preparation import merge_csv, expand_by_sliding_window
import os

_types = ['train', 'validation']
_n_samples = [500, 1000, 2000, 3000]

# Merge train.csv and validation.csv
for _type in _types:
    prom = os.path.join(raw_data_promoter_dir, '{}.kmer.csv'.format(_type))
    polya = os.path.join(raw_data_polya_dir, '{}.kmer.csv'.format(_type))
    for _n in _n_samples:
        ss = os.path.join(raw_data_ss_dir, "{}.{}.kmer.csv".format(_type, _n))
        target = os.path.join(workspace_dir, "{}.{}.kmer.all.csv".format(_type, _n))
        print("Merging data promoter, ss, and polya => {}: {}".format(target, merge_csv([prom, ss, polya], target)))

# Expand train and validation files.
for _type in _types:
    for _n in _n_samples:
        src = os.path.join(workspace_dir, "{}.{}.kmer.all.csv".format(_type, _n))
        dest = os.path.join(workspace_dir, "{}.{}.kmer.all.expanded.csv".format(_type, _n))
        print('Expanding {} => {}: '.format(src, dest, expand_by_sliding_window(src, dest, length=510)))
