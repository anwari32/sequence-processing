"""
This script is prepared to do merging and expansion.
Merging is carried out merging promoter, splice sites, and poly-A.
"""
from data_dir import raw_data_dir, raw_data_polya_dir, raw_data_promoter_dir, raw_data_ss_dir
from data_dir import workspace_promoter_dir, workspace_ss_dir, workspace_polya_dir, workspace_dir
from data_preparation import merge_csv, expand_by_sliding_window, split_and_store_csv

train_file_500 = 'train.500.csv'
train_file_1000 = 'train.1000.csv'
train_file_2000 = 'train.2000.csv'
train_file_3000 = 'train.3000.csv'
train_files = [train_file_500, train_file_1000, train_file_2000, train_file_3000]

train_ss_500 = 'train.500.kmer.csv'
train_ss_1000 = 'train.1000.kmer.csv'
train_ss_2000 = 'train.2000.kmer.csv'
train_ss_3000 = 'train.3000.kmer.csv'
train_ss_files = [train_ss_500, train_ss_1000, train_ss_2000, train_ss_3000]

train_filename = 'train.csv'

for ss_src, train in zip(train_ss_files, train_files):
    prom_src_file = '{}/{}'.format(raw_data_promoter_dir, train_filename)
    polya_src_file = '{}/{}'.format(raw_data_polya_dir, train_filename)
    ss_rc_file = '{}/{}'.format(raw_data_ss_dir, ss_src)
    target_file = '{}/{}'.format(workspace_dir, train)
    print("Merging inputs to {}: {}".format(target_file, merge_csv([prom_src_file, polya_src_file, ss_rc_file], target_file)))

# Split train csv into train.csv and validation.csv
train_ss_files = [train_ss_500, train_ss_1000, train_ss_2000, train_ss_3000]
validation_files = ['validation.500.csv', 'validation.1000.csv', 'validation.2000.csv', 'validation.3000.csv']


validation_file = 'validation.csv'
train_file = 'train.csv'
train_files = ['{}/{}'.format(_dir, train_file) for _dir in[workspace_promoter_dir, workspace_ss_dir, workspace_polya_dir]]
validation_files = ['{}/{}'.format(_dir, validation_file) for _dir in[workspace_promoter_dir, workspace_ss_dir, workspace_polya_dir]]



# Merge train and validation files.
merge_csv(train_files, '{}/{}'.format(workspace_dir, train_file))
merge_csv(validation_files, '{}/{}'.format(workspace_dir, validation_file))

# Expand train and validation files.
train_file_expanded = 'train.expanded.csv'
validation_file_expanded = 'validation.expanded.csv'
src_files = ['{}/{}'.format(workspace_dir, train_file), '{}/{}'.format(workspace_dir, validation_file)]
target_files = ['{}/{}'.format(workspace_dir, train_file_expanded), '{}/{}'.format(workspace_dir, validation_file_expanded)]
for src, target in zip(src_files, target_files):
    print('Expanding {} => {}: '.format(src, target, expand_by_sliding_window(src, target, length=510)))
