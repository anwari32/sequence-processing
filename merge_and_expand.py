from data_dir import workspace_promoter_dir, workspace_ss_dir, workspace_polya_dir, workspace_dir
from data_preparation import merge_csv, expand_by_sliding_window
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
