import sys
from data_preparation import kmer
import os
import pandas as pd
from tqdm import tqdm
from getopt import getopt

def _generate_gene_expansion_from_file(src_filepath, target_filepath, chunk_size=512, stride=1):
    if not os.path.exists(src_filepath):
        raise FileNotFoundError(f"File not found {src_filepath}.")

    target_dir = os.path.dirname(target_filepath)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(target_filepath):
        os.remove(target_filepath)
    
    df = pd.read_csv(src_filepath)
    target_file = open(target_filepath, 'x')
    target_file.write(f"sequence,label\n")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        seq_chunks = kmer(row['sequence'].strip(), chunk_size, window_size=stride)
        label_chunks = kmer(row['label'].strip(), chunk_size, window_size=stride)
        for seq, label in zip(seq_chunks, label_chunks):
            target_file.write(f"{seq},{label}\n")
    target_file.close()
    return True

def _generate_gene_expansion_from_dir(src_dirpath, target_dirpath, chunk_size=512, stride=1):
    if not os.path.exists(src_dirpath):
        raise FileNotFoundError(f"File {src_dirpath} not found.")

    if not os.path.exists(target_dirpath):
        os.makedirs(target_dirpath, exist_ok=True)
    
    for fname in os.listdir(src_dirpath):
        fpath = os.path.join(src_dirpath, fname)
        _fname = f"{fname.split('.')[0]}.expanded.csv"
        target_path = os.path.join(target_dirpath, _fname)
        print(f"Working on {fname} => {target_path}                                                             ", end='\r')
        success = _generate_gene_expansion_from_file(fpath, target_path, chunk_size=chunk_size, stride=stride)
        if not success:
            raise Exception(f"Something wrong while expanding gene file.")
    
    return True

def _parse_argv(argv):
    options, argument = getopt(argv, 's:d:c:m:', ["src=", "dest=", "chunk_size=", "stride=", "mode="])
    args = {}
    for opt, argument in options:
        if opt in ['-s', '--src']:
            args['src'] = argument
        elif opt in ['-d', '--dest']:
            args['dest'] = argument
        elif opt in ['-c', '--chunk_size']:
            args['chunk_size'] = int(argument)
        elif opt in ['-m', '--mode']:
            args['mode'] = argument
        elif opt in ['--stride']:
            args['stride'] = int(argument)
        else:
            print(f"Argument {opt} is not recognized.")
            sys.exit(2)

    return args


"""
    This script read gene file and expand the sequence into 512 character chunks.
    This script accepts directory path containing gene sequences in CSV and target directory path to store the expanded gene sequence.
    Gene CSV file contains two columns, ``sequence`` and ``label``.
"""
if __name__ == "__main__":
    print("Preparing sequential labelling data")

    args = _parse_argv(sys.argv[1:])
    for key in args.keys():
        print(f"{key} {args[key]}")

    chunk_size = args['chunk_size']
    stride = args['stride']
    src_path = args['src']
    target_path = args['dest']    
    if args['mode'] == 'multi':
        # Treat this mode as folder processing.
        success = _generate_gene_expansion_from_dir(src_path, target_path, chunk_size=chunk_size, stride=stride)
        if not success:
            print(f"Something went wrong with program execution. Please review the code.")
    else:
        # Assume this is just converting single gene file into expanded gene file.
        success = _generate_gene_expansion_from_file(src_path, target_path, chunk_size=chunk_size, stride=stride)
        if not success:
            print(f"Something went wrong with program execution. Please review the code.")

        

