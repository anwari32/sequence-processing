from curses import BUTTON1_DOUBLE_CLICKED
import fractions
from getopt import getopt
import os
import pandas as pd
from tqdm import tqdm
from getopt import getopt
import sys
from data_preparation import merge_csv, generate_kmer_csv
from utils.utils import split_and_store_csv



def _parse_argv(argv):
    arguments = {}
    opts, args = getopt(argv, "s:d:f:t:k:", ["src=", "dest=", "fractions=", "task=", "kmer="])
    for opt, argument in opts:
        if opt in ['-s', '--src']:
            arguments['src'] = argument
        elif opt in ['-d', '--dest']:
            arguments['dest'] = argument
        elif opt in ['-f', '--fractions']:
            arguments['fractions'] = [float(frac) for frac in argument.strip().split(',')]
        elif opt in ['-t', '--task']:
            possible_values = ['merge', 'split', 'kmer']
            if argument not in possible_values:
                print(f"Value {argument} not recognize. Available value {possible_values}")
                sys.exit(2)
            arguments['task'] = argument
        elif opt in ['-k', '--kmer']:
            arguments['kmer'] = int(argument)
        else:
            print(f"Argument {opt} not recognized.")
            sys.exit(2)
    return arguments


if __name__ == "__main__":
    print("Merging genetic sequence.")
    args = _parse_argv(sys.argv[1:])
    for k in args.keys():
        print(k, args[k])

    task = args['task']
    if task == "merge":
        src_dir = args['src']
        src_files = [os.path.join(src_dir, fname) for fname in os.listdir(src_dir)]
        dest_file = args['dest']

        status = merge_csv(src_files, dest_file)
        if not status:
            print(f"Something wrong with merging files in directory {src_dir}.")
            sys.exit(2)
    
    elif task == "split":
        fractions = args['fractions']
        ftypes = ['train', 'test']
        if len(fractions) == 3:
            ftypes = ['train', 'validation', 'test']
        if len(fractions) > 3:
            print("Fractions can only be two fractions.")

        src_filepath = args['src']
        fname = os.path.basename(src_filepath)
        fname = fname.split('.')[0]
        dest_files = ["{}.{}.csv".format(fname, ftype) for ftype in ftypes]
        dest_dir = args['dest']
        dest_filepaths = [os.path.join(dest_dir, f) for f in dest_files]
        
        status = split_and_store_csv(src_filepath, fractions, dest_filepaths)
        if not status:
            print(f"Something wrong with splitting file {src_filepath}.")
            sys.exit(2)
    
    elif task == "kmer":
        # This task only create a kmer version of existing data and save the converted data into separate file.
        src_file = args['src']
        dest_file = args['dest']
        kmer = args['kmer']
        status = generate_kmer_csv(src_file, dest_file, kmer_size=kmer)
        if not status:
            print(f"Something wrong with converting file {src_file} into kmer version.")
            sys.exit(2)
    
    else:
        print(f"Feature for {task} not implemented.")
        sys.exit(2)