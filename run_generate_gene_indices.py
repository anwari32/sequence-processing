from data_preparation import generate_sequence_labelling
from getopt import getopt
import sys
import os
import datetime

def _parse_arg(argv):
    output = {}
    opts, args = getopt(argv, "i:f:t:", ["index_file=", "fasta_file=", "target_file=", "index_dir=", "fasta_dir=", "target_dir="])
    for option, argument in opts:
        if option in ['-i', '--index_file']:
            output['index_file'] = argument
        elif option in ['-f', '--fasta_file']:
            output['fasta_file'] = argument
        elif option in ['-t', '--target_file']:
            output['target_file'] = argument
        elif option in ['--index_dir']:
            output['index_dir'] = argument
        elif option in ['--fasta_dir']:
            output['fasta_dir'] = argument
        elif option in ['--target_dir']:
            output['target_dir'] = argument
        else:
            print("Argument {} not recognized.".format(option))
            sys.exit(2)

    return output

if __name__ == "__main__":
    print('Prepare sequential labelling data')
    arguments = _parse_arg(sys.argv[1:])
    for k in arguments.keys():
        print(k, arguments[k])
    
    index_files = arguments['index_file'].strip().split(',') if 'index_file' in arguments.keys() else None
    fasta_files = arguments['fasta_file'].strip().split(',') if 'fasta_file' in arguments.keys() else None
    target_files = arguments['target_file'].strip().split(',') if 'target_file' in arguments.keys() else None
    index_dir = arguments['index_dir'].strip() if 'index_dir' in arguments.keys() else None
    fasta_dir = arguments['fasta_dir'].strip() if 'fasta_dir' in arguments.keys() else None
    target_dir = arguments['target_dir'].strip() if 'target_dir' in arguments.keys() else None

    if None in [index_files, fasta_files, target_dir, target_files]:
        print("Make sure index file, fasta files, target directory, and target files are provided.")
        sys.exit(2)
    
    if not (len(index_files) == len(fasta_files) == len(target_files)):
        print("Each index should correspond to one fasta file and results in one target files.")
        sys.exit(2)
    
    if index_dir:
        index_files = [os.path.join(index_dir, fname) for fname in index_files]
    if fasta_dir:
        fasta_files = [os.path.join(fasta_dir, fname) for fname in fasta_files]
    if target_dir:
        target_files = [os.path.join(target_dir, fname) for fname in target_files]

    start_time = datetime.datetime.now()
    for index, fasta, target in zip(index_files, fasta_files, target_files):
        generate_sequence_labelling(index, fasta, target, do_kmer=True, do_expand=True)
    end_time = datetime.datetime.now()
    print('Prepare sequential labelling data FINISHED for {}'.format(end_time-start_time))


    
