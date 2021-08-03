# This script previews first n sequences from given fasta file.

from typing import Sequence
from utils import read_sequence_from_file
from utils import clean_up

# Prints first several sequence samples.
# @param source_file : Fasta file from which sequence is previewed.
# @param n_samples : How many sequences are previewed from given fasta file.
# @clean_up : Remove 'N' sequence before previewed.
def preview_sequence(source_file, n_samples, cleanup):
    sequence_tuples = read_sequence_from_file(source_file)
    sequence_tuples = sequence_tuples[0:n_samples]
    for t in sequence_tuples:
        if (cleanup):
            seq = clean_up(t[1])
            print('{} length={}'.format(seq, len(seq)))
        else:
            print('{} length={}'.format(t[1], len(t[1])))            

import sys
# This script can be used in terminal like this.
# -s=<fasta path>, --source_file=<fasta path>
if __name__ == "__main__" :
    arguments = sys.argv[1:]
    long_codes = ['--source_file', '--clean_up', '--n_samples']
    short_codes = ['-s', '-n', '-c']

    source_file = ""
    n_samples = -1
    cleanup = False
    if len(arguments) < 1:
        print('source file not found.')
        sys.exit(0)
    for arg in arguments:
        arr = arg.split('=')
        if (arr[0] in long_codes or arr[0] in short_codes):
            if (arr[0] == '-s' or arr[0] == '--source_file'):
                source_file = str(arr[1])
            elif (arr[0] == '-n' or arr[0] == '--n_samples'):
                n_samples = int(arr[1])
            elif (arr[0] == '-c' or arr[0] == "--clean_up"):
                cleanup = True
        else:
            print('command "{}" not found.'.format(arr[0]))
            sys.exit(0)

    preview_sequence(source_file, n_samples, cleanup)



