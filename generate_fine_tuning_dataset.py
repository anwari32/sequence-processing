# This script generates dataset for fine tuning DNABert.
# Fine tuning instance is formatted like this.
# <k-mer 0><space><k-mer 1><space> ... <k-mer n><tab><class label>

from generate_training_dataset import create_k_mer
from utils import read_sequence_from_file
import os


# Generate fine tuning dataset from fasta file.
# @param source_file : Fasta file containing sequences.
# @param class_number : A number indicating a class.
# @param n_k_mer : How many k-mers are extracted from the sequences.
# @param output_file_path : Output file containing fine tuning dataset.
# @return : Output file path.
def generate_fine_tuning_dataset(source_file, class_number, k, n_k_mer, output_file_path):
    # Remove output file if exists.
    if os.path.exists(output_file_path):
        os.path.remove(output_file_path)
    output_file = open(output_file_path, 'w+')

    sequence_tuples = read_sequence_from_file(source_file)
    for tuple in sequence_tuples:
        instance = create_k_mer(tuple[1], k, n_k_mer) + '\t' + class_number
        output_file.write(instance)

    output_file.close()
    return output_file_path

import sys

# This script can be used in terminal like this.
# -s=<fasta file>, --source_file=<fasta file>
# -c=<fasta file>, --class_number=<class number>
# -k=<size of k in k-mer>,
# -n=<how many kmers are there per instance>
# -o=<output file path>, --output_file_path=<output_file_path>
if __name__ == "__main__":
    short_codes = ['-s', '-c', '-k', '-n', 'o']
    long_codes = ['--source_file', '--class_number', '--output_file_path']

    arguments = sys.argv
    source_file = ""
    class_number = -1
    k = -1
    n_k_mer = -1
    output_file_path = ""
    for arg in arguments:
        arr = arg.split('=')
        if (arr[0] in short_codes or arr[0] in long_codes):
            if (arr[0] == '--source_file' or arr[0]=='-s'):
                source_file = str(arr[1])
            elif (arr[0] == '--class_number' or arr[0] == '-c'):
                class_number = int(arr[1])
            elif (arr[0] == '--output_file_path' or arr[0] == '-o'):
                output_file_path = str(arr[1])
            elif (arr[0] == '-k'):
                k = int(arr[1])
            elif (arr[0] == '-n'):
                n_k_mer = int(arr[1])
        else:
            print("command '{}' not found. ".format(arr[0]))
            sys.exit(0)
    
    result = generate_fine_tuning_dataset(source_file, class_number, k, n_k_mer, output_file_path)
    print('dataset is generated at "{}".'.format(result))