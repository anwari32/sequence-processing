import sys

from Bio import SeqIO

# How to run this script:
# python create_k_mer.py -s <file_path> 
#                        -k <how many k in k-mer> 
#                        -n <how many k-mer is retrieved>


# Generate array of k-mer or return original sequence if 
# from https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py
#
# @param sequence : sequence you want to process
# @param k : how many length you want in k-mer. If k=-1 then original sequence is returned.
# @param n_k_mer : how many k-mers are retrieve. If all kmers are required, please put -1.
def create_k_mer(sequence, k, n_k_mer):
    # Clean sequence from N characters.
    sequence = ''.join(c for c in sequence if c not in ['N'])
    if k > 0:
        arr = [sequence[i:i+k] for i in range(len(sequence)+1-k)]
        if n_k_mer > 0:
            arr = arr[0:n_k_mer]
        kmer = ' '.join(arr)
        return kmer
    else:
        return sequence

# Read sequence from fasta file.
# @param source_file : fasta file containing sequence
def read_sequence_from_fasta_file(source_file, k, n_k_mer):
    records = list(SeqIO.parse(source_file, 'fasta'))
    for r in records:
        # id = r.id
        arr_k_mer = create_k_mer(str(r.seq), k, n_k_mer)

if __name__ == "__main__":
    print("argument count {}".format(len(sys.argv)))
    for i, arg in enumerate(sys.argv):
        print('argument {} : {}'.format(i, arg))    