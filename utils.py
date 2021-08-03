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

from Bio import SeqIO
# Read sequence from file. Returns array of sequence.
# @param source_file : Fasta file read for its sequences.
# @return : Array of tuple (id, sequence).
def read_sequence_from_file(source_file):
    sequences = []
    for record in SeqIO.parse(source_file, 'fasta'):
        sequences.append((record.id, str(record.seq)))
    return sequences

# Cleans up sequence by removing 'N'.
# @param sequence : Sequence that will be cleaned up.
# @return clean sequence.
def clean_up(sequence):
    sequence = ''.join(c for c in sequence if c not in ['N'])
    return sequence
