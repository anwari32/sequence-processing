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


import torch
import os
def save_model_state_dict(model, save_path, save_filename):
    """
    Save model state dictionary.
    """
    save_model_path = os.path.join(save_path, os.path.basename(save_filename))
    if os.path.exists(save_model_path):
        os.remove(save_model_path)
    if not os.path.exists(save_model_path):
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)

def load_model_state_dict(model, load_path):
    """
    Load model state dictionary.
    """
    # If path does not exists, raise Error.
    if not os.path.exists(load_path):
        raise FileNotFoundError("File at {} not found.".format(load_path))
    
    # If path exists but it's a directory, raise Error.
    if not os.path.isfile(load_path):
        raise FileNotFoundError("Path {} doesn't point to file.".format(load_path))

    model.load_state_dict(torch.load(load_path))
    return model