from Bio import SeqIO
from tqdm import tqdm
import os
import traceback
import pandas as pd
import random

def create_k_mer(sequence, k=3, n_k_mer=-1, clean_up=False):
    """Generate array of k-mer or return original sequence if 
    from https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py
    
    @param  sequence : sequence you want to process
    @param  k (int | None -> 3) : how many length you want in k-mer. If k=-1 then original sequence is returned.
    @param  n_k_mer (int | None -> -1) : how many k-mers are retrieve. If all kmers are required, please put -1.
    @param  clean_up (boolean : None -> False): Clean ``sequence`` from 'N' characters before converting to kmer.
    """

    # Clean sequence from N characters.
    if clean_up:
        sequence = ''.join(c for c in sequence if c not in ['N'])
    if k > 0:
        arr = [sequence[i:i+k] for i in range(len(sequence)+1-k)]
        if n_k_mer > 0:
            arr = arr[0:n_k_mer]
        kmer = ' '.join(arr)
        return kmer
    else:
        return sequence

def break_sequence(sequence, chunk_size=16):
    len_seq = len(sequence)
    chunks = [sequence[i:i+chunk_size] for i in range(0, len_seq, chunk_size)]
    return chunks

def shuffle_sequence(sequence, chunk_size=16, shuffle_options="odd"):
    chunks = break_sequence(sequence, chunk_size=chunk_size)
    new_chunks = []
    len_chunks = len(chunks)
    odd_chunks = []
    even_chunks = []
    for i in range(len_chunks):
        if i % 2 == 0:
            even_chunks.append(chunks[i])
            even_chunks.reverse()
        else:
            odd_chunks.append(chunks[i])
            odd_chunks.reverse()
    if shuffle_options=="odd":
        random.shuffle(odd_chunks)
    else:
        random.shuffle(even_chunks)
    for i in range(len_chunks):
        if i % 2 == 0:
            new_chunks.append(even_chunks.pop())
        else:
            new_chunks.append(odd_chunks.pop())
    return ''.join(new_chunks)

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
    if not os.path.exists(os.path.dirname(save_model_path)):
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)

def save_checkpoint(model, optimizer, config, path, replace=False):
    """
    Save model and optimizer internal state with other information (config).
    """
    save_object = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    dest_dir = os.path.dirname(path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if os.path.exists(path):
        if replace == True:
            os.remove(path)
        else:
            raise FileExistsError(f"Cannot remove old saved file. File with same name found and replace == ``False``.")
    
    torch.save(save_object, path)


def load_checkpoint(path, default_model, default_optimizer):
    saved_object = torch.load(path)
    model = default_model.load_state_dict(saved_object["model"])
    optimizer = default_optimizer.load_state_dict(saved_object["optimizer"])
    config = saved_object["config"]
    return model, optimizer, config

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

def generate_csv_from_fasta(src_fasta, target_csv, label):
    """
    Generate csv from fasta file.
    @param  src_fasta (string):
    @param  target_csv (string):
    @param  label (int):
    @return (boolean) : True if success.
    """
    try:
        if not os.path.exists(src_fasta):
            raise Exception("File {} not found.".format(src_fasta))
        _columns = ['seq_id','sequence','label']
        _columns = ','.join(_columns)
        fasta = SeqIO.parse(src_fasta, 'fasta')
        target_dir = os.path.dirname(target_csv)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(target_csv):
            os.remove(target_csv)
        target_file = open(target_csv, 'x')
        target_file.write(f"{_columns}\n")
        for record in fasta:
            seq_id = record.id
            sequence = str(record.seq)
            target_file.write(f"{seq_id},{sequence},{label}\n")
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

def generate_csvs_from_fastas(src_fastas, target_csvs, labels):
    """
    @param      src_fastas:
    @param      target_csvs:
    @param      labels:
    """
    for src in src_fastas:
        if not os.path.exists(src):
            raise FileNotFoundError(f"File {src} not exists. Make sure fasta file exists.")
    length = len(src_fastas)
    if length != len(target_csvs) != len(labels):
        raise ValueError(f"Source files must be equal to target files.")
    for i in tqdm(range(length), total=length):
        generate_csv_from_fasta(src_fastas[i], target_csvs[i], labels[i])
    return True

def split_csv(src_csv, fractions=[0.5, 0.5]):
    """
    Split data from src_csv using Pandas.
    @param      src_csv (string): path to csv file.
    @param      fractions (array of float): array containing fraction of each split.
    @returns    frames (array of pd.DataFrame): each frame contains data split.
    """
    if fractions == []:
        print('No splitting.')
        return False
    if sum(fractions) > 1:
        raise Exception("Sum of fractions not equal to one.")
    frames = []
    df = pd.read_csv(src_csv)
    for frac in fractions:
        split = df.sample(frac=frac)
        frames.append(split)
        df = df.drop(split.index)
    #endfor
    return frames

def split_and_store_csv(src_csv, fractions=[], store_paths=[]):
    """
    Split data from src_csv and store each split in store path.
    Each fraction corresponds to each store path.
    @param      src_csv (string): path to src csv.
    @param      fractions (array of float): array containing fraction of each split.
    @param      store_paths (array of string): array containing path of each split.
    @return     (boolean): True if success
    """
    if len(fractions) != len(store_paths):
        raise Exception("Not enough path to store splits.")
    if sum(fractions) > 1:
        raise Exception("Sum of fractions not equal to one.")
    frames = []
    df = pd.read_csv(src_csv)
    _i = 0
    _length = len(fractions)
    for _i in range(_length):
        _frac = fractions[_i]
        _path = store_paths[_i]
        if _i + 1 == _length:
            print("Splitting and storing split to {}".format(_path))
            df.to_csv(_path, index=False)
        else:
            split = df.sample(frac=_frac)
            print("Splitting and storing split to {}".format(_path))
            split.to_csv(_path, index=False)
            frames.append(split)
            df = df.drop(split.index)
            
    #endfor
    return True

def generate_sample(src_csv, target_csv, n_sample=10, seed=1337, replace=False):
    """
    Generate sample data from csv with header: 'sequence' and 'label'.
    Data generated is saved in different csv.
    @param src_csv : CSV source file.
    @param target_csv : CSV target file
    @param n_sample : how many samples selected randomly from source.
    @seed : random state.
    """
    print(f"Sampling {src_csv}")
    df = pd.read_csv(src_csv)
    sampled = {}

    # fraction take over n_sample.
    sampled = df.sample(n=n_sample, replace=replace, random_state=seed)
    try:
        if os.path.exists(target_csv):
            os.remove(target_csv)
        sampled.to_csv(target_csv, index=False)
        return target_csv
    except Exception as e:
        print('Error {}'.format(e))
        return False

def generate_samples(src_csvs, target_csvs, n_sample=10, seed=1337, replace=False):
    for f in src_csvs:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File {f} not found.")
    len_src = len(src_csvs)
    if len_src != len(target_csvs):
        raise ValueError(f"Each csv source corresponds to one target csv.")
    if n_sample == 0:
        print(f"No sample is generated since n_sample={n_sample}")
        return False

    for i in tqdm(range(len_src), total=len_src):
        generate_sample(src_csvs[i], target_csvs[i], n_sample=n_sample, seed=seed, replace=False)
    return True