from Bio import SeqIO
from random import shuffle

import os

def kmer(seq, length, window_size=1):
    return [seq[i:i+length] for i in range(0, len(seq)+1-length, window_size)]

def shuffle_sequence(seq, chunk_size):
    arr = kmer(seq, chunk_size, chunk_size)
    arr_even = [arr[i] for i in range(0, len(arr), 2)]
    arr_odds = [arr[i] for i in range(1, len(arr), 2)]

    shuffle(arr_odds)
    shuffled = []
    for i in range(len(arr)):
        if i % 2 == 0:
            shuffled.append(arr_even.pop(0))
        else:
            shuffled.append(arr_odds.pop(0))

    return ''.join(shuffled)

def generate_negative_set_from_fasta(fasta_src, csv_target, label=0, max_length=512, sliding_window=1):
    """
    Generate negative set from sequence originated from fasta file to new csv file with certain header.
    Header = 'sequence', 'label' with 'sequence' contains raw sequence and 'label' contains number representing label (0, 1, ...).
    """
    if not os.path.exists(fasta_src):
        raise IOError('"{}" not found.'.format(fasta_src))
    f = {}
    try:
        if os.path.exists(csv_target):
            os.remove(csv_target)
        f = open(csv_target, 'x')
        seqs = SeqIO.parse(fasta_src, 'fasta')
        header = '{},{}\n'.format('sequence', 'label')
        f.write(header)
        for s in seqs:
            seq = str(s.seq)
            kmers = kmer(seq, max_length, sliding_window)
            
            for sub in kmers:
                # Generate negative sample from positive sample using DeePromoter method (Oubounyt et. al., 2019).
                neg_kmers = shuffle_sequence(sub, 16)
                f.write('{},{}\n'.format(neg_kmers, 0))

        f.close()
        return csv_target
    except Exception as e:
        print('error {}'.format(e))
        f.close()
        return False
    

def generate_csv_from_fasta(src_fasta, target_csv, label):
    fasta = SeqIO.parse(src_fasta, 'fasta')
    target = {}
    if os.path.exists(target_csv):
        os.remove(target_csv)
    target = open(target_csv, 'x')
    target.write('{},{}\n'.format('sequence', 'label'))
    for f in fasta:
        seq = str(f.seq)
        kmers = kmer(seq, 512, 1)

        for sub in kmers:
            target.write('{},{}\n'.format(sub, label))

    target.close()