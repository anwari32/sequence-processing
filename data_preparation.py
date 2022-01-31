"""
Contains procedures and function related to data preparation.
"""
import csv
import os
import pandas as pd

def _parse_desc(desc):
    """
    
    """
    # desc_obj = {'gene': '', 'gene_id': '', 'genebank': '', 'ensembl': ''}
    desc_obj = {}
    arr = desc.split(';') # Split desc with semicolon as separator.
    for e in arr:
        det = e.split('=') # Split every parameter and its corresponding value.
        param = det[0].lower()
        val = det[1]
        desc_obj[param] = val
        if param == "dbxref":
            # Parse value of dbxref.
            # i.e. Dbxref=GeneID:653635,Genbank:NR_024540.1,HGNC:HGNC:38034
            # param = dbxref (in lowercase)
            # val = GeneID:653635,Genbank:NR_024540.1,HGNC:HGNC:38034
            dbxref_vals = val.split(',')
            for e in dbxref_vals:
                arr = e.split(':')
                dbxref_param = arr[0].lower()
                dbxref_val = arr[1]
                if dbxref_param == 'geneid':
                    desc_obj['gene_id'] = dbxref_val
                elif dbxref_param == 'genbank':
                    desc_obj['genbank'] = dbxref_val
                elif dbxref_param == 'ensembl':
                    desc_obj['ensembl'] = dbxref_val
                else:
                    break
    
    return desc_obj

def _gff_parseline(line, regions):
    """
    
    """
    if line[0] == '#':
        return False
    else:
        words = line.split('\t')
        sequence_id = words[0]
        refseq = words[1]
        region = words[2]
        start = int(words[3]) # One-based numbering.
        start_index = start-1 # Zero-based numbering.
        end = int(words[4])
        end_index = end-1
        desc = words[8] # Description.
        desc_obj = _parse_desc(desc)
        gene = desc_obj['gene'] if 'gene' in desc_obj.keys() else '' # Gene name.
        gene_id = desc_obj['gene_id'] if 'gene_id' in desc_obj.keys() else '' # Gene ID
        genbank = desc_obj['genbank'] if 'genbank' in desc_obj.keys() else '' # GeneBank
        ensembl = desc_obj['ensembl'] if 'ensembl' in desc_obj.keys() else '' # Ensembl
        if regions is None:
            return {'sequence_id': sequence_id, 'refseq': refseq, 'region': region, 'start': start, 'start_index': start_index, 'end': end, 'end_index': end_index, 'desc': desc_obj, 'gene': gene, 'gene_id': gene_id, 'genbank': genbank, 'ensembl': ensembl}
        elif region in regions:
            return {'sequence_id': sequence_id, 'refseq': refseq, 'region': region, 'start': start, 'start_index': start_index, 'end': end, 'end_index': end_index, 'desc': desc_obj, 'gene': gene, 'gene_id': gene_id, 'genbank': genbank, 'ensembl': ensembl}
        else:
            return False

def gff_to_csv(file, csv_output, regions):
    """
    
    """
    if os.path.exists(file):
        # Prepare file and dataframe.
        if os.path.exists(csv_output):
            os.remove(csv_output)
        colnames = ['sequence_id', 'refseq', 'region', 'start_index', 'end_index', 'start', 'end', 'gene', 'gene_id', 'genebank', 'ensembl']
        header = ",".join(colnames)
        f = open(file, 'r')
        out = open(csv_output, 'x')
        out.write("{} \n".format(header))
        
        for line in f:
            d = _gff_parseline(line, regions)
            try:
                if d != False:
                    if d:
                        output = "{},{},{},{},{},{},{},{},{},{},{}\n".format(d['sequence_id'], d['refseq'], d['region'], d['start_index'], d['end_index'], d['start'], d['end'], d['gene'], d['gene_id'], d['genbank'], d['ensembl'])
                        out.write(output)
                    else:
                        break
            except:
                out.close()
                f.close()
        out.close()
        f.close()

def gff_to_csvs(gff_file, target_folder, regions, header):
    """
    Convert gff file into CSV file.
    @param  gff_file (string): filepath to gff file.
    @param  target_folder (string): path to directory to which converted gff will be saved.
    @param  regions (array of string): array containing what region to look for.
    @param  header (string): header of converted file.
    """
    f = open(gff_file)
    target_file = target_folder + '/'
    cur_seq = ""
    temp_seq = ""
    output_file = ""
    file_to_write = {}
    for line in f:
        d = _gff_parseline(line, regions)
        if d:
            output = "{},{},{},{},{},{},{},{},{},{},{} \n".format(d['sequence_id'], d['refseq'], d['region'], d['start_index'], d['end_index'], d['start'], d['end'], d['gene'], d['gene_id'], d['genbank'], d['ensembl'])
            temp_seq = d['sequence_id']
            if cur_seq == "":
                cur_seq = temp_seq

            # Prepare desired file to write.
            output_file = target_file + temp_seq + '.csv'

            # Compare if this sequence_id is the as previous sequence_id.
            if temp_seq == cur_seq:

                # If it is then write to desired file.
                # Check if file exists. If not then create file.
                if os.path.exists(output_file):
                    file_to_write.write(output)
                else:
                    file_to_write = open(output_file, 'x')

                    # Write header first.
                    file_to_write.write("{}\n".format(header))
                    file_to_write.write(output)
            
            # If this sequence_id is not the same as previous sequence_id, close the existing file.
            elif cur_seq != temp_seq:
                file_to_write.close()
                cur_seq = temp_seq

    # Close any file related to this procedure.
    file_to_write.close()
    f.close()    

def generate_sample(src_csv, target_csv, n_sample=10, seed=1337):
    """
    Generate sample data from csv with header: 'sequence' and 'label'.
    Data generated is saved in different csv.
    @param src_csv : CSV source file.
    @param target_csv : CSV target file
    @param n_sample : how many samples selected randomly from source.
    @seed : random state.
    """
    df = pd.read_csv(src_csv)
    sampled = df.sample(n=n_sample, random_state=seed)
    try:
        if os.path.exists(target_csv):
            os.remove(target_csv)
        sampled.to_csv(target_csv, index=False)
        return target_csv
    except Exception as e:
        print('Error {}'.format(e))
        return False


from Bio import SeqIO

def kmer(seq, length, window_size=1):
    return [seq[i:i+length] for i in range(0, len(seq)+1-length, window_size)]

def generate_csv_from_fasta(src_fasta, target_csv, label, max_seq_length=512, sliding_window_size=1, expand=False):
    """
    Generate csv from fasta file.
    @param  src_fasta (string):
    @param  target_csv (string):
    @param  label (int):
    @param  max_seq_length (int): default 512,
    @param  sliding_window_size (int): default 1,
    """
    fasta = SeqIO.parse(src_fasta, 'fasta')
    target = {}
    if os.path.exists(target_csv):
        os.remove(target_csv)
    target = open(target_csv, 'x')
    target.write('{},{}\n'.format('sequence', 'label'))
    for f in fasta:
        seq = str(f.seq)
        if expand:
            if len(seq) > max_seq_length:
                kmers = kmer(seq, max_seq_length, sliding_window_size)
                for sub in kmers:
                    target.write('{},{}\n'.format(sub, label))
            else:
                target.write('{},{}\n'.format(seq, label))
        else:
            target.write('{},{}\n'.format(seq, label))
    target.close()

from random import shuffle

def shuffle_sequence(seq, chunk_size):
    """
    Shuffle a sequence by dividing sequence into several parts with equal parts.
    Parts are then categorized into odds and even parts, and shuffle the odds parts.
    Merge shuffle parts (odds) and even.
    i.e. AAA BBB CCC DDD => AAA DDD CCC BBB

    @param  seq (string): sequence.
    @param  chunk_size (int): chunk size.
    @return (string) a new sequence.
    """
    lenseq = len(seq)
    if lenseq % chunk_size > 0:
        raise Exception('sequence cannot be divided into equal parts. {} & {}'.format(lenseq, chunk_size))

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

from Bio import SeqIO

def generate_shuffled_data(fasta_src, target_shuffled_csv, label=0, max_sequence_length=512, chunk_size=16, sliding_window=1, expand=False):
    """
    Create negative promoter sequence based on positive TATA sequence.
    param   fasta_src (string): path to fasta file.
    param   target_shuffled_csv (string): path to file to which negative dataset generated will be written.
    param   label (int): default 0 as representation of negative value. 1 is positive value.
    param   max_sequence_length (int): default 512, max sequence length.
    param   chunk_size (int): default 16, chunk size of sequence when the sequence is divided into several chunks.
    param   sliding_window (int): default 1, if sequence length is more than 512 then subsequence of 512 character is taken by sliding window.
    param   expand (boolean): default False, whether is sequence more than max length is expanded.
    return True, if success. False is failed.
    """
    if os.path.exists(target_shuffled_csv):
        os.remove(target_shuffled_csv)
    header = '{},{}\n'.format('sequence', 'label')
    f = {}
    try:
        f = open(target_shuffled_csv, 'x')
        f.write(header)
        sequences = SeqIO.parse(fasta_src, 'fasta')
        for s in sequences:
            seq = str(s.seq)
            if expand:
                kmers = kmer(seq, max_sequence_length, sliding_window)
                for sub in kmers:
                    neg_kmers = shuffle_sequence(sub, chunk_size)
                    f.write('{},{}\n'.format(neg_kmers, 0))
            else:
                shuffled_seq = shuffle_sequence(seq, chunk_size)
                f.write('{},{}\n'.format(shuffled_seq, 0))
        f.close()
        return True
    except Exception as e:
        print('Error {}'.format(e))
        return False

def generate_datasets(src_csv, target_dir, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=1337):
    """
    Split dataset into three parts: training, validation, and test.
    @param src_csv : CSV source file from which dataset is generated.
    @param target_dir : which directory datasets are generated. Filename is created as 'train.csv', 'validation.csv', and 'test.csv'
    @param train_frac : training fraction.
    @param val_frac : validation fraction.
    @param test_frac : testing fraction.
    @param seed : random state.
    @return array of string: path of train file path, validation file path, and test file path.
    """
    df = pd.read_csv(src_csv)
    train_df = df.sample(frac=train_frac, random_state=seed)
    val_df = df.drop(train_df.index)
    test_df = val_df.sample(frac=test_frac/(test_frac + val_frac), random_state=seed)
    val_df = val_df.drop(test_df.index)

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    trainfile = '{}/train.csv'.format(target_dir)
    train_df.to_csv(trainfile, index=False)
    validationfile = '{}/validation.csv'.format(target_dir)
    val_df.to_csv(validationfile, index=False)
    testfile = '{}/test.csv'.format(target_dir)
    test_df.to_csv(testfile, index=False)

    return [trainfile, validationfile, testfile]

def expand_sequence_csv(csv_src, csv_output, length, sliding_window=1):
    """
    Expand sequence in csv file. For each sequence, if length of sequence is more than `length`
    then create kmers with k='length'. Each of mers is written to file as expansion of original sequence.
    param   csv_src (string): path to csv file containing sequences to expanded.
    param   csv_output (string): path to csv file to which expanded sequence will be written.
    param   length (int): size of sequence chunk.
    param   sliding_window (int): default 1, size of window.
    return  (boolean): True if success.
    """
    g = {}
    try:
        df = pd.read_csv(csv_src)
        columns = df.columns.tolist()
        columns = ','.join(columns)
        if os.path.exists(csv_output):
            os.remove(csv_output)
        g = open(csv_output, 'x')
        g.write('{}\n'.format(columns))
        for i, r in df.iterrows():
            seq = r['sequence']
            label = r['label']
            kmers = [seq[i:i+length] for i in range(len(seq)+1-length)]
            for mer in kmers:
                g.write('{},{}\n'.format(mer, label))
        g.close()
        return True
    except Exception as e:
        g.close()
        print('Error {}'.format(e))
        return False

def _parse_pas_line(line):
    """
    Parse line from PAS database. This function is special purpose.
    @param  line (string): a line to be parsed.
    @return (object): composed of pas_id, chr, and position
    """
    arr_line = line.strip().split('\t')
    pas_id = arr_line[0]
    chr = arr_line[1]
    position = arr_line[2]
    return {
        'pas_id': pas_id,
        'chr': chr,
        'position': position,
        'label': 1 if 'NoPAS' not in arr_line else 0
    }


def parse_pas(pas_file_path, pos_output_csv, neg_output_csv, chr_filter=[]):
    if not os.path.exists(pas_file_path):
        raise Exception('File {} not found'.format(pas_file_path))

    f = {}
    pos = {}
    neg = {}
    header = 'pas_id,chr,position,label'
    try:
        outputs = [pos_output_csv, neg_output_csv]
        for output in outputs:
            if os.path.exists(output):
                os.remove(output)
            o = open(output, 'x')
            o.write('{}\n'.format(header))
            o.close()

        f = open(pas_file_path, 'r')
        next(f) # Skip first line.
        pos = open(pos_output_csv, 'a')
        neg = open(neg_output_csv, 'a')
        for line in f:
            obj = _parse_pas_line(line)
            entry = '{},{},{},{}\n'.format(obj['pas_id'], obj['chr'], obj['position'], obj['label'])
            if chr_filter != []:
                if obj['chr'] in chr_filter:
                    if obj['label'] == 1:
                        pos.write(entry)
                    else:
                        neg.write(entry)
            else:
                if obj['label'] == 1:
                    pos.write(entry)
                else:
                    neg.write(entry)

        f.close()
        pos.close()
        neg.close()
        return True
    except Exception as e:
        f.close()
        pos.close()
        neg.close()
        print('error {}'.format(e))
        return False

chr_dict = {
    'chr1': 'NC_000001.11',
    'chr2': 'NC_000002.12',
    'chr3': 'NC_000003.12',
    'chr4': 'NC_000004.12',
    'chr5': 'NC_000005.10',
    'chr6': 'NC_000006.12',
    'chr7': 'NC_000007.14',
    'chr8': 'NC_000008.11',
    'chr9': 'NC_000009.12',
    'chr10': 'NC_000010.11',
    'chr11': 'NC_000011.10',
    'chr12': 'NC_000012.12',
    'chr13': 'NC_000013.11',
    'chr14': 'NC_000014.9',
    'chr15': 'NC_000015.10',
    'chr16': 'NC_000016.10',
    'chr17': 'NC_000017.11',
    'chr18': 'NC_000018.10',
    'chr19': 'NC_000019.10',
    'chr20': 'NC_000020.11',
    'chr21': 'NC_000021.9',
    'chr22': 'NC_000022.11',
    'chr23': 'NC_000023.11', 'chrX': 'NC_000023.11', # chr23 is chr X
    'chr24': 'NC_000024.10', 'chrY': 'NC_000024.10', # chr24 is chr Y
    'mitochondrion': 'NC_012920.1'
}

def generate_polya_sequence(polya_index_csv, target_csv, chr_index_dir, chr_index, flank_left_size=256, flank_right_size=256):
    """
    Generate polya sequence from given index and write the sequence into csv file with its label.
    @param  polya_index_csv (string): path to polya index csv file. This file contains information regarding position of polya motif in certain chromosome.
    @param  target_csv (string): path to csv file on which the generated sequence will be written.
    @param  chr_index_dir (string): path to directory containing chr index.
    @param  chr_index: dictionary mapping chr to its fasta file.
    @param  flank_left_size (int): default 256, how many base are included into sequence from left side of position.
    @param  flank_right_size (int): default 256, how manu base are included into sequence from right side of position.
    @return (boolean): True if sucess.
    """
    if not os.path.exists(polya_index_csv):
        raise Exception('File {} not found.'.format(polya_index_csv))

    df = pd.read_csv(polya_index_csv)
    len_df = len(df)
    header = 'sequence,label'
    f = {}
    chr_cache = {} # cache to store fasta so no need to re-read chr fasta files.
    progress_counter = 0
    try:
        if os.path.exists(target_csv): # Always remove if exists.
            os.remove(target_csv)

        print('Generating Poly-A sequence from {} indices.'.format(len_df))
        f = open(target_csv, 'x')
        f.write('{}\n'.format(header))
        for i, r in df.iterrows():
            pas_id = r['pas_id']
            chr = r['chr']
            position = r['position']
            label = r['label']
            progress_counter += 1

            print('Processing {}/{}, pas id {}'.format(progress_counter, len_df, pas_id), end='\r')

            chr_fasta = '{}/{}.fasta'.format(chr_index_dir, chr_index[chr])
            if chr_cache == {}:
                fastas = SeqIO.parse(chr_fasta, 'fasta')
                chr_cache = {
                    'chr': chr,
                    'fastas': fastas
                }
            else:
                if chr_cache['chr'] != chr:
                    fastas = SeqIO.parse(chr_fasta, 'fasta')
                    chr_cache = {
                        'chr': chr,
                        'fastas': fastas
                    }
            
            # position from original poly-a database is not zero-based.
            position = position - 1 
            for fasta in chr_cache['fastas']:
                left_side_arr = []
                right_side_arr = []
                str_seq = str(fasta.seq)

                # Take flank_left_size amount of base from the left side of position.
                start = position-flank_left_size
                for i in range(start, flank_left_size):
                    left_side_arr.append(str_seq[i]) 

                # Take flank_right_size amount of base from the right side of position. This side include position.
                for j in range(position, flank_right_size):
                    right_side_arr.append(str_seq[i])

                left_side = ''.join(left_side_arr)
                right_side = ''.join(right_side_arr)
                entry = ''.join([left_side, right_side])
                f.write('{},{}\n'.format(entry, label))
                # print('Processing {}/{}, pas id {} => {} ...'.format(progress_counter, len_df, pas_id, entry[0:10]), end='\r')

        f.close()
        return True
    except Exception as e:
        print('Error {}'.format(e))        
        f.close()
        return False

def merge_file_into_csv(dir_path, target_csv, label):
    try:
        if os.path.exists(target_csv):
            os.remove(target_csv)
        h = open(target_csv, 'x')
        h.write('{},{}\n'.format('sequence', 'label'))
        for fname in os.listdir(dir_path):
            path = '{}/{}'.format(dir_path, fname)
            f = open(path, 'r')
            for line in f:
                line = line.strip()
                h.write('{},{}\n'.format(line, label))
            f.close()
        h.close()
        return True
    except Exception as e:
        print('error {}'.format(e))
        return False

def merge_csv(csv_files, csv_target):
    """
    Merge multiple csv files. CSV files to be merge have to have the same header.
    @param  csv_files (array of string): array of csv file path to be merged.
    @param  csv_target (string): csv target file path.
    @return (boolean): True if success.
    """
    if all([os.path.exists(a) for a in csv_files]):

        if os.path.exists(csv_target):
            os.remove(csv_target)

        t = open(csv_target, 'x')
        t.write('sequence,label\n')
        for a in csv_files:
            f = open(a, 'r')
            next(f) # Skip first line.
            for line in f:
                t.write(line)
            f.close()
        t.close()
        return True

    return False