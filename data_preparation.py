"""
Contains procedures and function related to data preparation.
"""
import os
import pandas as pd
import traceback
from tqdm import tqdm

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

def _parse_desc(desc):
    """
    Parse description in GFF row.
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

def _check_segment_product(product_keywords, product_desc):
    """
    Check of certain keywords are in product description `product_desc`.
    @param  product_keywords (array of string): array containing keywords.
    @param  product_desc (string): a string in which keyword will be searched.
    @return (boolean): True if keyword found.
    """
    found = False
    len_keywords = len(product_keywords)
    arr_product_desc = product_desc.strip().split(' ') # Split words by space.
    i = 0
    while not found and i < len_keywords:
        keyword = product_keywords[i]
        if keyword in arr_product_desc:
            found = True
        else:
            i += 1
    return found

def _gff_parseline(line, regions=None, products=None):
    """
    Parse GFF line.
    @param  line (string): line in GFF.
    @param  regions (array): which region will be retrieved. If None then all regions are retrieved. 
    @return (object): Object containing sequence_id, refseq, region, start, end, start_index, end_index, desc, gene, gene_id, genbank, and emsembl.
    """
    if line[0] == '#':
        return False
    else:
        line = line.strip()
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
        product = desc_obj['product'] if 'product' in desc_obj.keys() else '' # Product
        if regions is None and product is None:
            return {'sequence_id': sequence_id, 'refseq': refseq, 'region': region, 'start': start, 'start_index': start_index, 'end': end, 'end_index': end_index, 'desc': desc_obj, 'gene': gene, 'gene_id': gene_id, 'genbank': genbank, 'ensembl': ensembl, 'product': product}
        else:
            if (regions != None and region not in regions) or (products != None and not _check_segment_product(products, product)):
                return False
            else:
                return {'sequence_id': sequence_id, 'refseq': refseq, 'region': region, 'start': start, 'start_index': start_index, 'end': end, 'end_index': end_index, 'desc': desc_obj, 'gene': gene, 'gene_id': gene_id, 'genbank': genbank, 'ensembl': ensembl, 'product': product}


def gff_to_csv(file, csv_output, regions=None):
    """
    Procedure to create csv file based on GFF file.
    @param  file (string): path to GFF.
    @param  csv_output (string): path to target csv.
    @param  regions (array): which region will be retrieved. If None then all regions are retrieved.
    """
    f = {}
    out = {}
    try:
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
                except Exception as e:
                    print('Error {}'.format(e))
                    out.close()
                    f.close()
            out.close()
            f.close()
            return True
        else:
            raise Exception('File {} not found'.format(file))
    except Exception as e:
        print('Error {}'.format(e))
        out.close()
        f.close()


def gff_to_csvs(gff_file, target_folder, header='sequence_id,refseq,region,start_index,end_index,start,end,gene,gene_id,genbank,ensembl', regions=None):
    """
    Convert gff file into multiple CSV files. Each file is for one sequence_id or chromosome.
    @param  gff_file (string): filepath to gff file.
    @param  target_folder (string): path to directory to which converted gff will be saved.
    @param  regions (array of string): array containing what region to look for.
    @param  header (string): header of converted file.
    @return (boolean): True if success.
    """
    f = {}
    file_to_write = {}
    try:
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
        return True
    except Exception as e:
        print('Error {}'.format(e))
        file_to_write.close()
        f.close()
        return False

def generate_annotated_sequence(chr_index_csv, chr_fasta, chr_annotation_file):
    """
    Read chromosome annotation index and chromosome fasta and generate label for each of base character in chromosome.
    @param      chr_index_csv (string): path to chromosome csv index.
    @param      chr_fasta (string): path to chromosome fasta.
    @param      chr_annotation_file: path to save chromosome sequence and its label.
    @return     (boolean): True if success.
    """
    records = SeqIO.parse(chr_fasta, 'fasta')
    genome_record = next(records)
    genome_str = str(genome_record.seq)
    genome_length = len(genome_str)
    annotation = ['.' for i in range(genome_length)]

    f = open(chr_index_csv, 'r')
    next(f) # Skip first line since first line is just header.


    for line in tqdm(f):
        """
        Since the file is basically CSV, parsing CSV is carried out manually line by line.
        The header is this.
        `sequence_id,refseq,region,start_index,end_index,start,end,gene,gene_id,genbank,ensembl`
        Enjoy tah siah!
        """
        arr_line = line.strip().split(',')
        seq_id = arr_line[0]
        refseq = arr_line[1]
        region = arr_line[2]
        start_index = int(arr_line[3])
        end_index = int(arr_line[4])
        start = int(arr_line[5])
        end = int(arr_line[6])
        
        if region == 'region':
            """
            This is the genome range. `annotation` has been initialized based on sequence in genome fasta file.
            Compare the size and use the largest.
            """
            if len(annotation) < end:
                annotation = ['.' for i in range(end)]
            
        elif region == 'exon':
            """
            This is exon. Take the region and write 'E' on annotation sequence.
            """
            start_region = start_index
            end_region = end
            for j in range(start_index, end):
                annotation[j] = 'E'
        else:
            continue
    #endfor
    """
    Compare the sequence and the annotation sequence. 
    If annotation sequence is smaller than sequence, pad it with '.'.
    If DNA sequence is smaller, pad it with 'N'.
    """
    annotation_length = len(annotation)
    delta = genome_length - annotation_length
    if (delta > 0):
        print('padding annotation')
        for k in range(0, delta):
            annotation.append('.')
    if (delta < 0):
        print('padding sequence')
        for k in range(0, delta):
            genome_str = genome_str + 'N'

    annotation_str = ''.join(annotation)
    arr_annotation_str = kmer(annotation_str, 512, 1)
    arr_genome_str = kmer(genome_str, 512, 1)
    path = chr_annotation_file
    if os.path.exists(path):
        os.remove(path)
    g = open(path, 'x')
    g.write('sequence,label\n')
    for seq, label in tqdm(zip(arr_genome_str, arr_annotation_str), total=len(arr_genome_str)):
        g.write('{},{}\n'.format(seq, label))
    g.close()
    return True

def generate_sequence_labelling(chr_index, chr_fasta, target_csv, do_kmer=False, do_expand=False, kmer_size=3, expand_size=512, region='exon', limit_index=0, random=True, random_seed=1337):
    """
    Generate sequence labelling from given chromosome index and chromosome fasta.
    This function reads chr index to get all exon ranges and create labelling based on the index.
    After that sequence and its label sequence will be written into `target_csv`.
    If `do_expand` = True then sequence from `chr_fasta` will be expanded into 512-character chunks before being written.
    If `do_kmer` = True then sequence from `chr_fasta` will be converted into kmer sequence with `k` = `kmer_size`. Only works if `do_expand` is True.
    Expansion is carried out before conversion to kmer.
    @param      chr_index (string):
    @param      chr_fasta (string):
    @param      target_csv (string):
    @param      do_kmer (boolean):
    @param      do_expand (boolean):
    @param      kmer_size (int): default is 3.
    @param      expand_size (int): default is 512.
    @param      refseq (string):
    @param      limit_index (int): Limit how many rows are processed from `chr_index`
    @param      random (boolean): If the number of rows are chosen randomly.
    @param      random_seed (int): random seed.
    @return     (boolean): True if success
    """
    if not os.path.exists(chr_index):
        raise FileNotFoundError("Chromosome index file {} not found".format(chr_index))
    if not os.path.exists(chr_fasta):
        raise FileNotFoundError("Chromosome source file {} not found.".format(chr_fasta))
    
    print("Processing index {}".format(chr_index), end='\r')
    records = SeqIO.parse(chr_fasta, 'fasta')
    record = next(records) # Assume that chromosome fasta contain just one record so get first record only.
    chr_sequence = str(record.seq) # Get string representation of chr sequence.
    label_sequence = '.' * len(chr_sequence) # Construct dot string with same length as chr_sequence.
    arr_label_sequence = list(label_sequence) # Convert dot string into array of character.
    index_df = pd.read_csv(chr_index)
    index_df = index_df[index_df['region'] == region]

    # Get maximum range of index.
    end_range = max(index_df['end_index'])

    len_df = len(index_df)
    _count = 0
    print("Processing index {} [0/{}]".format(chr_index, len_df), end='\r')
    for i, r in index_df.iterrows():
        _count += 1
        print("Processing index {}: [{}/{}]".format(chr_index, _count, len_df), end='\r')
        # Change label in label_sequence based on selected region.
        for j in range(r['start_index'], r['end']):
            arr_label_sequence[j] = 'E'
        #endfor
        label_sequence = ''.join(arr_label_sequence) # Convert array of character back to string.
    #endfor

    if os.path.exists(target_csv):
        os.remove(target_csv)
    target_file = open(target_csv, 'x')
    target_file.write('sequence,label\n')
    if do_expand:
        arr_chr_sequence = kmer(chr_sequence, expand_size)
        arr_label_sequence = kmer(label_sequence, expand_size)
        _count = 0
        _len = len(arr_chr_sequence)
        for seq, label in zip(arr_chr_sequence, arr_label_sequence):
            _count += 1
            print("Processing index {}, with fasta {}, to seq. labelling {}, expanding [{}/{}]".format(chr_index, chr_fasta, target_csv, _count, _len), end='\r')
            if do_kmer:
                seq = ' '.join(kmer(seq, 3))
                label = ' '.join(kmer(label, 3))
            target_file.write("{},{}\n".format(seq, label))
    else:
        print("Processing index {}, with fasta {}, to seq. labelling {}".format(chr_index, chr_fasta, target_csv), end='\r')
        target_file.write("{},{}\n".format(chr_sequence, label_sequence))
    
    target_file.close()
    return True


def generate_sample(src_csv, target_csv, n_sample=10, frac_sample=0, seed=1337):
    """
    Generate sample data from csv with header: 'sequence' and 'label'.
    Data generated is saved in different csv.
    @param src_csv : CSV source file.
    @param target_csv : CSV target file
    @param n_sample : how many samples selected randomly from source.
    @seed : random state.
    """
    df = pd.read_csv(src_csv)
    sampled = {}

    # fraction take over n_sample.
    print('Generate sample for {} => {}'.format(src_csv, target_csv), end='\r')
    if frac_sample > 0:
        sampled = df.sample(frac=frac_sample, random_state=seed)
    else:
        sampled = df.sample(n=n_sample, random_state=seed)
    try:
        if os.path.exists(target_csv):
            os.remove(target_csv)
        sampled.to_csv(target_csv, index=False)
        return target_csv
    except Exception as e:
        print('Error {}'.format(e))
        return False

def generate_sample_from_dir(dir_path, n_sample=0, frac_sample=0, seed=1337, files=[]):
    try:
        if not os.path.isdir(dir_path):
            raise Exception("Location <{}> is not valid directory.".format(dir_path))

        files = os.listdir(dir_path)
        for f in files:
            src_path = "{}/{}".format(dir_path, f)
            target_path = "{}/{}_sample.csv".format(dir_path, f.split('.')[0])
            if not generate_sample(src_path, target_path, n_sample=n_sample, frac_sample=frac_sample, seed=seed):
                raise Exception("Failed generating sample for <{}>".format(src_path))
        #endfor
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

from Bio import SeqIO

def kmer(seq, length, window_size=1):
    """
    Convert string `seq` into array of fixed `length` token (kmer) .
    @param      seq (string):
    @param      length (int):
    @param      window_size (int): stride.
    @return     (array of string): array of kmer.
    """
    return [seq[i:i+length] for i in range(0, len(seq)+1-length, window_size)]

def chunk_string(seq, length):
    """
    Chunk string `seq` into fixed `length` parts.
    @param      seq (string): string to break.
    @param      length (int): size of each chunk.
    @return     (array of string): array of `seq` parts.
    """
    return [seq[i:i+length] for i in range(0, len(seq), length)]

def generate_csv_from_fasta(src_fasta, target_csv, label, max_seq_length=512, sliding_window_size=1, expand=False):
    """
    Generate csv from fasta file.
    @param  src_fasta (string):
    @param  target_csv (string):
    @param  label (int):
    @param  max_seq_length (int): default 512,
    @param  sliding_window_size (int): default 1,
    """
    if not os.path.exists(src_fasta):
        raise Exception("File {} not found.".format(src_fasta))
    fasta = SeqIO.parse(src_fasta, 'fasta')
    target = {}
    try:
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
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False


from random import shuffle

def shuffle_sequence(seq, chunk_size):
    """
    Shuffle a sequence by dividing sequence into several parts with equal parts. 
    If there is an extra part, this part will be excluded from shuffling and will be concatenated after shuffled sequence.
    Parts are then categorized into odds and even parts, and shuffle the odds parts.
    Merge shuffle parts (odds) and even.
    i.e. AAA BBB CCC DDD => AAA DDD CCC BBB

    @param  seq (string): sequence.
    @param  chunk_size (int): chunk size.
    @return (string) a new sequence.
    """
    lenseq = len(seq)
    last_chunk = ""
    rem = lenseq % chunk_size
    if rem > 0:
        # raise Exception('sequence cannot be divided into equal parts. {} & {}'.format(lenseq, chunk_size))
        last_chunk = seq[lenseq-rem:]
    main_chunk = seq[0:lenseq-rem]

    arr = kmer(main_chunk, chunk_size, chunk_size)
    arr_even = [arr[i] for i in range(0, len(arr), 2)]
    arr_odds = [arr[i] for i in range(1, len(arr), 2)]

    shuffle(arr_odds)
    shuffled = []
    for i in range(len(arr)):
        if i % 2 == 0:
            shuffled.append(arr_even.pop(0))
        else:
            shuffled.append(arr_odds.pop(0))

    shuffled_seq = ''.join(shuffled)
    shuffled_seq = ''.join([shuffled_seq, last_chunk])

    return shuffled_seq

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
    @param  src_csv (string): CSV source file from which dataset is generated.
    @param  target_dir (string): which directory datasets are generated. Filename is created as 'train.csv', 'validation.csv', and 'test.csv'
    @param  train_frac (float): Default is 0.8, training fraction.
    @param  val_frac (float): Default is 0.1, validation fraction.
    @param  test_frac (float): Default is 0.1, testing fraction.
    @param  seed (int): Default is 1337, random state.
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

def _get_chr_fasta_path(chr_index, dirpath='./data/chr'):
    fpath = '{}/{}.fasta'.format(dirpath, chr_index)
    return fpath

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
    
def merge_dataset(prom_file, ss_file, polya_file, csv_target):
    """
    Merge data from each directory based on mentioned files.
    e.g. certain file from a directory will be merged with the same file from another directory.
    @param  prom_file (string):
    @param  ss_file (string):
    @param  polya_file (string):
    @param  csv_target (string):
    @return (boolean): True if success.
    """
    try:
        prom_df = pd.read_csv(prom_file)
        prom_df = prom_df.rename(columns={'label': 'label_prom'})
        prom_df['label_ss'] = 0
        prom_df['label_polya'] = 0

        ss_df = pd.read_csv(ss_file)
        ss_df = ss_df.rename(columns={'label': 'label_ss'})
        ss_df['label_prom'] = 0
        ss_df['label_polya'] = 0

        polya_df = pd.read_csv(ss_file)
        polya_df = polya_df.rename(columns={'label': 'label_polya'})
        polya_df['label_prom'] = 0
        polya_df['label_ss'] = 0

        target_df = pd.concat([prom_df, ss_df, polya_df])
        target_df.to_csv(csv_target, index=False)

        return True
    except Exception as e:
        print("Error {}".format(e))
        return False

def merge_prom_ss_polya_csv(csv_file_map, csv_target):
    """
    Merge dataset from promoter, splice site, and poly-a into one csv.
    @param  csv_file_map (object): object containing array of file paths.
            csv_file_map = {
                'prom': [<path1>, <path2>, ...],
                'ss': [<path1>, <path2>, ...],
                'polya': [<path1>, <path2>, ...],
            }
    @param csv_target (string): csv target path.
    @return (boolean): True if success.
    """
    try:
        if os.path.exists(csv_target):
            os.remove(csv_target)
        
        df = pd.DataFrame(columns=['sequence', 'label_prom', 'label_ss', 'label_polya'])
        prom_dataset = csv_file_map['prom']
        for path in prom_dataset:
            prom_df = pd.read_csv(path)
            for i, r in prom_df.iterrows():
                row = {}
                row['sequence'] = r['sequence']
                row['label_prom'] = r['label']
                df = df.append(row, ignore_index=True)

        ss_dataset = csv_file_map['ss']
        for path in ss_dataset:
            ss_df = pd.read_csv(path)
            for i, r in ss_df.iterrows():
                row = {}
                row['sequence'] = r['sequence']
                row['label_ss'] = r['label']
                df = df.append(row, ignore_index=True)

        polya_dataset = csv_file_map['polya']
        for path in polya_dataset:
            polya_df = pd.read_csv(path)
            for i, r in polya_df.iterrows():
                row = {}
                row['sequence'] = r['sequence']
                row['label_polya'] = r['label']
                df = df.append(row, ignore_index=True)

        df.to_csv(csv_target, index=False)
        return True
    except Exception as e:
        print('Error {}'.format(e))
        return False

def generate_polya_index_from_annotated_genome(annotated_genome_gff_path, target_csv_path, polya_keywords=['poly a', 'poly(a)', 'polya-a'], label=1, header='sequence_id,refseq,region,start_index,end_index,start,end,gene,gene_id,genbank,ensembl,product'):
    """
    Special purpose function to generate polya index.
    @param  annotated_genome_path (string): path to annotated genome GFF file.
    @param  target_csv (string): path to target file to write index.
    @param  label (int): default 1, means positive label or the genome segment contains Poly-A. Put zero to generate negative index.
    @return (boolean): True if success.
    """
    gff = {}
    target_csv = {}
    try:
        gff = open(annotated_genome_gff_path)
        if os.path.exists(target_csv_path):
            os.remove(target_csv_path)

        target_csv = open(target_csv_path, 'x')
        target_csv.write('{}\n'.format(header))
        for line in gff:
            gff_line = _gff_parseline(line)
            if gff_line:
                gff_line_product_desc = gff_line['product'].lower().strip()
                is_polya = _check_segment_product(polya_keywords, gff_line_product_desc)
                if is_polya:
                    target_csv.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        gff_line['sequence_id'],
                        gff_line['refseq'],
                        gff_line['region'],
                        gff_line['start_index'],
                        gff_line['end_index'],
                        gff_line['start'],
                        gff_line['end'],
                        gff_line['gene'],
                        gff_line['gene_id'],
                        gff_line['genbank'],
                        gff_line['ensembl'],
                        gff_line['product']
                    ))
        target_csv.close()
        gff.close()
        return True
    except Exception as e:
        print('Error {}'.format(e))
        print('Error Trace {}'.format(traceback.format_exc()))
        target_csv.close()
        gff.close()
        return False

def generate_polya_positive_dataset_from_index(polya_index_path, target_csv_path, tokenize=False):
    cur_seq = {} # current sequence stored in variable to avoid reading the same fasta multiple times.
    cur_seq_id = None
    try:
        if os.path.exists(target_csv_path):
            os.remove(target_csv_path)
        target_csv = open(target_csv_path, 'x')
        target_csv.write('{},{}\n'.format('sequence','label'))

        df_index = pd.read_csv(polya_index_path)
        for i, r in df_index.iterrows():
            sequence_id = r['sequence_id']
            start_index = r['start_index']
            end_index = r['end_index']

            sequence = ''
            if cur_seq_id == None:
                chr_fasta = SeqIO.parse(_get_chr_fasta_path(sequence_id), 'fasta')
                chr_fasta_record = next(chr_fasta) # Get first record only.
                chr_seq = str(chr_fasta_record.seq)

                cur_seq_id = sequence_id
                cur_seq = chr_seq
            else:
                if sequence_id != cur_seq_id:
                    cur_seq_id = sequence_id
                    chr_fasta = SeqIO.parse(_get_chr_fasta_path(sequence_id), 'fasta')
                    chr_fasta_record = next(chr_fasta) # Get first record only.
                    chr_seq = str(chr_fasta_record.seq)
                    cur_seq = chr_seq

            sequence = cur_seq[start_index:end_index + 1]
            if tokenize:
                sequence = ' '.join(kmer(sequence, 3))
            target_csv.write('{},{}\n'.format(sequence, 1))
        
        #endfor
        target_csv.close()

        return True
    except Exception as e:
        print('error {}'.format(e))
        print('error stacktrace {}'.format(traceback.format_exc()))
        return False

def generate_kmer_csv(src_csv_path, target_csv_path, kmer_size=3):
    try:
        if not os.path.exists(src_csv_path):
            raise Exception("File {} not found.".format(src_csv_path))

        if os.path.exists(target_csv_path):
            os.remove(target_csv_path)
        
        src_df = pd.read_csv(src_csv_path)
        len_src_df = len(src_df)
        _columns = src_df.columns.tolist()
        target_df = pd.DataFrame(columns=_columns)

        for i, r in src_df.iterrows():
            print("Generating kmer for <{}>: {}/{}".format(src_csv_path, i+1, len_src_df), end='\r')
            sequence = r['sequence']
            kmer_sequence = kmer(sequence, kmer_size)
            kmer_sequence = ' '.join(kmer_sequence)
            frame = pd.DataFrame([[kmer_sequence, r['label']]], columns=_columns)
            target_df = pd.concat([target_df, frame])
        #endfor
        target_df.to_csv(target_csv_path, index=False)
        return True
    except Exception as e:
        print("Error {}".format(e))
        return False

def generate_negative_dataset(positive_csv_path, target_csv_path, shuffle_chunk_size=16, negative_label=0):
    """
    Generate negative dataset by shuffling positive dataset. 
    Each instance in positive dataset is divided into several chunks and the order of those chunks are pseudorandomly rearranged.
    @param  positive_csv_path (string): path to positive csv.
    @param  target_csv_path (string): path to target csv to which negative or shuffled dataset is written.
    @param  shuffle_chunk_size (int): default value is 16.
    @param  negative_label (int): default value is 0. Label for negative data.
    @return (boolean): True if success.
    """
    try:
        if not os.path.exists(positive_csv_path):
            raise Exception("File {} not found.".format(positive_csv_path))
        
        src_df = pd.read_csv(positive_csv_path)
        target_csv_df = pd.DataFrame(columns=src_df.columns.tolist())
        for i, r in src_df.iterrows():
            sequence = r['sequence']
            shuffled_sequence = shuffle_sequence(sequence, chunk_size=shuffle_chunk_size)
            target_csv_df = target_csv_df.append({
                'sequence': shuffled_sequence,
                'label': negative_label
            }, ignore_index=True)
        # endfor
        target_csv_df.to_csv(target_csv_path, index=False)
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

def expand_and_split(src_csv, target_dir, stride=1, length=512, prefix='part'):
    """
    Expand each sequence in CSV source into fixed length.
    CSV source has column `sequence` and `label`. Each sequence is composed by tokens separated by space.
    @param      src_csv (string): path to csv source.
    @param      target_dir (string): directory to store expanded files.
    @param      stride (int): default is 1.
    @param      length (int): length of each expanded sequence, default is 512 tokens.
    @return     (boolean): True if success.
    """
    if not os.path.exists(src_csv):
        raise FileNotFoundError(src_csv)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    try:
        df = pd.read_csv(src_csv)
        _columns = df.columns.tolist()
        _part = 0
        for i, r in tqdm(df.iterrows(), total=df.shape[0]):
            arr_tokens = r['sequence'].strip().split(' ')
            label = r['label']
            arr_kmers = kmer(arr_tokens, length, stride)
            temp_df = pd.DataFrame(columns=_columns)
            for arr_kmer in arr_kmers:
                str_kmer = ' '.join(arr_kmer)
                _df = pd.DataFrame([[str_kmer, label]], columns=_columns)
                temp_df = pd.concat([temp_df, _df])
            #endfor
            _part += 1
            _path = "{}/{}_{}.csv".format(target_dir, prefix, _part)
            temp_df.to_csv(_path, index=False)
        #endfor
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False


def expand(src_csv, target_csv, sliding_window_size=1, col_to_expand='sequence', length=512):
    """
    Expand sequence in csv file. CSV must have 'sequence', 'label_prom', 'label_ss', 'label_polya' column in this precise order.
    Sequence is array of token (kmer) seperated by space.
    @param  src_csv (string): path to csv source file.
    @param  target_csv (string): path to csv target file.
    @param  col_to_expand (string): column whose content will be expanded.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if sucess.
    """
    try:
        df = pd.read_csv(src_csv)
        _columns = df.columns.tolist()
        _len_df = len(df)
        target_df = pd.DataFrame(columns=_columns)
        for i, r in df.iterrows():
            _i = i + 1
            if _i < _len_df:
                print("Processing {} [{}/{}]".format(src_csv, _i, _len_df), end='\r')
            else:
                print("Processing {} [{}/{}]".format(src_csv, _i, _len_df))
            sequence = r[col_to_expand].strip().split(' ')
            arr_sequence = kmer(sequence, length, window_size=sliding_window_size)
            for seq in arr_sequence:
                frame = pd.DataFrame([[seq, r['label_prom'], r['label_ss'], r['label_polya']]], columns=_columns)
                target_df = pd.concat([target_df, frame])
            #endfor
        #endfor
        target_df.to_csv(target_csv, index=False)
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

def expand_no_pandas(src_csv, target_csv, sliding_window_size=1, col_to_expand='sequence', length=512):
    """
    Expand sequence in csv file. CSV must have 'sequence', 'label_prom', 'label_ss', 'label_polya' column in this precise order.
    THIS IMPLEMENTATION DOES NOT USE PANDAS.
    Sequence is array of token (kmer) seperated by space.
    @param  src_csv (string): path to csv source file.
    @param  target_csv (string): path to csv target file.
    @param  col_to_expand (string): column whose content will be expanded.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if sucess.
    """
    target_file = {}
    src_file = {}
    try:
        if not os.path.exists(src_csv):
            raise Exception("File source {} not found.".format(src_csv))
        if os.path.exists(target_csv):
            os.remove(target_csv)
        
        _f = open(src_csv, 'r')
        _len_src = len(_f.readlines())
        _f.close()
        src_file = open(src_csv, 'r')
        next(src_file)
        _count = 0
        target_file = open(target_csv, 'x')
        target_file.write("{}\n".format('sequence,label_prom,label_ss,label_polya'.strip()))
        for line in src_file:
            _count += 1
            if _count < _len_src:
                print("Expanding {} [{}/{}]".format(src_csv, _count, _len_src), end='\r') 
            else:
                print("Expanding {} [{}/{}]".format(src_csv, _count, _len_src)) 

            arr_line = line.strip().split(',')
            sequence = arr_line[0]
            label_prom = arr_line[1]
            label_ss = arr_line[2]
            label_polya = arr_line[3]
            sequence_to_kmer = sequence.split(' ')
            arr_sequence = kmer(sequence_to_kmer, length, window_size=sliding_window_size)
            for seq in arr_sequence:
                str_seq = ' '.join(seq)
                entry = '{},{},{},{}\n'.format(str_seq, label_prom, label_ss, label_polya)
                target_file.write(entry)

        src_file.close()
        target_file.close()
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        if src_file != {}:
            src_file.close()
        if target_file != {}:
            target_file.close()
        return False

def expand_by_sliding_window(src_csv, target_csv, sliding_window_size=1, length=512):
    """
    Expand sequence in csv file. CSV must have 'sequence' and 'label' column.
    Sequence is array of token (kmer) seperated by space.
    @param  src_csv (string): path to csv source file.
    @param  target_csv (string): path to csv target file.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if sucess.
    """
    try:
        if not os.path.exists(src_csv):
            raise Exception("File {} not found.".format(src_csv))
        src_df = pd.read_csv(src_csv)
        src_df_len = len(src_df)
        _columns = src_df.columns.tolist()

        # Remove existing target_csv.
        if os.path.exists(target_csv):
            os.remove(target_csv)

        target_df = pd.DataFrame(columns=_columns)
        for i, r in src_df.iterrows():
            sequence = r['sequence'].split(' ')
            label = r['label']
            expanded_seq = kmer(sequence, length)
            print("Expanding source {}: {}/{}".format(src_csv, i+1, src_df_len), end='\r')
            for seq in expanded_seq:
                seq = ' '.join(seq)
                frame = pd.DataFrame([[seq, label]], columns=_columns)
                target_df = pd.concat([target_df, frame])
        #endfor
        target_df.to_csv(target_csv, index=False)
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

def expand_complete_data(src_csv, target_csv, stride=1, length=512):
    """
    Expand sequence in csv file. CSV must have `sequence`, `label_prom`, `label_ss`, and `label_polya`.
    Sequence in column `sequence` is array of token (kmer) separated by space.
    @param      src_csv (string):
    @param      target_csv (string):
    @param      stride (int):
    @param      length (int):
    @return     (boolean): True if success.
    """
    try:
        if not os.path.exists(src_csv):
            raise Exception("File {} not found.".format(src_csv))
        src_df = pd.read_csv(src_csv)
        src_df_len = len(src_df)
        _columns = src_df.columns.tolist()

        # Remove existing target_csv.
        if os.path.exists(target_csv):
            os.remove(target_csv)

        target_df = pd.DataFrame(columns=_columns)
        for i, r in src_df.iterrows():
            sequence = r['sequence'].split(' ')
            label_prom = r['label_prom']
            label_ss = r['label_ss']
            label_polya = r['label_polya']
            expanded_seq = kmer(sequence, length)
            print("Expanding source {}: {}/{}".format(src_csv, i+1, src_df_len), end='\r')
            for seq in expanded_seq:
                seq = ' '.join(seq)
                frame = pd.DataFrame([[seq, label_prom, label_ss, label_polya]], columns=_columns)
                target_df = pd.concat([target_df, frame])
        #endfor
        target_df.to_csv(target_csv, index=False)
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False

def expand_by_sliding_window_no_pandas(src_csv, target_csv, sliding_window_size=1, length=512):
    """
    Expand sequence in csv file. CSV must have 'sequence' and 'label' column.
    Sequence is array of token (kmer) separated by space.
    THIS FUNCTION DOESN'T USE PANDAS!
    @param  src_csv (string): path to csv source file.
    @param  target_csv (string): path to csv target file.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if sucess.
    """
    src_file = {}
    target_file = {}
    try:
        if not os.path.exists(src_csv):
            raise Exception("File {} not found.".format(src_csv))
        if os.path.exists(target_csv):
            os.remove(target_csv)

        _f = open(src_csv, 'r')
        _len = len(_f.readlines())
        _f.close()

        src_file = open(src_csv, 'r')
        next(src_file) # Skip header.
        target_file = open(target_csv, 'x')
        target_file.write('sequence,label\n')
        _count = 0
        for line in src_file:
            _count += 1
            arr_line = line.strip().split(',')
            sequence = arr_line[0]
            label = arr_line[1]
            tokens = sequence.split(' ')
            arr_subtokens = kmer(tokens, 510, window_size=sliding_window_size)
            if _count < _len:
                print('Expanding {} [{}/{}]'.format(src_csv, _count, _len), end='\r')
            else:
                print('Expanding {} [{}/{}]'.format(src_csv, _count, _len))
            for seq in arr_subtokens:
                str_seq = ' '.join(seq)
                entry = "{},{}\n".format(str_seq, label)
                target_file.write(entry)
            #endfor
        #endfor
        target_file.close()
        src_file.close()
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        if src_file != {}:
            src_file.close()
        if target_file != {}:
            target_file.close()
        return False

def expand_files_in_dir(dir_path, sliding_window_size=1, length=512):
    """
    @param  dir_path (string): path to directory.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if success.
    """
    try:
        if not os.path.isdir(dir_path):
            raise Exception("Path <{}> is not valid directory.".format(dir_path))
        
        for file in os.listdir(dir_path):
            src_csv_path = "{}/{}".format(dir_path, file)
            if os.path.isfile(src_csv_path):
                target_csv_path = "{}/{}_expanded.csv".format(dir_path, file.split('.')[0])
                if not expand_by_sliding_window(src_csv_path, target_csv_path, sliding_window_size=sliding_window_size, length=length):
                    raise Exception("Failed executing sequence expansion.")
        #endfor            
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        return False