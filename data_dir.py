import os

"""
Generic column format for CSV files.
"""
_partial_dataset_columns = ['sequence', 'label']
_complete_dataset_columns = ['sequence', 'label_prom', 'label_ss', 'label_polya']

"""
Generic file names for training data, validation data, and testing data.
"""
_generic_filenames = ['train.csv', 'validation.csv', 'test.csv']

"""
File paths for Promoter, SS, and Poly-A.
Datasets: Promoter, SS, and Poly-A
"""
dataset_dir = os.path.join("dataset")
dataset_full_dir = os.path.join(dataset_dir, "full")
dataset_full_prom_dir = os.path.join(dataset_full_dir, "promoter")
dataset_full_ss_dir = os.path.join(dataset_full_dir, "splice-sites")
dataset_full_polya_dir = os.path.join(dataset_full_dir, "polya")
dataset_sample_dir = os.path.join(dataset_dir, "sample")
dataset_sample_prom_dir = os.path.join(dataset_sample_dir, "promoter")
dataset_sample_ss_dir = os.path.join(dataset_sample_dir, "splice-sites")
dataset_sample_polya_dir = os.path.join(dataset_sample_dir, "polya")
dataset_full_prom_train_csv = os.path.join(dataset_full_prom_dir, "train.csv")
dataset_full_polya_train_csv = os.path.join(dataset_full_polya_dir, "train.csv")
dataset_full_ss_train_csv = os.path.join(dataset_full_ss_dir, "train.csv")
"""
Samples.
"""
sample_dir = os.path.join('sample')
sample_prom_dir = os.path.join(sample_dir, "promoter")
sample_prom_pos_dir = '{}/positive'.format(sample_prom_dir)
sample_prom_neg_dir = '{}/negative'.format(sample_prom_dir)
sample_prom_train_csv = '{}/train.csv'.format(sample_prom_dir)
sample_prom_validation_csv = '{}/validation.csv'.format(sample_prom_dir)
sample_prom_test_csv = '{}/test.csv'.format(sample_prom_dir)

sample_ss_dir = '{}/splice-sites'.format(sample_dir)
sample_ss_pos_acc_dir = "{}/pos_acc".format(sample_ss_dir)
sample_ss_pos_don_dir = "{}/pos_don".format(sample_ss_dir)
sample_ss_neg_acc_dir = "{}/neg_acc".format(sample_ss_dir)
sample_ss_neg_don_dir = "{}/neg_don".format(sample_ss_dir)

sample_ss_train_csv = '{}/train.csv'.format(sample_ss_dir)
sample_ss_validation_csv = '{}/validation.csv'.format(sample_ss_dir)
sample_ss_test_csv = '{}/test.csv'.format(sample_ss_dir)

sample_polya_dir = '{}/polya'.format(sample_dir)
sample_polya_pos_dir = "{}/positive".format(sample_polya_dir)
sample_polya_neg_dir = "{}/negative".format(sample_polya_dir)
sample_polya_train_csv = '{}/train.csv'.format(sample_polya_dir)
sample_polya_validation_csv = '{}/validation.csv'.format(sample_polya_dir)
sample_polya_test_csv = '{}/test.csv'.format(sample_polya_dir)

"""
`data` folder contains raw data for promoter, splice sites, and poly A. 
The end products of this folder are `train.csv`, `validation.csv`, and `test.csv` files.
Sequence in these files is tokenized and each token is separated by single space.
"""
data_dir = os.path.join('data')
data_epd_dir = os.path.join(data_dir, 'epd')
epd_tata = os.path.join(data_epd_dir, 'human_tata.fasta')
epd_pos_tata_csv = os.path.join(data_epd_dir, 'human_tata.csv')
epd_neg_tata_csv = os.path.join(data_epd_dir, 'human_non_tata.csv')
epd_pos_tata_kmer_csv = os.path.join(data_epd_dir, 'human_tata_kmer.csv')
epd_neg_tata_kmer_csv = os.path.join(data_epd_dir, 'human_non_tata_kmer.csv')
epd_pos_tata_kmer_dir = os.path.join(data_epd_dir, 'human_tata_kmer')
epd_neg_tata_kmer_dir = os.path.join(data_epd_dir, 'human_non_tata_kmer')
epd_pos_tata_train_csv = os.path.join(epd_pos_tata_kmer_dir, 'train_expanded.csv')
epd_neg_tata_train_csv = os.path.join(epd_neg_tata_kmer_dir, 'train_expanded.csv')
epd_pos_tata_validation_csv = os.path.join(epd_pos_tata_kmer_dir, 'validation_expanded.csv')
epd_neg_tata_validation_csv = os.path.join(epd_neg_tata_kmer_dir, 'validation_expanded.csv')
epd_pos_tata_test_csv = os.path.join(epd_pos_tata_kmer_dir, 'test_expanded.csv')
epd_neg_tata_test_csv = os.path.join(epd_neg_tata_kmer_dir, 'test_expanded.csv')
epd_train_csv = os.path.join(data_epd_dir, 'train.csv')
epd_validation_csv = os.path.join(data_epd_dir, 'validation.csv')
epd_test_csv = os.path.join(data_epd_dir, 'test.csv')

"""
Chromosome and genome.
"""
data_chr_dir = os.path.join(data_dir, 'chr')
data_genome_dir = os.path.join(data_dir, 'genome')
data_genome_grch38_dir = os.path.join(data_genome_dir, 'grch38')
data_genome_grch38_index_dir = os.path.join(data_genome_grch38_dir, 'csvs')
data_genome_grch38_exon_dir = os.path.join(data_genome_grch38_dir, 'exon')
data_genome_grch38_labels_dir = os.path.join(data_genome_grch38_dir, 'labels')

annotated_grch38_gff = os.path.join(data_genome_grch38_dir, 'GRCh38_latest_genomic.gff')
annotated_grch38_gff_dir = os.path.join(data_genome_grch38_dir, 'csvs')
annotated_grch38_gtf = os.path.join(data_genome_grch38_dir, 'GCF_000001405.39_GRCh38.p13_genomic.gtf')
annotated_grch38_gff_csv = os.path.join(data_genome_grch38_dir, 'grch38_gff.csv')
annotated_grch38_gtf_csv = os.path.join(data_genome_grch38_dir, 'grch38_gtf.csv')

chr1_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000001.11.csv')
chr2_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000002.12.csv')
chr3_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000003.12.csv')
chr4_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000004.12.csv')
chr5_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000005.10.csv')
chr6_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000006.12.csv')
chr7_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000007.14.csv')
chr8_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000008.11.csv')
chr9_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000009.12.csv')
chr10_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000010.11.csv')
chr11_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000011.10.csv')
chr12_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000012.12.csv')
chr13_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000013.11.csv')
chr14_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000014.9.csv')
chr15_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000015.10.csv')
chr16_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000016.10.csv')
chr17_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000017.11.csv')
chr18_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000018.10.csv')
chr19_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000019.10.csv')
chr20_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000020.11.csv')
chr21_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000021.9.csv')
chr22_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000022.11.csv')
chr23_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000023.11.csv')
chr24_index_csv = os.path.join(data_genome_grch38_index_dir, 'NC_000024.10.csv')

chr_fasta_mapname = {
    'chr1': 'NC_000001.11.fasta',
    'chr2': 'NC_000002.12.fasta',
    'chr3': 'NC_000003.12.fasta',
    'chr4': 'NC_000004.12.fasta',
    'chr5': 'NC_000005.10.fasta',
    'chr6': 'NC_000006.12.fasta',
    'chr7': 'NC_000007.14.fasta',
    'chr8': 'NC_000008.11.fasta',
    'chr9': 'NC_000009.12.fasta',
    'chr10': 'NC_000010.11.fasta',
    'chr11': 'NC_000011.10.fasta',
    'chr12': 'NC_000012.12.fasta',
    'chr13': 'NC_000013.11.fasta',
    'chr14': 'NC_000014.9.fasta',
    'chr15': 'NC_000015.10.fasta',
    'chr16': 'NC_000016.10.fasta',
    'chr17': 'NC_000017.11.fasta',
    'chr18': 'NC_000018.10.fasta',
    'chr19': 'NC_000019.10.fasta',
    'chr20': 'NC_000020.11.fasta',
    'chr21': 'NC_000021.9.fasta',
    'chr22': 'NC_000022.11.fasta',
    'chr23': 'NC_000023.11.fasta',
    'chr24': 'NC_000024.10.fasta',
}

chr_index_csvs = [
    chr1_index_csv,
    chr2_index_csv,
    chr3_index_csv,
    chr4_index_csv,
    chr5_index_csv,
    chr6_index_csv,
    chr7_index_csv,
    chr8_index_csv,
    chr9_index_csv,
    chr10_index_csv,
    chr11_index_csv,
    chr12_index_csv,
    chr13_index_csv,
    chr14_index_csv,
    chr15_index_csv,
    chr16_index_csv,
    chr17_index_csv,
    chr18_index_csv,
    chr19_index_csv,
    chr20_index_csv,
    chr21_index_csv,
    chr22_index_csv,
    chr23_index_csv,
    chr24_index_csv
]

labseq_dir = os.path.join(data_genome_dir, 'labseq')
labseq_names = [
    'chr1.csv','chr2.csv','chr3.csv','chr4.csv','chr5.csv','chr6.csv','chr7.csv','chr8.csv','chr9.csv','chr10.csv',
    'chr11.csv','chr12.csv','chr13.csv','chr14.csv','chr15.csv','chr16.csv','chr17.csv','chr18.csv','chr19.csv','chr20.csv',
    'chr21.csv','chr22.csv','chr23.csv','chr24.csv'
]

labseq_chr_files = [os.path.join(labseq_dir, a) for a in labseq_names]

"""
Poly A.
"""
polya_grch38_dir = os.path.join(data_dir, 'poly-a/grch38') 
polya_grch38_index_csv = os.path.join(polya_grch38_dir, 'polya_index.csv')
polya_grch38_positive_csv = os.path.join(polya_grch38_dir, 'polya_positive.csv')
polya_grch38_negative_csv = os.path.join(polya_grch38_dir, 'polya_negative.csv')
polya_grch38_positive_kmer_csv = os.path.join(polya_grch38_dir, 'polya_positive_kmer.csv')
polya_grch38_negative_kmer_csv = os.path.join(polya_grch38_dir, 'polya_negative_kmer.csv')
polya_grch38_positive_dir = os.path.join(polya_grch38_dir, 'positive')
polya_grch38_negative_dir = os.path.join(polya_grch38_dir, 'negative')
polya_grch38_train_csv = os.path.join(polya_grch38_dir, 'train.csv')
polya_grch38_validation_csv = os.path.join(polya_grch38_dir, 'validation.csv')
polya_grch38_test_csv = os.path.join(polya_grch38_dir, 'test.csv')

"""
Splice sites.
"""
ss_dir = os.path.join(data_dir, "splice-sites/splice-deep")
ss_pos_acc_dir = os.path.join(ss_dir, "pos_acc")
ss_pos_don_dir = os.path.join(ss_dir, "pos_don")
ss_neg_acc_dir = os.path.join(ss_dir, "neg_acc")
ss_neg_don_dir = os.path.join(ss_dir, "neg_don")

ss_pos_acc_hs_csv = os.path.join(ss_dir, "pos_ss_acc_hs.csv")
ss_pos_don_hs_csv = os.path.join(ss_dir, "pos_ss_don_hs.csv")
ss_neg_acc_hs_csv = os.path.join(ss_dir, "neg_ss_acc_hs.csv")
ss_neg_don_hs_csv = os.path.join(ss_dir, "neg_ss_don_hs.csv")

ss_pos_acc_hs_kmer_csv = os.path.join(ss_dir, "pos_ss_acc_hs_kmer.csv")
ss_pos_don_hs_kmer_csv = os.path.join(ss_dir, "pos_ss_don_hs_kmer.csv")
ss_neg_acc_hs_kmer_csv = os.path.join(ss_dir, "neg_ss_acc_hs_kmer.csv")
ss_neg_don_hs_kmer_csv = os.path.join(ss_dir, "neg_ss_don_hs_kmer.csv")

ss_pos_acc_hs_non_kmer_csv = os.path.join(ss_dir, "pos_ss_acc_hs_non_kmer.csv")
ss_pos_don_hs_non_kmer_csv = os.path.join(ss_dir, "pos_ss_don_hs_non_kmer.csv")
ss_neg_acc_hs_non_kmer_csv = os.path.join(ss_dir, "neg_ss_acc_hs_non_kmer.csv")
ss_neg_don_hs_non_kmer_csv = os.path.join(ss_dir, "neg_ss_don_hs_non_kmer.csv")

ss_pos_acc_hs_fasta = os.path.join(ss_dir, "positive_DNA_seqs_acceptor_hs.fa")
ss_pos_don_hs_fasta = os.path.join(ss_dir, "positive_DNA_seqs_donor_hs.fa")
ss_neg_acc_hs_fasta = os.path.join(ss_dir, "negative_DNA_seqs_acceptor_hs.fa")
ss_neg_don_hs_fasta = os.path.join(ss_dir, "negative_DNA_seqs_donor_hs.fa")

ss_train_csv = os.path.join(ss_dir, "train.csv")
ss_validation_csv = os.path.join(ss_dir, "validation.csv")
ss_test_csv = os.path.join(ss_dir, "test.csv")

"""
`result` folder.
"""
result_dir = os.path.join('result')
"""
`pretrained` folder.
"""
pretrained_dir = os.path.join('pretrained')
pretrained_3kmer_dir = os.path.join(pretrained_dir, "3-new-12w-0")

"""
`logs` folder.
"""
log_dir = os.path.join('log')
def get_current_log_folder(root=log_dir):
    import datetime
    import os
    _now = datetime.datetime.now()
    _str_now = _now.strftime('%Y-%m-%d')
    _log_dir = os.path.join(log_dir, _str_now)
    if not os.path.exists(_log_dir):
        os.makedirs(_log_dir, exist_ok=True)
    return _log_dir

def get_current_log_name():
    import datetime
    import os
    _now = datetime.datetime.now()
    _str_now = _now.strftime('%Y-%m-%d-%H-%M-%S')
    return "{}.csv".format(_str_now)

"""
`workspace` folder.
`workspace` folder is used for training, validating, testing, and experiments.
"""
workspace_dir = os.path.join('workspace')
if not os.path.exists(workspace_dir):
    os.mkdir(workspace_dir)

workspace_promoter_dir = os.path.join(workspace_dir, 'promoter')
workspace_ss_dir = os.path.join(workspace_dir, 'ss')
workspace_polya_dir = os.path.join(workspace_dir, 'polya')
"""
`chr` folder.
This folder contains chromosome fasta files.
"""
chr1_fasta = os.path.join(data_chr_dir, 'NC_000001.11.fasta')
chr2_fasta = os.path.join(data_chr_dir, 'NC_000002.12.fasta')
chr3_fasta = os.path.join(data_chr_dir, 'NC_000003.12.fasta')
chr4_fasta = os.path.join(data_chr_dir, 'NC_000004.12.fasta')
chr5_fasta = os.path.join(data_chr_dir, 'NC_000005.10.fasta')
chr6_fasta = os.path.join(data_chr_dir, 'NC_000006.12.fasta')
chr7_fasta = os.path.join(data_chr_dir, 'NC_000007.14.fasta')
chr8_fasta = os.path.join(data_chr_dir, 'NC_000008.11.fasta')
chr9_fasta = os.path.join(data_chr_dir, 'NC_000009.12.fasta')
chr10_fasta = os.path.join(data_chr_dir, 'NC_000010.11.fasta')
chr11_fasta = os.path.join(data_chr_dir, 'NC_000011.10.fasta')
chr12_fasta = os.path.join(data_chr_dir, 'NC_000012.12.fasta')
chr13_fasta = os.path.join(data_chr_dir, 'NC_000013.11.fasta')
chr14_fasta = os.path.join(data_chr_dir, 'NC_000014.9.fasta')
chr15_fasta = os.path.join(data_chr_dir, 'NC_000015.10.fasta')
chr16_fasta = os.path.join(data_chr_dir, 'NC_000016.10.fasta')
chr17_fasta = os.path.join(data_chr_dir, 'NC_000017.11.fasta')
chr18_fasta = os.path.join(data_chr_dir, 'NC_000018.10.fasta')
chr19_fasta = os.path.join(data_chr_dir, 'NC_000019.10.fasta')
chr20_fasta = os.path.join(data_chr_dir, 'NC_000020.11.fasta')
chr21_fasta = os.path.join(data_chr_dir, 'NC_000021.9.fasta')
chr22_fasta = os.path.join(data_chr_dir, 'NC_000022.11.fasta')
chr23_fasta = os.path.join(data_chr_dir, 'NC_000023.11.fasta')
chr24_fasta = os.path.join(data_chr_dir, 'NC_000024.10.fasta')

"""
`rawdata` folder.
"""
raw_data_dir = os.path.join('rawdata')
raw_data_promoter_dir = os.path.join(raw_data_dir, 'promoter')
raw_data_ss_dir = os.path.join(raw_data_dir, 'splice-sites')
raw_data_polya_dir = os.path.join(raw_data_dir, 'poly-a')
