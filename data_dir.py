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
dataset_dir = './dataset'
dataset_full_dir = '{}/full'.format(dataset_dir)
dataset_full_prom_dir = '{}/promoter'.format(dataset_full_dir)
dataset_full_ss_dir = '{}/splice-sites'.format(dataset_full_dir)
dataset_full_polya_dir = '{}/polya'.format(dataset_full_dir)
dataset_sample_dir = '{}/sample'.format(dataset_dir)
dataset_sample_prom_dir = '{}/promoter'.format(dataset_sample_dir)
dataset_sample_ss_dir = '{}/splice-sites'.format(dataset_sample_dir)
dataset_sample_polya_dir = '{}/polya'.format(dataset_sample_dir)
dataset_full_prom_train_csv = "{}/train.csv".format(dataset_full_prom_dir)
dataset_full_polya_train_csv = "{}/train.csv".format(dataset_full_polya_dir)
dataset_full_ss_train_csv = "{}/train.csv".format(dataset_full_ss_dir)
"""
Samples.
"""
sample_dir = './sample'
sample_prom_dir = '{}/promoter'.format(sample_dir)
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
data_dir = './data'
data_epd_dir = '{}/epd'.format(data_dir)
epd_tata = '{}/human_tata.fasta'.format(data_epd_dir)
epd_pos_tata_csv = '{}/human_tata.csv'.format(data_epd_dir)
epd_neg_tata_csv = '{}/human_non_tata.csv'.format(data_epd_dir)
epd_pos_tata_kmer_csv = '{}/human_tata_kmer.csv'.format(data_epd_dir)
epd_neg_tata_kmer_csv = '{}/human_non_tata_kmer.csv'.format(data_epd_dir)
epd_pos_tata_kmer_dir = '{}/human_tata_kmer'.format(data_epd_dir)
epd_neg_tata_kmer_dir = '{}/human_non_tata_kmer'.format(data_epd_dir)
epd_pos_tata_train_csv = '{}/train_expanded.csv'.format(epd_pos_tata_kmer_dir)
epd_neg_tata_train_csv = '{}/train_expanded.csv'.format(epd_neg_tata_kmer_dir)
epd_pos_tata_validation_csv = '{}/validation_expanded.csv'.format(epd_pos_tata_kmer_dir)
epd_neg_tata_validation_csv = '{}/validation_expanded.csv'.format(epd_neg_tata_kmer_dir)
epd_pos_tata_test_csv = '{}/test_expanded.csv'.format(epd_pos_tata_kmer_dir)
epd_neg_tata_test_csv = '{}/test_expanded.csv'.format(epd_neg_tata_kmer_dir)
epd_train_csv = '{}/train.csv'.format(data_epd_dir)
epd_validation_csv = '{}/validation.csv'.format(data_epd_dir)
epd_test_csv = '{}/test.csv'.format(data_epd_dir)

"""
Chromosome and genome.
"""
data_chr_dir = '{}/chr'.format(data_dir)
data_genome_dir = '{}/genome'.format(data_dir)
data_genome_grch38_dir = '{}/grch38'.format(data_genome_dir)
data_genome_grch38_index_dir = '{}/exon'.format(data_genome_grch38_dir)

annotated_grch38_gff = '{}/GRCh38_latest_genomic.gff'.format(data_genome_grch38_dir)
annotated_grch38_gff_dir = '{}/csvs'.format(data_genome_grch38_dir)
annotated_grch38_gtf = '{}/GCF_000001405.39_GRCh38.p13_genomic.gtf'.format(data_genome_grch38_dir)
annotated_grch38_gff_csv = '{}/grch38_gff.csv'.format(data_genome_grch38_dir)
annotated_grch38_gtf_csv = '{}/grch38_gtf.csv'.format(data_genome_grch38_dir)

chr1_index_csv = '{}/NC_000001.11.csv'.format(data_genome_grch38_index_dir)
chr2_index_csv = '{}/NC_000002.12.csv'.format(data_genome_grch38_index_dir)
chr3_index_csv = '{}/NC_000003.12.csv'.format(data_genome_grch38_index_dir)
chr4_index_csv = '{}/NC_000004.12.csv'.format(data_genome_grch38_index_dir)
chr5_index_csv = '{}/NC_000005.10.csv'.format(data_genome_grch38_index_dir)
chr6_index_csv = '{}/NC_000006.12.csv'.format(data_genome_grch38_index_dir)
chr7_index_csv = '{}/NC_000007.14.csv'.format(data_genome_grch38_index_dir)
chr8_index_csv = '{}/NC_000008.11.csv'.format(data_genome_grch38_index_dir)
chr9_index_csv = '{}/NC_000009.12.csv'.format(data_genome_grch38_index_dir)
chr10_index_csv = '{}/NC_000010.11.csv'.format(data_genome_grch38_index_dir)
chr11_index_csv = '{}/NC_000011.10.csv'.format(data_genome_grch38_index_dir)
chr12_index_csv = '{}/NC_000012.12.csv'.format(data_genome_grch38_index_dir)
chr13_index_csv = '{}/NC_000013.11.csv'.format(data_genome_grch38_index_dir)
chr14_index_csv = '{}/NC_000014.9.csv'.format(data_genome_grch38_index_dir)
chr15_index_csv = '{}/NC_000015.10.csv'.format(data_genome_grch38_index_dir)
chr16_index_csv = '{}/NC_000016.10.csv'.format(data_genome_grch38_index_dir)
chr17_index_csv = '{}/NC_000017.11.csv'.format(data_genome_grch38_index_dir)
chr18_index_csv = '{}/NC_000018.10.csv'.format(data_genome_grch38_index_dir)
chr19_index_csv = '{}/NC_000019.10.csv'.format(data_genome_grch38_index_dir)
chr20_index_csv = '{}/NC_000020.11.csv'.format(data_genome_grch38_index_dir)
chr21_index_csv = '{}/NC_000021.9.csv'.format(data_genome_grch38_index_dir)
chr22_index_csv = '{}/NC_000022.11.csv'.format(data_genome_grch38_index_dir)
chr23_index_csv = '{}/NC_000023.11.csv'.format(data_genome_grch38_index_dir)
chr24_index_csv = '{}/NC_000024.10.csv'.format(data_genome_grch38_index_dir)

labseq_dir = "{}/labseq".format(data_genome_dir)
labseq_names = [
    'chr1.csv','chr2.csv','chr3.csv','chr4.csv','chr5.csv','chr6.csv','chr7.csv','chr8.csv','chr9.csv','chr10.csv',
    'chr11.csv','chr12.csv','chr13.csv','chr14.csv','chr15.csv','chr16.csv','chr17.csv','chr18.csv','chr19.csv','chr20.csv',
    'chr21.csv','chr22.csv','chr23.csv','chr24.csv'
]

"""
Poly A.
"""
polya_grch38_dir = '{}/poly-a/grch38'.format(data_dir)
polya_grch38_index_csv = '{}/polya_index.csv'.format(polya_grch38_dir)
polya_grch38_positive_csv = '{}/polya_positive.csv'.format(polya_grch38_dir)
polya_grch38_negative_csv = '{}/polya_negative.csv'.format(polya_grch38_dir)
polya_grch38_positive_kmer_csv = '{}/polya_positive_kmer.csv'.format(polya_grch38_dir)
polya_grch38_negative_kmer_csv = '{}/polya_negative_kmer.csv'.format(polya_grch38_dir)
polya_grch38_positive_dir = '{}/positive'.format(polya_grch38_dir)
polya_grch38_negative_dir = '{}/negative'.format(polya_grch38_dir)
polya_grch38_train_csv = '{}/train.csv'.format(polya_grch38_dir)
polya_grch38_validation_csv = '{}/validation.csv'.format(polya_grch38_dir)
polya_grch38_test_csv = '{}/test.csv'.format(polya_grch38_dir)

"""
Splice sites.
"""
ss_dir = "{}/splice-sites/splice-deep".format(data_dir)
ss_pos_acc_dir = "{}/pos_acc".format(ss_dir)
ss_pos_don_dir = "{}/pos_don".format(ss_dir)
ss_neg_acc_dir = "{}/neg_acc".format(ss_dir)
ss_neg_don_dir = "{}/neg_don".format(ss_dir)

ss_pos_acc_hs_csv = "{}/pos_ss_acc_hs.csv".format(ss_dir)
ss_pos_don_hs_csv = "{}/pos_ss_don_hs.csv".format(ss_dir)
ss_neg_acc_hs_csv = "{}/neg_ss_acc_hs.csv".format(ss_dir)
ss_neg_don_hs_csv = "{}/neg_ss_don_hs.csv".format(ss_dir)

ss_pos_acc_hs_kmer_csv = "{}/pos_ss_acc_hs_kmer.csv".format(ss_dir)
ss_pos_don_hs_kmer_csv = "{}/pos_ss_don_hs_kmer.csv".format(ss_dir)
ss_neg_acc_hs_kmer_csv = "{}/neg_ss_acc_hs_kmer.csv".format(ss_dir)
ss_neg_don_hs_kmer_csv = "{}/neg_ss_don_hs_kmer.csv".format(ss_dir)

ss_pos_acc_hs_non_kmer_csv = "{}/pos_ss_acc_hs_non_kmer.csv".format(ss_dir)
ss_pos_don_hs_non_kmer_csv = "{}/pos_ss_don_hs_non_kmer.csv".format(ss_dir)
ss_neg_acc_hs_non_kmer_csv = "{}/neg_ss_acc_hs_non_kmer.csv".format(ss_dir)
ss_neg_don_hs_non_kmer_csv = "{}/neg_ss_don_hs_non_kmer.csv".format(ss_dir)

ss_pos_acc_hs_fasta = "{}/positive_DNA_seqs_acceptor_hs.fa"
ss_pos_don_hs_fasta = "{}/positive_DNA_seqs_donor_hs.fa"
ss_neg_acc_hs_fasta = "{}/negative_DNA_seqs_acceptor_hs.fa"
ss_neg_don_hs_fasta = "{}/negative_DNA_seqs_donor_hs.fa"

ss_train_csv = "{}/train.csv".format(ss_dir)
ss_validation_csv = "{}/validation.csv".format(ss_dir)
ss_test_csv = "{}/test.csv".format(ss_dir)

"""
`result` folder.
"""
result_dir = "./result"
pretrained_dir = "./pretrained"
pretrained_3kmer_dir = "{}/3-new-12w-0".format(pretrained_dir)

"""
`logs` folder.
"""
log_dir = './log'
def get_current_log_folder():
    import datetime
    import os
    _now = datetime.datetime.now()
    _str_now = _now.strftime('%Y-%m-%d')
    _log_dir = "{}/{}".format(log_dir, _str_now)
    if not os.path.exists(_log_dir):
        os.mkdir(_log_dir)
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
workspace_dir = './workspace'
if not os.path.exists(workspace_dir):
    os.mkdir(workspace_dir)

"""
`chr` folder.
This folder contains chromosome fasta files.
"""
chr1_fasta = '{}/NC_000001.11.fasta'.format(data_chr_dir)
chr2_fasta = '{}/NC_000002.12.fasta'.format(data_chr_dir)
chr3_fasta = '{}/NC_000003.12.fasta'.format(data_chr_dir)
chr4_fasta = '{}/NC_000004.12.fasta'.format(data_chr_dir)
chr5_fasta = '{}/NC_000005.10.fasta'.format(data_chr_dir)
chr6_fasta = '{}/NC_000006.12.fasta'.format(data_chr_dir)
chr7_fasta = '{}/NC_000007.14.fasta'.format(data_chr_dir)
chr8_fasta = '{}/NC_000008.11.fasta'.format(data_chr_dir)
chr9_fasta = '{}/NC_000009.12.fasta'.format(data_chr_dir)
chr10_fasta = '{}/NC_000010.11.fasta'.format(data_chr_dir)
chr11_fasta = '{}/NC_000011.10.fasta'.format(data_chr_dir)
chr12_fasta = '{}/NC_000012.12.fasta'.format(data_chr_dir)
chr13_fasta = '{}/NC_000013.11.fasta'.format(data_chr_dir)
chr14_fasta = '{}/NC_000014.9.fasta'.format(data_chr_dir)
chr15_fasta = '{}/NC_000015.10.fasta'.format(data_chr_dir)
chr16_fasta = '{}/NC_000016.10.fasta'.format(data_chr_dir)
chr17_fasta = '{}/NC_000017.11.fasta'.format(data_chr_dir)
chr18_fasta = '{}/NC_000018.10.fasta'.format(data_chr_dir)
chr19_fasta = '{}/NC_000019.10.fasta'.format(data_chr_dir)
chr20_fasta = '{}/NC_000020.11.fasta'.format(data_chr_dir)
chr21_fasta = '{}/NC_000021.9.fasta'.format(data_chr_dir)
chr22_fasta = '{}/NC_000022.11.fasta'.format(data_chr_dir)
chr23_fasta = '{}/NC_000023.11.fasta'.format(data_chr_dir)
chr24_fasta = '{}/NC_000024.10.fasta'.format(data_chr_dir)