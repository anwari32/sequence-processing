"""
File paths for Promoter, SS, and Poly-A.
Datasets: Promoter, SS, and Poly-A
"""
dataset_dir = './dataset'
dataset_sample_dir = '{}/sample'.format(dataset_dir)
dataset_sample_prom_dir = '{}/promoter'.format(dataset_sample_dir)
dataset_sample_ss_dir = '{}/splice-sites'.format(dataset_sample_dir)
dataset_sample_polya_dir = '{}/polya'.format(dataset_sample_dir)


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

data_chr_dir = '{}/chr'.format(data_dir)
data_genome_dir = '{}/genome'.format(data_dir)
data_genome_grch38_dir = '{}/grch38'.format(data_genome_dir)

annotated_grch38_gff = '{}/GRCh38_latest_genomic.gff'.format(data_genome_grch38_dir)
annotated_grch38_gff_dir = '{}/csvs'.format(data_genome_grch38_dir)
annotated_grch38_gtf = '{}/GCF_000001405.39_GRCh38.p13_genomic.gtf'.format(data_genome_grch38_dir)
annotated_grch38_gff_csv = '{}/grch38_gff.csv'.format(data_genome_grch38_dir)
annotated_grch38_gtf_csv = '{}/grch38_gtf.csv'.format(data_genome_grch38_dir)

polya_grch38_dir = '{}/poly-a/grch38'.format(data_dir)
polya_grch38_index_csv = '{}/polya_index.csv'.format(polya_grch38_dir)
polya_grch38_positive_csv = '{}/polya_positive.csv'.format(polya_grch38_dir)
polya_grch38_negative_csv = '{}/polya_negative.csv'.format(polya_grch38_dir)
polya_grch38_positive_kmer_csv = '{}/polya_positive_kmer.csv'.format(polya_grch38_dir)
polya_grch38_negative_kmer_csv = '{}/polya_negative_kmer.csv'.format(polya_grch38_dir)
polya_grch38_positive_dir = '{}/positive'.format(polya_grch38_dir)
polya_grch38_negative_dir = '{}/negative'.format(polya_grch38_dir)

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


ss_pos_acc_hs_fasta = "{}/positive_DNA_seqs_acceptor_hs.fa"
ss_pos_don_hs_fasta = "{}/positive_DNA_seqs_donor_hs.fa"
ss_neg_acc_hs_fasta = "{}/negative_DNA_seqs_acceptor_hs.fa"
ss_neg_don_hs_fasta = "{}/negative_DNA_seqs_donor_hs.fa"

ss_train_csv = "{}/train.csv".format(ss_dir)
ss_validation_csv = "{}/validation.csv".format(ss_dir)
ss_test_csv = "{}/test.csv".format(ss_dir)