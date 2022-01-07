# Responsible to provide interface to data directory.
data_dir = './data'
deepromoter_dir = data_dir + '/deepromoter'
promoter_dir = data_dir + '/promoter'
polya_dir = data_dir + '/poly-a'
splicesites_dir = data_dir + '/splice-sites'
genome_dir = data_dir + '/genome'
genome_grch37_dir = genome_dir + '/grch37'
genome_grch37 = genome_grch37_dir + '/GRCh37_latest_genomic.gff'
genome_grch37_exon_dir = genome_grch37_dir + '/exon'
genome_grch37_exon_latest_dir = genome_grch37_exon_dir + '/latest'

genome_grch38_dir = genome_dir + '/grch38'
genome_grch38 = genome_grch38_dir + '/GRCh38_latest_genomic.gff'
genome_grch38_exon_dir = genome_grch38_dir + '/exon'
genome_grch38_exon_latest_dir = genome_grch38_exon_dir + '/latest'
chr1 = genome_grch38_exon_latest_dir + '/NC000001.11.csv'

# Sample directory.
sample_dir = './sample'
grch37_sample_dir = sample_dir + '/grch37'
grch38_sample_dir = sample_dir + '/grch38'

# Homo sapiens-specific data.
hs_dir = data_dir + '/homo-sapiens'
hs_nc1 = hs_dir + '/NC_000001.11.fasta'
hs_nc2 = hs_dir + '/NC_000002.12.fasta'