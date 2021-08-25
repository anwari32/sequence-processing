# Generate TATA positive-dataset.
human_tata_ccaat = "data/homo-sapiens/human_TATA_CCAAT_hg38.fa"
human_tata_nonccaat = "data/homo-sapiens/human_TATA_nonCCAAT_hg38.fa"
human_tata = "data/homo-sapiens/human_TATA_hg38.fa"

human_nontata_ccaat = "data/homo-sapiens/human_nonTATA_CCAAT_hg38.fa"
human_nontata_nonccaat = "data/homo-sapiens/human_nonTATA_nonCCAAT_hg38.fa"
human_nontata = "data/homo-sapiens/human_nonTATA_hg38.fa"

positive_tata = [human_tata_ccaat, human_tata_nonccaat, human_tata]
negative_tata = [human_nontata_ccaat, human_nontata_nonccaat, human_nontata]

positive_label = "1"
negative_label = "0"

# This data is generated for DeePromoter so generate 300bp only for each sequence.
# Data is in FASTA format so use SeqIO from Bio
positive_dataset_path = "deepromoter/human_tata_positive.txt"
negative_dataset_path = "deepromoter/human_tata_negative.txt"

from Bio import SeqIO

# Begin process positive tata.
file = open(positive_dataset_path, 'w+')
for p in positive_tata:
    records = list(SeqIO.parse(p, "fasta"))
    records = records[0:100] # Get only first 100 records.
    for r in records:
        sequence = str(r.seq)[0:300] # Retrieve only first 300 bp.
        file.write(sequence + '\n')
        
file.close()

# Begin process negative tata.
file = open(negative_dataset_path, 'w+')
for p in negative_tata:
    records = list(SeqIO.parse(p, 'fasta'))
    records = records[0:100]
    for r in records:
        sequence = str(r.seq)[0:300] # Retrieve only first 300 bp.
        file.write(sequence + '\n')

file.close()
