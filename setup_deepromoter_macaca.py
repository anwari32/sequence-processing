# Generate TATA positive-dataset.
macaca_tata_ccaat = "data/macaca/macaca_TATA_CCAAT_rheMac8.fa"
macaca_tata_nonccaat = "data/macaca/macaca_TATA_nonCCAAT_rheMac8.fa"
macaca_tata = "data/macaca/macaca_TATA_rheMac8.fa"

macaca_nontata_ccaat = "data/macaca/macaca_nonTATA_CCAAT_rheMac8.fa"
macaca_nontata_nonccaat = "data/macaca/macaca_nonTATA_nonCCAAT_rheMac8.fa"
macaca_nontata = "data/macaca/macaca_nonTATA_rheMac8.fa"

positive_tata = [macaca_tata_ccaat, macaca_tata_nonccaat, macaca_tata]
negative_tata = [macaca_nontata_ccaat, macaca_nontata_nonccaat, macaca_nontata]

positive_label = "1"
negative_label = "0"

# This data is generated for DeePromoter so generate 300bp only for each sequence.
# Data is in FASTA format so use SeqIO from Bio
positive_dataset_path = "deepromoter/macaca_tata_positive.txt"
negative_dataset_path = "deepromoter/macaca_tata_negative.txt"

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
