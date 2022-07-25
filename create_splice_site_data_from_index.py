# Filter data bundle into sets of sequence with splice site and non splice site.
# Non splice site sequence is split into exon and intron sequence.

import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":

    def at_least_one_exists(list, target_list):
    # Check if at leats one element of list exists in target_list.
    found = False
    for elem in list:
        if elem in target_list:
            found = True
    return found

splice_sites = ['iiE', 'iEi', 'Eii', 'iEE', 'EEi', 'EiE']

gene_bundle_dirpath = os.path.join("workspace", "seqlab", "seqlab.strand-positive.kmer.stride-510.from-index")
gene_train_bundle_path = os.path.join(gene_bundle_dirpath, "gene_train_bundle.csv")
gene_validation_bundle_path = os.path.join(gene_bundle_dirpath, "gene_validation_bundle.csv")
gene_test_bundle_path = os.path.join(gene_bundle_dirpath, "gene_test_bundle.csv")

for p in [gene_train_bundle_path, gene_validation_bundle_path, gene_test_bundle_path]:
    df = pd.read_csv(p)
    basename = os.path.basename(p).split(".")[0]
    splice_site_target = os.path.join(gene_bundle_dirpath, f"{basename}_splice_site.csv")
    intron_target = os.path.join(gene_bundle_dirpath, f"{basename}_intron.csv")
    exon_target = os.path.join(gene_bundle_dirpath, f"{basename}_exon.csv")

    for a in [splice_site_target, intron_target, exon_target]:
        if os.path.exists(a):
            os.remove(a)

    splice_site_bundle = open(splice_site_target, "x")
    intron_bundle = open(intron_target, "x")
    exon_bundle = open(exon_target, "x")

    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.write("sequence,label\n")

    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Processing :"):
        arr_labels = r["label"].split(" ")
        if at_least_one_exists(splice_sites, arr_labels):
            # raise NotImplementedError("TODO: write sequence into splice site bundle.")
            # Extract splice sites from gene_train_bundle.csv, gene_validation_bundle.csv, and gene_test_bundle.csv.
            splice_site_bundle.write(f"{r['sequence']},{r['label']}\n")
        elif all([a == "iii" for a in arr_labels]):
            # raise NotImplementedError("TODO: write sequence into intron bundle.")
            # Extract introns.
            intron_bundle.write(f"{r['sequence']},{r['label']}\n")
        elif all([a == "EEE" for a in arr_labels]):
            # raise NotImplementedError("TODO: write sequence into exon bundle.")
            # Extract exons.
            exon_bundle.write(f"{r['sequence']},{r['label']}\n")

    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.close()
