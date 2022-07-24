# Filter data bundle into sets of sequence with splice site and non splice site.
# Non splice site sequence is split into exon and intron sequence.

import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":

    path = os.path.join("workspace", "seqlab", "seqlab.strand-positive.kmer.stride-510")
    bundle_path = os.path.join(path, "bundle.csv")
    splice_site_bundle_path = os.path.join(path, "splice_site_bundle.csv")
    intron_bundle_path = os.path.join(path, "intron_bundle.csv")
    exon_bundle_path = os.path.join(path, "exon_bundle.csv")

    for a in [splice_site_bundle_path, intron_bundle_path, exon_bundle_path]:
        if os.path.exists(a):
            os.remove(a)

    def at_least_one_exists(list, target_list):
        # Check if at leats one element of list exists in target_list.
        found = False
        for elem in list:
            if elem in target_list:
                found = True
        return found

    splice_sites = ['iiE', 'iEi', 'Eii', 'iEE', 'EEi', 'EiE']

    bundle_df = pd.read_csv(bundle_path)
    splice_site_bundle = open(splice_site_bundle_path, "x")
    intron_bundle = open(intron_bundle_path, "x")
    exon_bundle = open(exon_bundle_path, "x")

    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.write("sequence,label\n")

    for i, r in tqdm(bundle_df.iterrows(), total=bundle_df.shape[0], desc="Processing :"):
        arr_labels = r["label"].split(" ")
        if at_least_one_exists(splice_sites, arr_labels):
            # raise NotImplementedError("TODO: write sequence into splice site bundle.")
            splice_site_bundle.write(f"{r['sequence']},{r['label']}\n")
        elif all([a == "iii" for a in arr_labels]):
        #raise NotImplementedError("TODO: write sequence into intron bundle.")
            intron_bundle.write(f"{r['sequence']},{r['label']}\n")
        elif all([a == "EEE" for a in arr_labels]):
            #raise NotImplementedError("TODO: write sequence into exon bundle.")
            exon_bundle.write(f"{r['sequence']},{r['label']}\n")
        
    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.close()

    # Split data into train, validation, and test data.
    splice_site_train_bundle_path = os.path.join(path, "splice_site_train_bundle.csv")
    splice_site_validation_bundle_path = os.path.join(path, "splice_site_validation_bundle.csv")
    splice_site_test_bundle_path = os.path.join(path, "splice_site_test_bundle.csv")
    splice_site_df = pd.read_csv(splice_site_bundle_path)

    splice_site_train_df = splice_site_df.sample(frac=0.8)
    splice_site_val_df = splice_site_df.drop(splice_site_train_df.index)
    splice_site_test_df = splice_site_val_df.sample(frac=0.5)
    splice_site_val_df = splice_site_val_df.drop(splice_site_test_df.index)

    splice_site_train_df.to_csv(splice_site_train_bundle_path, index=False)
    splice_site_val_df.to_csv(splice_site_validation_bundle_path, index=False)
    splice_site_test_df.to_csv(splice_site_test_bundle_path, index=False)