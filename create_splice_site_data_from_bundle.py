# Filter data bundle into sets of sequence with splice site and non splice site.
# Non splice site sequence is split into exon and intron sequence.

import pandas as pd
import os
from tqdm import tqdm
import sys
from getopt import getopt

def parse_args(argv):
    opts, arguments = getopt(argv, "s:d:", ["source-bundle=", "destination-bundle-dir="])
    outputs = {}
    for o, a in opts:
        if o in ["-s", "--source-bundle"]:
            outputs["source-bundle"] = a
        elif o in ["-d", "--destination-bundle-dir"]:
            outputs["destination-bundle-dir"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")
    return outputs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    # path = os.path.join("workspace", "seqlab", "seqlab.strand-positive.kmer.stride-510")
    # bundle_path = os.path.join(path, "bundle.csv")
    bundle_path = args.get("source-bundle", False)
    destination_bundle_dir_path = args.get("destination-bundle-dir", False)
    splice_site_bundle_path = os.path.join(destination_bundle_dir_path, "splice_site_bundle.csv")
    intron_bundle_path = os.path.join(destination_bundle_dir_path, "intron_bundle.csv")
    exon_bundle_path = os.path.join(destination_bundle_dir_path, "exon_bundle.csv")

    for a in [splice_site_bundle_path, intron_bundle_path, exon_bundle_path]:
        adirname = os.path.dirname(a)
        if not os.path.exists(adirname):
            os.makedirs(adirname, exist_ok=True)
        if os.path.exists(a):
            os.remove(a)

    splice_sites = ['iiE', 'iEi', 'Eii', 'iEE', 'EEi', 'EiE']

    bundle_df = pd.read_csv(bundle_path)
    splice_site_bundle = open(splice_site_bundle_path, "x")
    intron_bundle = open(intron_bundle_path, "x")
    exon_bundle = open(exon_bundle_path, "x")

    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.write("sequence,label\n")

    for i, r in tqdm(bundle_df.iterrows(), total=bundle_df.shape[0], desc="Processing :"):
        arr_labels = r["label"].split(" ")
        if all([a == "iii" for a in arr_labels]):
            intron_bundle.write(f"{r['sequence']},{r['label']}\n")
        elif all([a == "EEE" for a in arr_labels]):
            exon_bundle.write(f"{r['sequence']},{r['label']}\n")
        else:
            splice_site_bundle.write(f"{r['sequence']},{r['label']}\n")
        
    for a in [splice_site_bundle, intron_bundle, exon_bundle]:
        a.close()

    # Split data into train, validation, and test data.
    splice_site_train_bundle_path = os.path.join(destination_bundle_dir_path, "splice_site_train_bundle.csv")
    splice_site_validation_bundle_path = os.path.join(destination_bundle_dir_path, "splice_site_validation_bundle.csv")
    splice_site_test_bundle_path = os.path.join(destination_bundle_dir_path, "splice_site_test_bundle.csv")
    splice_site_df = pd.read_csv(splice_site_bundle_path)

    splice_site_train_df = splice_site_df.sample(frac=0.8)
    splice_site_val_df = splice_site_df.drop(splice_site_train_df.index)
    splice_site_test_df = splice_site_val_df.sample(frac=0.5)
    splice_site_val_df = splice_site_val_df.drop(splice_site_test_df.index)

    splice_site_train_df.to_csv(splice_site_train_bundle_path, index=False)
    splice_site_val_df.to_csv(splice_site_validation_bundle_path, index=False)
    splice_site_test_df.to_csv(splice_site_test_bundle_path, index=False)


