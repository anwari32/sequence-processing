import os
import pandas as pd
from data_preparation import str_kmer
_columns = ["sequence", "label_prom", "label_ss", "label_polya"]
src_paths = [
    os.path.join("workspace", "mtl", "original", p) for p in [
        'train.balanced.csv',
        'train.csv',
        'train.sample.csv',
        'validation.balanced.csv',
        'validation.csv',
        'validation.sample.csv'
    ]
]

for src_path in src_paths:
    print(f"Working in {src_path}")
    src_df = pd.read_csv(src_path)
    # target_df = pd.DataFrame(columns=_columns)
    dest_path = os.path.join("workspace", "mtl", os.path.basename(src_path))
    if os.path.exists(dest_path):
        os.remove(dest_path)
    dest = open(dest_path, "x")
    dest.write(f"{','.join(_columns)}\n")
    for i, r in src_df.iterrows():
        seq_kmer = str_kmer(r["sequence"], 3)
        #target_df = pd.concat([target_df, pd.DataFrame([
        #    [seq_kmer, r["label_prom"], r["label_ss"], r["label_polya"]]
        #], columns=_columns)])
        dest.write(f"{seq_kmer},{r['label_prom']},{r['label_ss']},{r['label_polya']}\n")
    #target_df.to_csv(os.path.join("workspace", "mtl", os.path.basename(src_path)), index=False)
    dest.close()

