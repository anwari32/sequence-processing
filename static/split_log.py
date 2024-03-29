"""
This scripts are written to split logs.
"""
import os
import sys
import pandas as pd
from getopt import getopt
from tqdm import tqdm

def parse_args(argv):
    opts, _ = getopt(argv, "s:d:", ["source-log=", "destination-dir="])
    outputs = {}
    for o, a in opts:
        if o in ["-s", "--source-log"]:
            outputs["source-log"] = a 
        elif o in ["-d", "--destination-dir"]:
            outputs["destination-dir"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")

    return outputs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    source_log = args.get("source-log", False)
    dest_dir = args.get("destination-dir", os.path.dirname(source_log))

    print(f"Splitting {source_log}")
    print(f"Store splitted log at {dest_dir}")

    df = pd.read_csv(source_log)
    epochs = df["epoch"].unique()
    fname = os.path.basename(source_log).split('.')[:-1]
    fname = '.'.join(fname)
    for epoch in tqdm(epochs, total=len(epochs), desc="Processing log"):
        ndf = df[df["epoch"] == epoch]
        ndf.to_csv(os.path.join(dest_dir, f"{fname}.{epoch}.csv"), index=False)
