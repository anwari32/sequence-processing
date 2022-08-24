import os
import pandas as pd
import sys
from getopt import getopt

def parse_args(argv):
    opts, arguments = getopt(argv, "s:d:", ["source=", "destination-dir="])
    outputs = {}
    for o, a in opts:
        if o in ["-s", "--source"]:
            outputs["source"] = a
        elif o in ["-d", "--destination-dir"]:
            outputs["destination-dir"] = a
        else:
            raise ValueError(f"Argument {o} not recognized.")

    return outputs

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    fsource = args.get("source", False)

    df = pd.read_csv(fsource)
    train_df = df.sample(frac=0.8, random_state=1337)
    validation_df = df.drop(train_df.index)
    test_df = validation_df.sample(frac=0.5, random_state=1337)
    validation_df = validation_df.drop(test_df.index)

    fname = fsource.split('.')[:-1]
    fname = '.'.join(fname)
    dirname = args.get("destination-dir", False)
    train_path = os.path.join(dirname, f"{fname}_train.csv")
    validation_path = os.path.join(dirname, f"{fname}_validation.csv")
    test_path = os.path.join(dirname, f"{fname}_test.csv")

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)
