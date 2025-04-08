import os
import sys
from getopt import getopt
from .utils.classification import extract_acceptor_donor_exon_intron


def parse_args(args):
    options, argument = getopt(args, "i:g:d:", ["index=", "gene-dir=", "dest-dir="])
    output = {}
    for o, a in options:
        if o in ["-i", "--index"]:
            output["index"] = a
        elif o in ["-g", "--gene-dir"]:
            output["gene-dir"] = a
        elif o in ["-d", "--dest-dir"]:
            output["dest-dir"] = a
        else:
            raise ValueError(f"Option {o} not recognized.")
    return output

if __name__ == "__main__":
    output = parse_args(sys.argv[1:])
    index = output.get("index")
    gene_dir = output.get("gene-dir")
    dest_dir = output.get("dest-dir")

    extract_acceptor_donor_exon_intron(
        index,
        gene_dir,
        dest_dir
    )
