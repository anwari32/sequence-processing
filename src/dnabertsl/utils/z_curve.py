"""
This notebook contains Z-curve representation of DNA sequence.
The idea is to create numeric represenation of base (A, T, G, and C) in 3D space with this formula.

xn = (An + Gn) - (Cn + Tn) as purine vs. pyrimidine,
yn = (An + Cn) - (Gn + Tn) as amino vs. keto,
zn = (An + Tn) - (Gn + Cn) as hydrogen bonds representation.

Reference can be found at https://pubmed.ncbi.nlm.nih.gov/8204213/
"""

DATASET_PATH = "data/ft/fine_tuning_sample_k-mer_3_ALPHA_BETA_DELTA_merged.txt"

def get_z_curve(seq):
    """
    Generate array of (x, y, z) coordinate from a sequence.
    @param seq : Sequence to be coverted.
    @return : Array of coordinates.
    """
    nucleotides = {
        'A': 0,
        'T': 0,
        'G': 0,
        'C': 0
    }
    z_coordinates = []
    for c in seq:
        nucleotides[c] += 1
        xn = (nucleotides['A'] + nucleotides['G']) - (nucleotides['C'] + nucleotides['T'])
        yn = (nucleotides['A'] + nucleotides['C']) - (nucleotides['G'] + nucleotides['T'])
        zn = (nucleotides['A'] + nucleotides['T']) - (nucleotides['G'] + nucleotides['C'])
        z_coordinates.append(
            (xn, yn, zn)
        )
    return z_coordinates

# Read file containing sequences and its labels.
# Convert each sequence into z-curve representation.
# @param dataset_file_path : Path to source file.
# @param target_file_path : Target file to write z-curve representation of sequences in dataset. 
#                           If empty then no file is created.
def generate_z_curve_from_file(dataset_file_path, target_file_path=False):
    f = open(dataset_file_path, 'r')
    
    next(f) # The file has header so the reader needs to skip the first line. Data starts at second line.
    z_curves = []
    
    # Iterate each line.
    for line in f:
        arr = line.split('\t')
        seq = arr[0].strip()
        label = arr[1].strip()
        z_curves.append(get_z_curve(seq))
    
if __name__ == "__main__":
    # Test
    s = "GTTCTCTAAACGAACTTTAAAATCTGTGTGGCTGTCACTCGGCTGCATGCTTAGTGCACT"
    print(get_z_curve(s))