import sys, getopt
import os

def kmer(seq, length, window_size=1):
    return [seq[i:i+length] for i in range(0, len(seq)+1-length, window_size)]

def convert_sequence_to_kmer(seq, length, store_file=None, window_size=1):
    """
    Convert sequence `seq` into kmer representation and store the converted sequence in `store_file`.
    """
    arr = kmer(seq, length, window_size=window_size)
    str_arr = ' '.join(arr)
    if os.path.exists(store_file):
        os.remove(store_file)
    f = open(store_file, 'x')
    f.write("{}\n").format(str_arr)
    f.close()

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "s:l:w", ["sequence=","length=", "window_size="])
    sequence = ""
    length = 0
    window_size = 1
    fpath = "kmer.txt"
    for option, argument in opts:
        if option in ['-s', '--sequence']:
            sequence = argument
        elif option in ['-l', '--length']:
            length = argument
        elif option in ['-w', '--window_size']:
            window_size = argument
        else:
            raise getopt.GetoptError("Argument Error!")
    convert_sequence_to_kmer(sequence, length, window_size=window_size, store_file=fpath)