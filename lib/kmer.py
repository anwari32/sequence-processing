def kmer(seq, length, window_size=1):
    return [seq[i:i+length] for i in range(0, len(seq)+1-length, window_size)]

if __name__ == "main":
    raise NotImplementedError("Not Implemented")