def expand_no_pandas(src_csv, target_csv, sliding_window_size=1, col_to_expand='sequence', length=512):
    import os
    import traceback
    import kmer
    """
    Expand sequence in csv file. CSV must have 'sequence', 'label_prom', 'label_ss', 'label_polya' column in this precise order.
    THIS IMPLEMENTATION DOES NOT USE PANDAS.
    Sequence is array of token (kmer) seperated by space.
    @param  src_csv (string): path to csv source file.
    @param  target_csv (string): path to csv target file.
    @param  col_to_expand (string): column whose content will be expanded.
    @param  sliding_window_size (int): default is 1. Set value for how much character is skipped for each slide.
    @param  length (int): default is 512. Length of each window.
    @return (boolean): True if sucess.
    """
    target_file = {}
    src_file = {}
    try:
        if not os.path.exists(src_csv):
            raise Exception("File source {} not found.".format(src_csv))
        if os.path.exists(target_csv):
            os.remove(target_csv)
        
        _f = open(src_csv, 'r')
        _len_src = len(_f.readlines())
        _f.close()
        src_file = open(src_csv, 'r')
        next(src_file)
        _count = 0
        target_file = open(target_csv, 'x')
        target_file.write("{}\n".format('sequence,label_prom,label_ss,label_polya'.strip()))
        for line in src_file:
            _count += 1
            if _count < _len_src:
                print("Expanding {} [{}/{}]".format(src_csv, _count, _len_src), end='\r') 
            else:
                print("Expanding {} [{}/{}]".format(src_csv, _count, _len_src)) 

            arr_line = line.strip().split(',')
            sequence = arr_line[0]
            label_prom = arr_line[1]
            label_ss = arr_line[2]
            label_polya = arr_line[3]
            sequence_to_kmer = sequence.split(' ')
            arr_sequence = kmer(sequence_to_kmer, length, window_size=sliding_window_size)
            for seq in arr_sequence:
                str_seq = ' '.join(seq)
                entry = '{},{},{},{}\n'.format(str_seq, label_prom, label_ss, label_polya)
                target_file.write(entry)

        src_file.close()
        target_file.close()
        return True
    except Exception as e:
        print("Error {}".format(e))
        print("Error {}".format(traceback.format_exc()))
        if src_file != {}:
            src_file.close()
        if target_file != {}:
            target_file.close()
        return False