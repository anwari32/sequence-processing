import os, sys, getopt
import pandas as pd
from tqdm import tqdm
from kmer import kmer
"""
This script defines `merge and expand` procedure.
Merge means merging multiple csv files stated in `inputs` and write the merged file into `output`.
Expansion means that each sequence in csv is expanded into a set of substring with certain `length`.
CSV file has column 'sequence' and 'label' and each sequence is already in kmer form separated by spaces.
If `length` <= 0 then there is no expansion.
"""
if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'i:o:l:s', ['input=', 'output=', 'length=', 'stride='])
    input_files = []
    output_file = ""
    length = 0
    stride = 0
    for option, argument in opts:
        if option in ["-i", "--input"]:
            input_files = argument.strip().split(',')
        elif option in ["-o", "--output"]:
            output_file = argument
        elif option in ["-l", "--length"]:
            length = int(argument)
        elif option in ["-s", "--stride"]:
            stride = int(argument)
        else:
            print('Argument Error!')
            sys.exit(2)

    _columns = ['sequence', 'label']
    temp_df = pd.DataFrame(columns=_columns)
    fin_df = pd.DataFrame(columns=_columns)
    for src in input_files:
        temp_df = pd.concat([temp_df, pd.read_csv(src)])
    
    if length > 0:
        for i, r in tqdm(temp_df.iterrows()):
            _label = r['label']
            _sequence = r['sequence'].strip().split(' ')
            _arr_sequence = kmer(_sequence, length, window_size=stride)
            for _arr_kmer in _arr_sequence:
                _str_arr_kmer = ' '.join(_arr_kmer)
                _frame = pd.DataFrame([[_str_arr_kmer, _label]], columns=_columns)
                fin_df = pd.concat([fin_df, _frame])
            #endfor
        #endfor
        if os.path.exists(output_file):
            os.remove(output_file)
        fin_df.to_csv(output_file, index=False)



