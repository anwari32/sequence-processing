# Create fine tuning file from fasta file.
from Bio import SeqIO
import os
from utils import create_k_mer

# Generate file for fine tuning using FASTA file.
# @param fasta_file : Original fasta file.
# @param label_for_this_file : What label for this fine tuning file.
# @param output_file_path : What and where the fine tuning is named and stored. 
#                           If file path exists, existing file will be removed.
# @param n_samples : How many sequence will be put in fine tuning file. 
#                    If all sequence is to be generated, please put -1.
# @param k_mer : Size of k-mer. If k-mer is not required, please put -1.
# @param n_k_mer : How many kmers are written to file for each sequence in fasta file. 
#                  If all kmers are written, please put -1.
def generate_sample_fine_tuning_file(fasta_file, label_for_this_file, output_file_path, n_samples, k_mer, n_k_mer):
    records = list(SeqIO.parse(fasta_file, 'fasta'))
    if len(records) >= n_samples:
        records = records[0:n_samples]
    
    if (os.path.exists(output_file_path)):
        os.remove(output_file_path)
        
    output_file = open(output_file_path, 'w+')
    for r in records:
        output_file.write(create_k_mer(str(r.seq), k_mer, n_k_mer) + '\t' + str(label_for_this_file) + '\n')
    output_file.close()
    return output_file_path

# Merge two files together.
# @param fp : First file path.
# @param gp : Second file path.
# @param hp : Third file as result from merging two files together.
def merge_file(fp, gp, hp):
    data1 = data2 = ""
    with open(fp) as f:
        data1 = f.read()
    with open(gp) as g:
        data2 = g.read()
    
    final_data = data1 + data2      
    with open (hp, 'w') as h:
        h.write(final_data)
        h.close()
        
# Merge files into single file.
# @param origin_files : Original files in list. Each file has header in first line.
# @param merged_file : Merged file.
# @param headers : Header for this file in list. Each header is separated by tabs.
def merge_files(origin_files, merged_file_path, headers):
    merged_data = ""
    
    for file_path in origin_files:
        print('reading file {}'.format(file_path))
        of = open(file_path, 'r')
        next(of) # Skip the header.
        for line in of:
            d = of.read()
            merged_data += d
            
    merged_file = open(merged_file_path, 'w+')
    print('writing to file {}'.format(merge_file_path))
    if (headers):
        header = headers[0]
        for h in headers[1:]:
            header +='\t' + h
        merged_file.write(header + '\n')
    
    merged_file.write(merged_data)
    merged_file.close()
