from transformers import BertTokenizer
from data_dir import pretrained_3kmer_dir

def init_dnabert_tokenizer(path=pretrained_3kmer_dir):
    return BertTokenizer.from_pretrained(path)