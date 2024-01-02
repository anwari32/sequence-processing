from transformers import BertForTokenClassification, BertTokenizer
from seqproc.utils import utils

if __name__ == "__main__":
    print(utils.kmer)
    # pretrained_path = "../models/dnabert/dnabert-6"
    # model = BertForTokenClassification.from_pretrained(pretrained_path)
    # tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    # dna_str = "ACTCGTAGCATGCATGATGCATGCATCGATGCA"
    # dna_tokens = kmer(dna_str)

    # print(model)
    # print(tokenizer)
    # print(dna_str)
    # print(dna_tokens)