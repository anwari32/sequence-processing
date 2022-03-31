def get_default_tokenizer():
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    tokenizer = BertTokenizer.from_pretrained(pretrained_3kmer_dir)
    return tokenizer