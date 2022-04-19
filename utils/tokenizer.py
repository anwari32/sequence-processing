from pathlib import PureWindowsPath


def get_default_tokenizer():
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    from pathlib import Path, PureWindowsPath
    pretrained_path = str(Path(PureWindowsPath(pretrained_3kmer_dir)))
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    return tokenizer