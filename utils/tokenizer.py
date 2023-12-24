from pathlib import PureWindowsPath
import torch

class DNATokenizer:
    def __init__(self):
        self.voss_dict = {
            "A": torch.Tensor([1, 0, 0, 0]),
            "C": torch.Tensor([0, 1, 0, 0]),
            "G": torch.Tensor([0, 0, 1, 0]),
            "T": torch.Tensor([0, 0, 0, 1])
        }

    def voss_representation(self, dna: str):
        vector = torch.Tensor([self.voss_dict[n] for n in dna])
        return vector



def get_default_tokenizer():
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    from pathlib import Path, PureWindowsPath
    pretrained_path = str(Path(PureWindowsPath(pretrained_3kmer_dir)))
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    return tokenizer
