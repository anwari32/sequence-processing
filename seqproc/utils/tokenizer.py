from pathlib import PureWindowsPath
import torch


def get_DNABERTTokenizer():
    from transformers import BertTokenizer
    from data_dir import pretrained_3kmer_dir
    from pathlib import Path, PureWindowsPath
    pretrained_path = str(Path(PureWindowsPath(pretrained_3kmer_dir)))
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    return tokenizer


def voss_tokenize(dna_str):
    assert(all([p in ["A", "C", "G", "T"]] for p in dna_str))
    voss_dict = {
        "A": torch.Tensor([1, 0, 0, 0]),
        "C": torch.Tensor([0, 1, 0, 0]),
        "G": torch.Tensor([0, 0, 1, 0]),
        "T": torch.Tensor([0, 0, 0, 1])
    }
    return [voss_dict[p] for p in dna_str]


if __name__ == "__main__":
    dna_str = "ACGTACGT"
    dna_rep = voss_tokenize(dna_str)
    assert(all(torch.eq(dna_rep[0], torch.Tensor([1, 0, 0, 0]))))
    assert(all(torch.eq(dna_rep[1], torch.Tensor([0, 1, 0, 0]))))
    assert(all(torch.eq(dna_rep[2], torch.Tensor([0, 0, 1, 0]))))
    assert(all(torch.eq(dna_rep[3], torch.Tensor([0, 0, 0, 1]))))
    print("Voss sample test passed.")

