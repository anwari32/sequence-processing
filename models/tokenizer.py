
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

