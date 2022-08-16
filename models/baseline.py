import torch

class Baseline(torch.nn.Module):
    """
    Baseline architecture for sequential labelling.
    """
    def __init__(self, config=None):
        super().__init__()
        self.num_labels = config.get("num_labels", 8) if config else 8
        self.num_layers = config.get("num_layers", 1) if config else 1
        self.hidden_size = config.get("hidden_size", 512) if config else 512
        self.input_layer = torch.nn.Linear(1, self.hidden_size)
        self.hidden_layer = torch.nn.Sequential()
        for i in range(self.num_layers):
            self.hidden_layer.add_module(
                f"linear-{i}", torch.nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.hidden_layer.add_module(
                f"relu-{i}", torch.nn.ReLU()
            )
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.Softmax(dim=2)

    def forward(self, input):
        output = self.input_layer(input)
        output = self.hidden_layer(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output


if __name__ == "__main__":
    from transformers import BertTokenizer
    import os

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join("pretrained", "3-new-12w-0")
    )
    Label_Dictionary = {
        '[CLS]': -100, #_create_one_hot_encoding(0, 10),
        '[SEP]': -100,
        # '[PAD]': 2, #_create_one_hot_encoding(9, 10)
        'III': -100,   # Created III instead of iii for PAD special token. 
                    # This is for enabling reading contigs if token is predicted as padding. 
                    # #_create_one_hot_encoding(9, 10)
        'iii': 0,   #_create_one_hot_encoding(1, 10),
        'iiE': 1,   #_create_one_hot_encoding(2, 10),
        'iEi': 2,   #_create_one_hot_encoding(3, 10),
        'Eii': 3,   #_create_one_hot_encoding(4, 10),
        'iEE': 4,   #_create_one_hot_encoding(5, 10),
        'EEi': 5,   #_create_one_hot_encoding(6, 10),
        'EiE': 6,   #_create_one_hot_encoding(7, 10),
        'EEE': 7,  #_create_one_hot_encoding(8, 10),
    }
    sample_input = "CTG TGA GAG AGA GAT ATC TCG CGC GCG CGC GCC CCA CAG AGT GTG TGC GCA CAC ACT CTC TCC CCA CAG AGC GCC CCT CTA TAG AGG GGT GTG TGA GAC ACA CAG AGA GAG AGT GTG TGA GAG AGA GAC ACT CTC TCT CTG TGT GTC TCT CTC TCA CAA AAA AAA AAA AAC ACA CAT ATA TAT ATA TAT ATA TAA AAT ATA TAA AAT ATA TAA AAT ATA TAA AAT ATG TGC GCT CTA TAT ATA TAT ATC TCT CTA TAA AAG AGG GGT GTT TTG TGT GTG TGT GTA TAT ATC TCA CAT ATT TTA TAC ACT CTG TGA GAA AAA AAG AGG GGT GTT TTA TAA AAT ATT TTT TTT TTG TGC GCT CTG TGT GTG TGT GTT TTT TTT TTT TTC TCT CTT TTA TAT ATA TAC ACT CTA TAT ATT TTT TTT TTA TAC ACT CTG TGC GCT CTT TTT TTT TTT TTA TAA AAA AAC ACT CTT TTT TTT TTA TAT ATC TCT CTT TTT TTT TTA TAA AAA AAA AAA AAT ATT TTA TAT ATA TAG AGA GAA AAA AAT ATA TAA AAA AAA AAT ATA TAT ATA TAT ATG TGT GTA TAT ATG TGA GAT ATA TAG AGG GGA GAA AAA AAA AAT ATT TTA TAG AGA GAA AAA AAA AAT ATT TTT TTC TCA CAG AGA GAT ATA TAA AAG AGT GTA TAT ATG TGA GAA AAG AGA GAA AAG AGA GAA AAA AAA AAT ATA TAA AAA AAA AAA AAT ATC TCA CAC ACT CTT TTG TGT GTA TAA AAT ATC TCT CTT TTT TTA TAC ACT CTG TGA GAT ATA TAT ATT TTT TTT TTA TAA AAA AAA AAT ATA TAA AAT ATT TTT TTA TAG AGT GTT TTA TAT ATG TGT GTA TAT ATT TTC TCT CTT TTT TTT TTA TAG AGA GAC ACT CTT TTT TTT TTT TTA TAA AAT ATG TGT GTA TAT ATA TAT ATA TAT ATA TAC ACC CCT CTA TAC ACT CTT TTT TTT TTT TTT TTT TTC TCT CTT TTA TAA AAG AGA GAG AGT GTT TTT TTA TAT ATA TAA AAT ATA TAG AGT GTA TAA AAT ATA TAC ACT CTT TTA TAT ATC TCC CCT CTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTT TTG TGC GCT CTC TCA CAC ACA CAT ATA TAG AGG GGA GAA AAA AAA AAG AGA GAC ACA CAG AGA GAG AGT GTC TCT CTC TCT CTA TAA AAT ATA TAC ACA CAG AGC GCA CAA AAT ATA TAT ATG TGA GAA AAG AGC GCC CCA CAT ATT TTA TAG AGC GCC CCT CTA TAC ACT CTC TCA CAA AAT ATT TTC TCA CAG AGA GAG AGC GCG CGT GTT TTA TAT ATG TGC GCC CCC CCG CGC GCC CCT CTA TAG AGA GAG AGC GCG CGG GGG GGC GCC CCC CCA CAG AGG GGT GTC TCT CTT TTA TAG AGT GTA TAA AAA AAC ACC CCA CAG AGT GTT TTT TTT TTG TGG GGG GGA GAA AAA AAC ACT CTT TTA TAT ATG TGA GAA AAG AGA GAG AGC GCT CTC,iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iii iiE iEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE EEE"
    sample_input = sample_input.split(',')
    sample_sequence = sample_input[0]
    sample_label = sample_input[1]
    sample_label = [Label_Dictionary[k] for k in sample_label.split(' ')]
    len_label = len(sample_label)
    if  len(sample_label) < 512:
        for i in range(512 - len(sample_label)):
            sample_label.append(0)


    encoded = tokenizer.encode_plus(text=sample_sequence, return_attention_mask=True, return_token_type_ids=True, padding="max_length")
    input_ids = torch.tensor(encoded.get('input_ids'))
    input_ids = input_ids.reshape(1, 512, 1).float()
    attention_mask = torch.tensor(encoded.get('attention_mask'))
    token_type_ids = torch.tensor(encoded.get('token_type_ids'))
    target_label = torch.tensor(sample_label)
    target_label = target_label.reshape(1, 512)

    criterion = torch.nn.CrossEntropyLoss()

    model = Baseline()
    output = model(input_ids)
    loss = criterion(output.view(-1, 8), target_label.view(-1))
    print(output, output.shape)
    print(loss)
