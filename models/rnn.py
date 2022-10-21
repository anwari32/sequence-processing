r"""
RNN model implementation based on Untary et. al., 2022.
"""

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


class RNN_Config:
    def __init__(self, dicts={}):
        self.name = dicts.get("name", "unnamed")
        self.num_embeddings = dict.get("num_embeddings")
        self.embedding_dim = dict.get("embedding_size")
        self.rnn = dicts.get("rnn", "unknown")
        self.input_size = dicts.get("input_size", 1)
        self.hidden_size = dicts.get("hidden_size", 1)
        self.num_labels = dicts.get("num_labels", 8)
        self.num_layers = dicts.get("num_layers", 1)
        self.dropout_prob = dicts.get("dropout_prob", 0.2)
        self.bidirectional = dicts.get("bidirectional", False)

    def export(self, export_path: str):
        # Write instance attributes into single json file.
        raise NotImplementedError()

class RNN_Model(torch.nn.Module):
    def __init__(self, config: RNN_Config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            self.config.num_embeddings,
            self.config.embedding_dim
        )
        self.rnn = None
        self.dropout = torch.nn.Dropout(self.config.dropout_prob)
        self.classifier = torch.nn.Linear(
            (2 if self.config.bidirectional else 1) * self.config.hidden_size,
            self.config.num_labels
        )


default_bilstm_dict = {
    "name": "default_bilstm",
    "rnn": "bilstm",
    "input_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2,
    "bidirectional": True
}

default_bigru_dict = {
    "name": "default_bigru",
    "rnn": "bigru",
    "input_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2,
    "bidirectional": True
}

default_bilstm_config = RNN_Config(default_bilstm_dict)
default_bigru_config = RNN_Config(default_bigru_dict)

class RNN_BiLSTM(RNN_Model):
    def __init__(self, config=default_bilstm_config):
        super().__init__(config)
        self.rnn = torch.nn.LSTM(
            self.config.input_size,
            self.config.hidden_size, 
            self.config.num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input, hidden_units=None):
        if hidden_units:
            hn = hidden_units[0]
            hn = hn[:, 0:input.shape[0], :].contiguous()
            cn = hidden_units[1]
            cn = cn[:, 0:input.shape[0], :].contiguous()
            hidden_units = (hn, cn)

        output, hidden_output = self.rnn(input, hidden_units)
        output = self.dropout(output)
        output = self.classifier(output)
        return output, hidden_output

class RNN_BiGRU(RNN_Model):
    def __init__(self, config=default_bigru_config):
        super().__init__(config)
        self.rnn = torch.nn.GRU(
            self.config.input_size,
            self.config.hidden_size,
            self.config.num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input, hidden_unit=None):
        output = self.embedding(input)
        if hidden_unit:
            cn = hidden_unit[:, 0:output.shape[0], :]
            hidden_unit = cn

        output, hidden_output = self.rnn(input, hidden_unit)
        output = self.dropout(output)
        output = self.classifier(output)

        return output, hidden_output

if __name__ == "__main__":
    # Evaluate model to see if model is properly initialized.
    bilstm_model = RNN_BiLSTM()
    bigru_model = RNN_BiGRU()

    print(bilstm_model)
    print(bigru_model)

    input_sample = torch.randint(0, 69, (5, 512, 1), dtype=torch.float)
    bilstm_output, bilstm_hidden_output = bilstm_model(input_sample)
    bigru_output, bigru_hidden_output  = bigru_model(input_sample)

    assert input_sample.shape == (5, 512, 1), "Input sample shape not match."
    assert bilstm_output.shape == (5, 512, 8), "Desired dimension not match."
    assert bigru_output.shape == (5, 512, 8), "Desired dimension not match."
