r"""
RNN model implementation based on Untary et. al., 2022.
"""

from unicodedata import bidirectional
import torch

class RNN_Config:
    def __init__(self, dicts={}):
        self.input_size = dicts.get("input_size", 1)
        self.hidden_size = dicts.get("hidden_size", 1)
        self.num_labels = dicts.get("num_labels", 8)
        self.num_layers = dicts.get("num_layers", 1)
        self.dropout_prob = dicts.get("dropout_prob", 0.2)

    def export(self, export_path: str):
        # Write instance attributes into single json file.
        raise NotImplementedError()

class RNN_Model(torch.nn.Module):
    def __init__(self, config: RNN_Config):
        super().__init__()
        self.config = config
        self.rnn = None

default_bilstm_dict = {
    "input_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2
}

default_bigru_dict = {
    "input_size": 1,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2
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
        self.dropout = torch.nn.Dropout(self.config.dropout_prob)
        self.classifier = torch.nn.Linear(
            (2 if self.rnn.bidirectional else 1) * self.config.hidden_size,
            self.config.num_labels
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
        self.dropout = torch.nn.Dropout(self.config.dropout_prob)
        self.classifier = torch.nn.Linear(
            (2 if self.rnn.bidirectional else 1) * self.config.hidden_size,
            self.config.num_labels
        )

    def forward(self, input, hidden_unit=None):
        if hidden_unit:
            cn = hidden_unit[:, 0:input.shape[0], :]
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
