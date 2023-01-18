r"""
RNN model implementation based on Wisesty et. al., 2022.
"""

import torch
import os
import json

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
        self.dropout = dicts.get("dropout", 0.2)
        self.bidirectional = dicts.get("bidirectional", False)

    def save_config(self, dest_dir: str):
        """
        Save config
        """
        if os.path.isfile(dest_dir):
            raise ValueError("value must be folder not file. file found.")
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        obj = {
            "name": self.name,
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "rnn": self.rnn,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_labels": self.num_labels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional
        }
        json_obj = json.dump(obj, indent=4)
        json_path = os.path.join(dest_dir, "config.json")
        with open(json_path, "w") as json_file:
            json_file.write(json_obj)


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

    def save_pretrained(self, dest_dir):
        # save config.
        config_path = os.path.join(dest_dir, "config.json")
        self.config.save_config(config_path)

        # clean up existing model.
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        model_path = os.path.join("dest_dir", "model.bin")
        if os.path.exists(model_path):
            os.remove(model_path)

        # save model.
        torch.save(
            self.state_dict(),
            os.path.join(dest_dir, "model.bin")
        )

    @classmethod
    def from_pretrained(cls, src_dir, map_location="cpu"):
        # load config
        config_path = os.path.join(src_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("config not found.")
        json_obj = json.load(open(config_path, "r"))
        config = RNN_Config(json_obj)
        
        saved_path = os.path.join(src_dir, "model.bin")
        loaded_state_dict = torch.load(saved_path, map_location=map_location)

        model = cls(config)
        model.load_state_dict(loaded_state_dict)

        return model

default_bilstm_dict = {
    "name": "default_bilstm",
    "rnn": "bilstm",
    "input_size": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2,
    "bidirectional": True
}

default_bigru_dict = {
    "name": "default_bigru",
    "rnn": "bigru",
    "input_size": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "num_labels": 8,
    "dropout_prob": 0.2,
    "bidirectional": True
}

default_bilstm_config = RNN_Config(default_bilstm_dict)
default_bigru_config = RNN_Config(default_bigru_dict)

def __create_token_embeddings__():
    characters = ["A", "C", "G", "T"]
    idx = 0
    embeddings = []
    dictionary = {}
    for a in characters:
        for b in characters:
            for c in characters:
                v = [0 for i in range(64)]
                v[idx] = 1
                token = f"{a}{b}{b}"
                dictionary[token] = idx
                embeddings.append(v)
    
    return embeddings, dictionary

def __create_3mer_embeddings__():
    characters = ["A", "C", "G", "T"]
    embeddings = []
    dictionary = {}

    # added zero-vector embedding for padding.
    embeddings.append(
        [0 for i in range(64)]
    )
    dictionary["NNN"] = 0
    index = 0
    for a in characters:
        for b in characters:
            for c in characters:
                tokens = f"{a}{b}{c}"
                vectors = [0 for i in range(64)]
                vectors[index] = 1
                index += 1
                dictionary[index] = tokens
                embeddings.append(vectors)

    return embeddings, dictionary


default_token_embeddings, default_token_dictionary = __create_token_embeddings__()

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
