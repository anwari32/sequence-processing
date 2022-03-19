from multiprocessing.sharedctypes import Value
from torch import nn
from transformers import BertForMaskedLM
import os

class Seq2SeqBlock(nn.Module):
    def __init__(self, in_dims, out_dims, norm_layer=False, prob=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dims, out_features=out_dims)
        self.norm_layer = nn.LayerNorm(out_dims) if norm_layer else None
        self.dropout = nn.Dropout(p=prob)
        self.activation = nn.ReLU()
    
    def forward(self, input):
        output = self.linear(input)
        if self.norm_layer != None:
            output = self.norm_layer(output)
        output = self.dropout(output)
        output = self.activation(output)
        return output

class LSTM_Block(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.lstm = nn.LSTM(
            input_size = config["input_dim"],
            hidden_size = config["hidden_dim"],
            num_layers = config["num_layers"],
            batch_first = True,
            dropout = config["dropout"]
        )

    def forward(self, input):
        return self.lstm(input)


class Seq2SeqHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = LSTM_Block(config["lstm"]) if config["use_lstm"] > 0 else None
        num_blocks = config["linear"]["num_layers"] 
        num_labels = config["linear"]["num_labels"]
        dim = config["linear"]["hidden_dim"] 
        norm_layer =  (config["linear"]["norm_layer"] > 0) 
        dropout_prob= config["linear"]["dropout"] if "linear" in config["linear"] else 0.1
        self.linear = nn.Sequential()
        for i in range(num_blocks):
            self.linear.add_module(
                "seq2seq_block-{}".format(i), Seq2SeqBlock(dim, dim, norm_layer=norm_layer, prob=dropout_prob)
            )
        #self.hidden_layers = [nn.Linear(d[0], d[1]) for d in dims_ins_outs]
        #self.norm_layer = [nn.LayerNorm(d[0]) for d in dims_ins_outs]
        self.classifier = nn.Linear(in_features=dim, out_features=num_labels)
        #for i in range(0, len(self.hidden_layers)):
        #    linear_layer = self.hidden_layers[i]
        #    self.stack.add_module("hidden-{}".format(i+1), linear_layer)
        #    if norm_layer:
        #        self.stack.add_module("norm_layer-{}".format(i+1), )
        #    self.stack.add_module("relu-{}".format(i+1), nn.ReLU())
        #self.stack.add_module("dropout-1", nn.Dropout(0.1))
    
    def forward(self, input):
        x = self.lstm(input) if self.lstm else input
        x = self.linear(input)
        x = self.classifier(x)
        return x



class DNABERTSeq2Seq(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, config):
        """
        This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @param  bert_pretrained_path (string): Path to DNABERT pretrained.
        @param  seq2seq_dims:
        @param  loss_strategy (string) | None -> "sum"
        @param  device (string): Default is 'cpu' but you can put 'cuda' if your machine supports cuda.
        @return (DNASeqLabelling): Object of this class.
        """
        super().__init__()
        if "pretrained" not in config.keys():
            raise KeyError("Pretrained path not found in config. Check if there is `pretrained` key in config.")
        bert_pretrained_path = config["pretrained"]
        if not os.path.exists(bert_pretrained_path):
            raise FileNotFoundError(bert_pretrained_path)
        if not os.path.isdir(bert_pretrained_path):
            raise IsADirectoryError(bert_pretrained_path)

        self.bert = BertForMaskedLM.from_pretrained(bert_pretrained_path).bert
        self.seq2seq_head = Seq2SeqHead(config)
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_masks, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = output[0] # Last hidden state
        output = self.seq2seq_head(output)
        output = self.activation(output)
        return output