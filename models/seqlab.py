from multiprocessing.sharedctypes import Value
from torch import nn
from transformers import BertForMaskedLM
import os
from models.lstm import LSTM_Block

class SeqLabBlock(nn.Module):
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

class SeqLabHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = LSTM_Block(config["lstm"]) if config["use_lstm"] > 0 else None
        num_blocks = config["linear"]["num_layers"] 
        num_labels = config["linear"]["num_labels"]
        input_dim = config["linear"]["input_dim"]
        dim = config["linear"]["hidden_dim"] 
        norm_layer =  (config["linear"]["norm_layer"] > 0) 
        dropout_prob= config["linear"]["dropout"] if "linear" in config["linear"] else 0.1
        self.linear = nn.Sequential()
        for i in range(num_blocks):
            if i > 0:
                self.linear.add_module(
                    "seq2seq_block-{}".format(i), SeqLabBlock(dim, dim, norm_layer=norm_layer, prob=dropout_prob)
                )
            else:
                self.linear.add_module(
                    "seq2seq_block-{}".format(i), SeqLabBlock(input_dim, dim, norm_layer=norm_layer, prob=dropout_prob)
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
        #print(self.lstm)
        #print(self.linear)
        #print(self.classifier)
    
    def forward(self, input):
        x = input
        if self.lstm:
            x, (h_n, c_n) = self.lstm(input)
        #print(x.shape)
        x = self.linear(x)
        #print(x.shape)
        x = self.classifier(x)
        return x

class DNABERTSeqLab(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert, config):
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
        
        self.bert = bert
        self.seqlab_head = SeqLabHead(config)
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_masks, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = output[0] # Last hidden state
        output = self.seqlab_head(output)
        output = self.activation(output)
        return output