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

class Seq2SeqHead(nn.Module):
    def __init__(self, num_blocks, num_labels, dim=768, norm_layer=False, dropout_prob=0.1):
        super().__init__()
        self.stack = nn.Sequential()
        for i in range(num_blocks):
            self.stack.add_module(
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
        x = self.stack(input)
        x = self.classifier(x)
        return x

class DNABERTSeq2Seq(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert_pretrained_path, num_blocks=1, num_labels=11, hidden_dim=768, norm_layer=False, dropout_prob=0.1):
        """
        This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @param  bert_pretrained_path (string): Path to DNABERT pretrained.
        @param  seq2seq_dims:
        @param  loss_strategy (string) | None -> "sum"
        @param  device (string): Default is 'cpu' but you can put 'cuda' if your machine supports cuda.
        @return (DNASeqLabelling): Object of this class.
        """
        if not os.path.exists(bert_pretrained_path):
            raise FileNotFoundError(bert_pretrained_path)
        if not os.path.isdir(bert_pretrained_path):
            raise IsADirectoryError(bert_pretrained_path)

        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_pretrained_path).bert
        self.seq2seq_head = Seq2SeqHead(num_blocks, num_labels, hidden_dim, norm_layer=norm_layer, dropout_prob=dropout_prob)
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_masks, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = output[0] # Last hidden state
        output = self.seq2seq_head(output)
        output = self.activation(output)
        return output