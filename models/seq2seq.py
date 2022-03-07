from torch import nn
from transformers import BertForMaskedLM
import os

class Seq2SeqHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dims_ins_outs = [dims[i:i+2] for i in range(len(dims)-2+1)]
        self.hidden_layers = [nn.Linear(d[0], d[1]) for d in dims_ins_outs]
        self.stack = nn.Sequential()
        self.activation = nn.LogSoftmax(dim=1)
        for i in range(0, len(self.hidden_layers)):
            linear_layer = self.hidden_layers[i]
            self.stack.add_module("hidden-{}".format(i+1), linear_layer)
            self.stack.add_module("relu-{}".format(i+1), nn.ReLU())
        self.stack.add_module("dropout-1", nn.Dropout(0.1))
    
    def forward(self, input):
        x = self.stack(input)
        x = self.activation(x)
        return x

class DNABERTSeq2Seq(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert_pretrained_path, seq2seq_dims=[768, 512, 512, 10], loss_strategy="sum", device='cpu'):
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
        self.seq2seq_head = Seq2SeqHead(seq2seq_dims)
        self.loss_function = nn.NLLLoss()
        self.loss_strategy = loss_strategy
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_masks, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = output[0]
        output = self.seq2seq_head(output)
        output = self.activation(output)
        return output