from torch import nn
from .lstm import LSTM_Block
import utils.seqlab

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
        self.dim = 768
        self.input_dim = 768
        self.norm_layer = False
        self.dropout = 0
        self.num_blocks = 1
        self.num_labels = utils.seqlab.NUM_LABELS

        if config:
            self.num_blocks = config["linear"]["num_layers"] 
            self.num_labels = config["linear"]["num_labels"]
            self.input_dim = config["linear"]["input_dim"]
            self.dim = config["linear"]["hidden_dim"] 
            self.norm_layer = (config["linear"]["norm_layer"] > 0) 
            self.dropout= config["linear"]["dropout"] if "linear" in config["linear"] else 0.1
        
        self.linear = nn.Sequential()
        for i in range(self.num_blocks):
            if i > 0:
                self.linear.add_module(
                    "seq2seq_block-{}".format(i), SeqLabBlock(self.dim, self.dim, norm_layer=self.norm_layer, prob=self.dropout)
                )
            else:
                self.linear.add_module(
                    "seq2seq_block-{}".format(i), SeqLabBlock(self.input_dim, self.dim, norm_layer=self.norm_layer, prob=self.dropout)
                )
        #self.hidden_layers = [nn.Linear(d[0], d[1]) for d in dims_ins_outs]
        #self.norm_layer = [nn.LayerNorm(d[0]) for d in dims_ins_outs]
        self.classifier = nn.Linear(in_features=self.dim, out_features=self.num_labels)
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
        x = self.linear(x)
        x = self.classifier(x)
        return x

class DNABERT_SL(nn.Module):
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
        @return object of this class.
        """
        super().__init__()
        
        self.bert = bert
        self.seqlab_head = SeqLabHead(config)
        self.activation = nn.Softmax(dim=2)

        freeze = config.get("freeze_bert", False)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    # def forward(self, input_ids, attention_masks, token_type_ids):
    # No need to include `token_type_ids`` since this is single sequence-related prediction.
    # `token_type_ids` is used for next sentence prediction where input contains two sentence.
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_masks, output_attentions=True)
        output = bert_output[0] # Last hidden state
        output = self.seqlab_head(output)
        output = self.activation(output)
        return output, bert_output
