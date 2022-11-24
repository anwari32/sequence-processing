from torch import nn
import utils.seqlab

class HeadBlock(nn.Module):
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

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = 768
        self.input_dim = 768
        self.norm_layer = False
        self.dropout_prob = 0.1
        self.num_blocks = 1
        self.num_labels = 8
        
        linear_config = config.get("linear")
        if linear_config:
            self.num_blocks = linear_config.get("num_layers", 0)
            self.num_labels = linear_config.get("num_labels", 8)
            self.input_dim = linear_config.get("input_dim", 768)
            self.dim = linear_config.get("hidden_dim", 768)
            self.norm_layer = linear_config.get("norm_layer", False) 
            self.dropout_prob = linear_config.get("dropout", 0.1)
        
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.linear = nn.Sequential()
        for i in range(self.num_blocks):
            if i > 0:
                self.linear.add_module(
                    "hidden-block-{}".format(i), HeadBlock(self.dim, self.dim, norm_layer=self.norm_layer, prob=self.dropout_prob)
                )
            else:
                self.linear.add_module(
                    "hidden-block-{}".format(i), HeadBlock(self.input_dim, self.dim, norm_layer=self.norm_layer, prob=self.dropout_prob)
                )
        self.classifier = nn.Linear(in_features=self.dim, out_features=self.num_labels)
    
    def forward(self, input):
        x = input
        x = self.dropout(x) # Dropout after BERT layer.
        x = self.linear(x) # Continue to linear layer.
        x = self.classifier(x) # Classification layer.
        return x

class DNABERT_SL(nn.Module):
    r"""
    Core architecture of sequential labelling.
    """
    def __init__(self, bert, config):
        r"""
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
        self.head = Head(config)
        self.activation = nn.Softmax(dim=2)

        freeze = config.get("freeze_bert", False)
        self.feature_based_approach = config.get("feature_based", "last")
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    # def forward(self, input_ids, attention_masks, token_type_ids):
    # No need to include `token_type_ids`` since this is single sequence-related prediction.
    # `token_type_ids` is used for next sentence prediction where input contains two sentence.
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_masks, output_attentions=True)
        if self.feature_based_approach == "last":
            # Use last hidden layer.
            output = bert_output[0] # Last hidden state
        elif self.feature_based_approach == "c4":
            # Concatenating last four hidden layer.
            hidden_output = bert_output[1]
            

        output = self.head(output)
        output = self.activation(output)
        return output, bert_output

from transformers import BertForTokenClassification

num_classes = 8

class DNABertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output