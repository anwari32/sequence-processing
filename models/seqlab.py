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
        
        if config:
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
        @return object of this class.
        """
        super().__init__()
        
        self.bert = bert
        self.head = Head(config)
        self.activation = nn.Softmax(dim=2)

    # def forward(self, input_ids, attention_masks, token_type_ids):
    # No need to include `token_type_ids`` since this is single sequence-related prediction.
    # `token_type_ids` is used for next sentence prediction where input contains two sentence.
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)            
        bert_output = output[0]
        output = output[0]
        output = self.head(output)
        head_output = output
        output = self.activation(output)
        return output, bert_output, head_output

class DNABERT_BILSTM(nn.Module):
    def __init__(self, bert, config):
        super().__init__()

        self.bert = bert
        self.dropout1 = nn.Dropout(config.get("dropout", 0.1))
        self.rnn = nn.LSTM(
            768, 
            config.get("rnn_hidden_dim", 256), 
            config.get("rnn_layer", 2),
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(config.get("dropout", 0.1))
        self.classifier = nn.Linear(
            config.get("rnn_hidden_dim", 256),
            config.get("num_labels", 8)
        )
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0]
        output = self.dropout1(output)
        output, hidden_outputs = self.rnn(output)
        output = self.dropout2(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output, hidden_outputs

class DNABERT_BIGRU(nn.Module):
    def __init__(self, bert, config):
        super().__init__()

        self.bert = bert
        self.dropout1 = nn.Dropout(config.get("dropout", 0.1))
        self.rnn = nn.GRU(
            768, 
            config.get("rnn_hidden_dim", 256), 
            config.get("rnn_layer", 2),
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(config.get("dropout", 0.1))
        self.classifier = nn.Linear(
            config.get("rnn_hidden_dim", 256),
            config.get("num_labels", 8)
        )
        self.activation = nn.Softmax(dim=2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0]
        output = self.dropout1(output)
        output, hidden_output = self.rnn(output)
        output = self.dropout2(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output, hidden_output