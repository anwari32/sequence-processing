from transformers import BertForTokenClassification, BertForSequenceClassification, BertPreTrainedModel, BertModel
from torch import nn

num_classes = 8

# To compensate my own doing in creating model from scratch while there is easier way. 
# I hate this.

from transformers import BertPreTrainedModel, BertModel
from .seqlab import Head
from torch.nn import Softmax

class DNABERT_SL(BertPreTrainedModel):
    def __init__(self, config, additional_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.head = Head(additional_config)
        self.activation = Softmax(dim=2)

        self.post_init()
    
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)            
        bert_output = output[0]
        output = output[0]
        output = self.head(output)
        head_output = output
        output = self.activation(output)
        return output, bert_output, head_output


class RNNConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_layers = kwargs.get("num_layers", 2)
        self.bidirectional = kwargs.get("bidirectional", False)
        self.batch_first = True

class DNABERTRNNForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, rnn_config, additional_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.rnn = nn.LSTM(
            config.hidden_size,
            rnn_config.hidden_size,
            rnn_config.num_layers,
            bidirectional=rnn_config.bidirectional,
            batch_first=True
        )

        # modify dimension of additional hidden layers.
        if rnn_config.bidirectional:
            additional_config["linear"]["input_dim"] = config.hidden_size * 2
            additional_config["linear"]["hidden_dim"] = config.hidden_size * 2

        self.head = Head(additional_config)
        self.activation = nn.Softmax(dim=2)

        self.post_init()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        bert_output = output[0]
        output = output[0]
        output = self.dropout1(output)
        output, rnn_hidden_output= self.rnn(output)
        output = self.dropout2(output)
        output = self.head(output)
        output = self.activation(output)
        return output, rnn_hidden_output, bert_output

class DNABERTLSTMForTokenClassification(DNABERTRNNForTokenClassification):
    def __init__(self, config, rnn_config, head_config):
        super().__init__(config, rnn_config, head_config)
        self.rnn = nn.LSTM(
            config.hidden_size,
            rnn_config.hidden_size,
            rnn_config.num_layers,
            bidirectional=rnn_config.bidirectional,
            batch_first=True
        )
        self.post_init()

    def forward(self, input_ids, attention_mask):
        return super().forward(input_ids, attention_mask)

class DNABERTGRUForTokenClassification(DNABERTRNNForTokenClassification):
    def __init__(self, config, rnn_config, head_config):
        super().__init__(config, rnn_config, head_config)
        self.rnn = nn.GRU(
            config.hidden_size,
            rnn_config.hidden_size,
            rnn_config.num_layers,
            bidirectional=rnn_config.bidirectional,
            batch_first=True
        )
        self.post_init()

    def forward(self, input_ids, attention_mask):
        return super().forward(input_ids, attention_mask)


    
