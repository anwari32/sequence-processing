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
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

class DNABertBiLstmForTokenClassification(BertPreTrainedModel):
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
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.head = Head(additional_config)
        self.classifier = nn.Linear(
            rnn_config.hidden_size,
            self.num_labels
        )
        self.activation = nn.Softmax(dim=2)

        self.post_init()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0]
        output = self.dropout1(output)
        output, rnn_hidden_output= self.rnn(output)
        output = self.dropout2(output)
        output = self.head(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output, rnn_hidden_output

class DNABertBiGruForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, rnn_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.rnn = nn.GRU(
            config.hidden_size,
            rnn_config.hidden_size,
            rnn_config.num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            rnn_config.hidden_size,
            self.num_labels
        )
        self.activation = nn.Softmax(dim=2)

        self.post_init()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0]
        output = self.dropout1(output)
        output, rnn_hidden_output = self.rnn(output)
        output = self.dropout2(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output, rnn_hidden_output


    
