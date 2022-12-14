from transformers import BertForTokenClassification, BertForSequenceClassification, BertPreTrainedModel, BertModel
from torch import nn

num_classes = 8

class DNABertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output

class DNABertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output

class RNNConfig:
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    

class DNABertBiLstmForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, rnn_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
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


    
