from transformers import BertPreTrainedModel, BertModel
from torch import nn


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