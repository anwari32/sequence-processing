"""
Binary classification model.
"""

import torch

class DNABERT_BINARY(torch.nn.Module):
    def __init__(self, bert, config):
        super().__init__()
        self.bert = bert
        self.num_labels = 2
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = 768
        self.hidden_layer = torch.nn.Sequential()
        for i in range(self.num_layers):
            self.hidden_layer.add_module(
                f"linear-{i}", torch.nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.hidden_layer.add_module(
                f"relu-{i}", torch.nn.ReLU()
            )
        self.dropout = torch.nn.Dropout(p=config.get("dropout_prob", 0.1))
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.Softmax(2)
        
        self.freeze_bert = config.get("freeze_bert", False)
        if self.freeze_bert:
            print(f"Freezing BERT")
            for p in self.bert.parameters():
                p.requires_grad = False


    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids, attn_mask)
        output = output[0] # Last hidden layer.
        output = self.dropout(output)
        output = self.hidden_layer(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output