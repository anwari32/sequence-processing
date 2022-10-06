r"""
RNN model implementation.
"""

import torch

class LSTM_SL(torch.nn.Module):
    def __init__(self):
        self.num_labels = 8
        self.num_layers = 1
        self.bidirectional = True

    def forward(self, input):
        output = input
        return output


