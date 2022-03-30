from unicodedata import bidirectional
from torch import nn

class LSTM_Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = config["input_dim"],
            hidden_size = config["hidden_dim"],
            num_layers = config["num_layers"],
            batch_first = True,
            dropout = config["dropout"],
            bidirectional = False if config["bidirectional"] <= 0 else True
        )

        self.last_hn_cn = None

    def forward(self, input):
        output, (hn, cn) = self.lstm(input, self.last_hn_cn.detach())
        self.last_hn_cn = (hn, cn)
        return output, (hn, cn)
