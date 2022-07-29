from unicodedata import bidirectional
from torch import nn

class LSTM_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_size = config["input_dim"]
        self.hidden_size = config["hidden_dim"]
        self.num_layers = config["num_layers"]  

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
        if self.last_hn_cn != None:

            # Somehow I need to match hidden state (h) and cell state (c) dimension to input dimension.
            # Case: input dimension (1, 512, 768), h dimension (4, 4, 768) and c dimension (4, 4, 768).
            # Input (1, 512, 768) requires h and c (4, 1, 768).
            # Proposed treatment: adjust h and c dimension.
            hn = self.last_hn_cn[0]
            cn = self.last_hn_cn[1]

            batch_size = input.shape[0]
            hn_cn_second_dim = hn.shape[1]
            if batch_size < hn_cn_second_dim:
                hn = hn[:,0:batch_size,:].contiguous()
                cn = cn[:,0:batch_size,:].contiguous()

            # self.last_hn_cn = (self.last_hn_cn[0].detach(), self.last_hn_cn[1].detach())
            self.last_hn_cn = (hn.detach(), cn.detach())

        output, (hn, cn) = self.lstm(input, self.last_hn_cn)
        self.last_hn_cn = (hn, cn)
        return output, (hn, cn)

    def reset_hidden(self):
        self.last_hn_cn = None