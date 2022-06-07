from turtle import forward
from torch import nn

class LinearBlock(nn.Module):
    """
    Linear layer.
    """
    def __init__(self, num_layers=1):
        super().__init__()

        self.layers = nn.Sequential()
        for n in range(num_layers):
            if n + 1 == num_layers:
                self.layers.add_module(
                    f"linear-{n}", nn.Linear(768, 11)
                )
            else:
                self.layers.add_module(
                    f"linear-{n}", nn.Linear(768, 768)
                )

    def forward(self, input):
        return self.layers(input)

class RNNBlock(nn.Module):
    """
    RNN Block
    """
    def __init__(self, rnn="lstm", num_layers=1, dropout=0):
        """
        @param rnn: "lstm", "bilstm", "gru", and "bigru
        """

        super().__init__()

        self.rnn = None
        self.last_hn_cn = None # Tuple, to store hidden state.
        if rnn == "bilstm":
            self.rnn = nn.LSTM(
                input_size = 768,
                hidden_size = 768,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = True
            )
        elif rnn == "bigru":
            self.rnn = nn.GRU(
                input_size = 768,
                hidden_size = 768,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = True
            )
        elif rnn == "gru":
            self.rnn = nn.GRU(
                input_size = 768,
                hidden_size = 768,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = False
            )
        else:
            self.rnn = nn.LSTM(
                input_size = 768,
                hidden_size = 768,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout,
                bidirectional = False
            )
    
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

        output, (hn, cn) = self.rnn(input, self.last_hn_cn)
        self.last_hn_cn = (hn, cn)
        return output, (hn, cn)

class DNABERT_GSL(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert, config):
        """
        This model uses BERT as its feature extraction layer.
        This BERT layer is initiated from pretrained model which is located at `bert_pretrained_path`.
        @return object of this class.
        """
        super().__init__()
        
        self.bert = bert
        self.rnn_num_layers = 1
        self.rnn_dropout = 0
        self.rnn_name = "lstm"
        self.lin_num_layers = 1
        if config:
            rnn_cfg = config["rnn"]
            self.rnn_name = rnn_cfg.get("rnn", False)
            self.rnn_num_layers = rnn_cfg.get("num_layers", 1)
            self.rnn_dropout = rnn_cfg.get("dropout", 0)

            lin_cfg = config["linear"]
            self.lin_num_layers = lin_cfg["num_layers"]

        self.rnn = RNNBlock(self.rnn_name, self.rnn_num_layers, self.rnn_dropout)
        self.linear = LinearBlock(self.lin_num_layers)
        self.activation = nn.Softmax(dim=2)

    # def forward(self, input_ids, attention_masks, token_type_ids):
    # No need to include `token_type_ids`` since this is single sequence-related prediction.
    # `token_type_ids` is used for next sentence prediction where input contains two sentence.
    def forward(self, input_ids, attention_masks):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        output = output[0] # Last hidden state
        output, (h_n, c_n) = self.rnn(output)
        output = self.linear(output)
        output = self.activation(output)
        return output