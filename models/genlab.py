from pathlib import Path, PureWindowsPath
from torch import nn

class LinearBlock(nn.Module):
    """
    Linear layer.
    """
    def __init__(self, num_layers=1):
        super().__init__()

        self.layers = nn.Sequential()
        self.num_layers = num_layers
        self.num_labels = 11
        for n in range(self.num_layers):
            if n + 1 == self.num_layers:
                self.layers.add_module(
                    f"linear-{n}", nn.Linear(768, self.num_labels)
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

            hn, cn = None, None
            if isinstance(self.rnn, nn.LSTM):
                hn = self.last_hn_cn[0]
                cn = self.last_hn_cn[1]

                batch_size = input.shape[0]
                hn_cn_second_dim = hn.shape[1]
                if batch_size < hn_cn_second_dim:
                    hn = hn[:,0:batch_size,:].contiguous()
                    cn = cn[:,0:batch_size,:].contiguous()

                # self.last_hn_cn = (self.last_hn_cn[0].detach(), self.last_hn_cn[1].detach())
                self.last_hn_cn = (hn.detach(), cn.detach())

            if isinstance(self.rnn, nn.GRU):
                hn = self.last_hn_cn
                
                batch_size = input.shape[0]
                hn_cn_second_dim = hn.shape[1]
                if batch_size < hn_cn_second_dim:
                    hn = hn[:,0:batch_size,:].contiguous()
                
                self.last_hn_cn = hn.detach()

        output = None
        hn = None
        cn = None
        if isinstance(self.rnn, nn.LSTM):
            output, (hn, cn) = self.rnn(input, self.last_hn_cn)
            self.last_hn_cn = (hn, cn)
            return output, (hn, cn)

        if isinstance(self.rnn, nn.GRU):
            output, hn = self.rnn(input, self.last_hn_cn)
            self.last_hn_cn = hn
            return output, hn
        
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
            self.lin_num_layers = lin_cfg.get("num_layers", 1)

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
        output, hidden_units = self.rnn(output)
        output = self.linear(output)
        output = self.activation(output)
        return output

    def reset_hidden(self):
        self.rnn.last_hn_cn = None


if __name__ == "__main__":
    from transformers import BertForMaskedLM
    import json

    pretrained_bert_path = str(Path(PureWindowsPath("dnabert-3")))
    print(f"bert path {pretrained_bert_path}")
    bert = BertForMaskedLM.from_pretrained(pretrained_bert_path).bert
    model_base = DNABERT_GSL(bert, None)
    model_gru = DNABERT_GSL(bert, json.load(open("config/genlab/gru.json", "r")))
    model_multi_lstm = DNABERT_GSL(bert, json.load(open("config/genlab/multi-lstm.json", "r")))
    model_multi_gru = DNABERT_GSL(bert, json.load(open("config/genlab/multi-gru.json", "r")))

    import torch

    input_ids = torch.randint(0, 69, (5, 512))
    attn_mask = torch.randint(0, 2, (5, 512))
    output_base = model_base(input_ids, attn_mask)
    output_gru = model_gru(input_ids, attn_mask)
    output_multi_lstm = model_multi_lstm(input_ids, attn_mask)
    output_multi_gru = model_multi_gru(input_ids, attn_mask)
    print(f"Output base (lstm) {output_base.shape}")
    print(f"Output gru {output_gru.shape}")
    print(f"Output multi lstm {output_multi_lstm.shape}")
    print(f"Output multi gru {output_multi_gru.shape}")
    print(f"LSTM Hidden state {model_base.rnn.last_hn_cn[0].shape} and cell state {model_base.rnn.last_hn_cn[1].shape}")
    print(f"GRU Hidden state {model_gru.rnn.last_hn_cn.shape}")