from torch import cuda
from torch import nn
from torch.optim import AdamW
from models.lstm import LSTM_Block

_device = "cuda" if cuda.is_available() else "cpu"
_device
"""
Create simple multitask learning architecture with three task.
1. Promoter detection.
2. Splice-site detection.
3. poly-A detection.
"""

def _get_adam_optimizer(parameters, lr=0, eps=0, beta=0):
    return AdamW(parameters, lr=lr, eps=eps, betas=beta)

class PromoterHead(nn.Module):
    """
    Network configuration can be found in DeePromoter (Oubounyt et. al., 2019).
    Classification is done by using Sigmoid. Loss is calculated by CrossEntropyLoss.
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config["num_labels"])
        self.stack = nn.Sequential(
            nn.Linear(config["input_dim"], out_features=config["hidden_size"]), # Adapt 768 unit from BERT to 128 unit for DeePromoter's fully connected layer.
            nn.ReLU(), 
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["hidden_size"], config["num_labels"]),
        )
        self.activation = nn.Softmax(dim=1) if config["num_labels"] > 1 else nn.Sigmoid()

    def forward(self, x):
        # x = x[0][:,0,:]
        x = x[:,0,:]
        x = self.stack(x)
        x = self.activation(x)
        return x

class SpliceSiteHead(nn.Module):
    """
    Network configuration can be found in Splice2Deep (Albaradei et. al., 2020).
    Classification layer is using Softmax function and loss is calculated by CrossEntropyLoss.
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config["num_labels"])
        self.stack = nn.Sequential(
            nn.Linear(config["input_dim"], out_features=config["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["hidden_size"], config["num_labels"])
        )
        self.activation = nn.Softmax(dim=1) if config["num_labels"] > 1 else nn.Sigmoid()

    def forward(self, x):
        # x = x[0][:,0,:]
        x = x[:,0,:]
        x = self.stack(x)
        x = self.activation(x)
        return x

class PolyAHead(nn.Module):
    """
    Network configuration can be found in DeeReCT-PolyA (Xia et. al., 2018).
    Loss function is cross entropy and classification is done by using Softmax.
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config["num_labels"])
        self.stack = nn.Sequential(
            nn.Linear(config["input_dim"], config["hidden_size"]), # Adapt from BERT layer which provide 768 outputs.
            nn.ReLU(), # Assume using ReLU.
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["hidden_size"], config["num_labels"]),
        )
        self.activation = nn.Softmax(dim=1) if config["num_labels"] > 1 else nn.Sigmoid()
        
    def forward(self, x):
        # x = x[0][:,0,:]
        x = x[:,0,:]
        x = self.stack(x)
        x = self.activation(x)
        return x

class DNABERT_MTL(nn.Module):
    """
    Core architecture. This architecture consists of input layer, shared parameters, and heads for each of multi-tasks.
    """
    def __init__(self, bert, config):
        super().__init__()
        self.shared_layer = bert
        self.promoter_layer = PromoterHead(config["prom_head"])
        self.splice_site_layer = SpliceSiteHead(config["ss_head"])
        self.polya_layer = PolyAHead(config["polya_head"])

    def forward(self, input_ids, attention_masks):
        x = self.shared_layer(input_ids=input_ids, attention_mask=attention_masks)
        x = x[0] # Last hidden state.
        x1 = self.promoter_layer(x)
        x2 = self.splice_site_layer(x)
        x3 = self.polya_layer(x)
        return x1, x2, x3
