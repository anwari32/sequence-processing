from torch import cuda

_device = "cuda" if cuda.is_available() else "cpu"
_device
"""
Create simple multitask learning architecture with three task.
1. Promoter detection.
2. Splice-site detection.
3. poly-A detection.
"""
from torch import nn
from torch.optim import AdamW
from transformers import BertForMaskedLM

crossentropy_loss_func = nn.CrossEntropyLoss()

def _get_adam_optimizer(parameters, lr=0, eps=0, beta=0):
    return AdamW(parameters, lr=lr, eps=eps, betas=beta)

class PromoterHead(nn.Module):
    """
    Network configuration can be found in DeePromoter (Oubounyt et. al., 2019).
    Classification is done by using Sigmoid. Loss is calculated by CrossEntropyLoss.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, out_features=128, device=device), # Adapt 768 unit from BERT to 128 unit for DeePromoter's fully connected layer.
            nn.ReLU(), # Asssume using ReLU.
            nn.Linear(128, 2, device=device),
        )

    def forward(self, x):
        x = self.stack(x)
        return x

class SpliceSiteHead(nn.Module):
    """
    Network configuration can be found in Splice2Deep (Albaradei et. al., 2020).
    Classification layer is using Softmax function and loss is calculated by ???.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, out_features=512, device=device),
            nn.ReLU(),
            nn.Linear(512, 2, device=device)
        )

    def forward(self, x):
        x = self.stack(x)
        return x


class PolyAHead(nn.Module):
    """
    Network configuration can be found in DeeReCT-PolyA (Xia et. al., 2018).
    Loss function is cross entropy and classification is done by using Softmax.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, 64, device=device), # Adapt from BERT layer which provide 768 outputs.
            nn.ReLU(), # Assume using ReLU.
            nn.Linear(64, 2, device=device),
        )

    def forward(self, x):
        x = self.stack(x)
        return x

class MTModel(nn.Module):
    """
    Core architecture. This architecture consists of input layer, shared parameters, and heads for each of multi-tasks.
    """
    def __init__(self, shared_parameters, promoter_head, splice_site_head, polya_head):
        super().__init__()
        self.shared_layer = shared_parameters
        self.promoter_layer = promoter_head
        self.splice_site_layer = splice_site_head
        self.polya_layer = polya_head


    def forward(self, input_ids, attention_masks):
        x = self.shared_layer(input_ids=input_ids, attention_mask=attention_masks)
        x = x[0][:, 0, :]
        x1 = self.promoter_layer(x)
        x2 = self.splice_site_layer(x)
        x3 = self.polya_layer(x)
        return {'prom': x1, 'ss': x2, 'polya': x3}


polya_head = PolyAHead(_device)
promoter_head = PromoterHead(_device)
splice_head = SpliceSiteHead(_device)

dnabert_3_pretrained = './pretrained/3-new-12w-0'
shared_parameter = BertForMaskedLM.from_pretrained(dnabert_3_pretrained).bert

model = MTModel(shared_parameters=shared_parameter, promoter_head=promoter_head, polya_head=polya_head, splice_site_head=splice_head).to(_device)

def initialize_multitask_leaning(bert_pretrained_path, )