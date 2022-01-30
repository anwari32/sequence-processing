from aifc import Error
import torch
import os
from torch import nn

from transformers import BertForMaskedLM, PreTrainedModel, BertModel

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

class DNAMultiTask(nn.Module):
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

def initialize_training_model(pretrained_path=None, device='cpu'):
    prom_head = PromoterHead()
    ss_head = SpliceSiteHead()
    polya_head = PolyAHead()
    if os.path.isdir(pretrained_path):
        bert_layer = BertForMaskedLM.from_pretrained(pretrained_path).bert
        model = DNAMultiTask(bert_layer, prom_head, ss_head, polya_head)
        return model
    else:
        raise Error('Pretrained path not found.')

class DNASeqLabelling(nn.Module):
    """
    Core architecture of sequential labelling.
    """
    def __init__(self, bert, device='cpu'):
        super().__init__()
        self.bert = bert
        self.stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, input_ids, attention_masks):
        output = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        output = output[0][:,0,:]
        output = self.stack(output)
        return output

def initialize_sequence_labelling_model(pretrained_path, device='cpu'):
    bert = BertModel.from_pretrained(pretrained_path)
    dnabertseq = DNASeqLabelling(bert)
    return dnabertseq

