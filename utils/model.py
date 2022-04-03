import json
from transformers import BertForMaskedLM
from models.mtl import MTModel
from models.seqlab import DNABERTSeqLab

# def init_mtl_model(pretrained_path: str, config: json, device="cpu"):
def init_mtl_model(config: str):
    cfg = json.load(open(config, "r"))
    bert = BertForMaskedLM.from_pretrained(cfg["pretrained"]).bert
    model = MTModel(bert, cfg)
    return model

# def init_seqlab_model(pretrained_path: str, config: json, device="cpu"):
def init_seqlab_model(config: str):
    cfg = json.load(open(config, "r"))
    bert = BertForMaskedLM.from_pretrained(cfg["pretrained"]).bert
    model = DNABERTSeqLab(bert, config)
    return model