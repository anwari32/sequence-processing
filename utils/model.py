import json
from transformers import BertForMaskedLM
from models.mtl import MTModel
from models.seqlab import DNABERTSeqLab
from pathlib import Path

# def init_mtl_model(pretrained_path: str, config: json, device="cpu"):
def init_mtl_model(config_path: str):
    cfg = json.load(open(config_path, "r"))
    bert = BertForMaskedLM.from_pretrained(Path(cfg["pretrained"])).bert
    model = MTModel(bert, cfg)
    return model

# def init_seqlab_model(pretrained_path: str, config: json, device="cpu"):
def init_seqlab_model(config_path: str):
    cfg = json.load(open(config_path, "r"))
    bert = BertForMaskedLM.from_pretrained(cfg["pretrained"]).bert
    model = DNABERTSeqLab(bert, cfg)
    return model