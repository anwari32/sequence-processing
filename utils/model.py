import json
from transformers import BertForMaskedLM
from models.mtl import DNABERT_MTL
from models.seqlab import DNABERT_SL
from pathlib import Path, PureWindowsPath

# def init_mtl_model(pretrained_path: str, config: json, device="cpu"):
def init_mtl_model(config_path: str):
    config_path = PureWindowsPath(config_path)
    config_path = str(Path(config_path))
    cfg = json.load(open(config_path, "r"))
    
    bert_path = PureWindowsPath(cfg["pretrained"])
    bert_path = str(Path(bert_path))
    bert = BertForMaskedLM.from_pretrained(bert_path).bert
    model = DNABERT_MTL(bert, cfg)
    return model

# def init_seqlab_model(pretrained_path: str, config: json, device="cpu"):
def init_seqlab_model(config_path: str):
    config_path = PureWindowsPath(config_path)
    config_path = str(Path(config_path))
    cfg = json.load(open(config_path, "r"))

    bert_path = PureWindowsPath(cfg["pretrained"])
    bert_path = str(Path(bert_path))
    bert = BertForMaskedLM.from_pretrained(bert_path).bert
    model = DNABERT_SL(bert, cfg)
    return model
