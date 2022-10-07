import json
from multiprocessing.sharedctypes import Value
from pathlib import Path, PureWindowsPath
import torch
from torch.cuda.amp import autocast
import sys
import os
import wandb
from utils.cli import parse_args
from utils.utils import create_loss_weight
from utils.seqlab import Index_Dictionary, preprocessing_kmer
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from models.rnn import RNN_BiGRU, RNN_BiLSTM
from utils.metrics import Metrics
from utils.seqlab import NUM_LABELS
import pandas as pd
import pathlib

default_baseline_training_config = {}

class BaselineConfig:
    def __init__(self, dict):
        self.num_epochs = dict.get("num_epochs")
        self.batch_size = dict.get("batch_size")
        self.training_index = dict.get("training_index")
        self.validation_index = dict.get("validation_index")
        self.test_index = dict.get("test_index")
        self.gene_dir = dict.get("gene_dir")
        self.optimizer = dict.get("optimizer")
        self.learning_rate = self.optimizer.get("learning_rate")
        self.epsilon = self.optimizer.get("epsilon")
        self.beta1 = self.optimizer.get("beta1")
        self.beta2 = self.optimizer.get("beta2")
        self.weight_decay = self.optimizer.get("weight_decay")
        self.tokenizer_path = os.path.join("pretrained", "3-new-12w-0")
    # "training_index": "index\\gene_train_index.1.csv",
    # "validation_index": "index\\gene_validation_index.1.csv",
    # "test_index": "index\\gene_test_index.1.csv",
    # "gene_dir": "workspace\\genlab\\genlab\\gene_dir",

class Baseline:
    def __init__(self, config: BaselineConfig, model=None):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        self.training_index = self.config.training_index
        self.validation_index = self.config.validation_index
        self.test_index = self.config.test_index
        self.gene_dir = self.config.gene_dir
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.training_log_path = None
        self.validation_log_path = None
        self.test_log_path = None
        self.tokenizer = BertTokenizer.from_pretrained(self.config.tokenizer_path)

    def train_and_val(self):
        # Training and validation.
        if not self.model:
            raise ValueError(f"model {self.model} is None")
        if not self.optimizer:
            raise ValueError(f"optimizer {self.optimizer} is None")

        # Metrics
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("training_step")
        wandb.define_metric("training/*", step_metric="training_step")
        wandb.define_metric("validation_step")
        wandb.define_metric("validation/*", step_metric="validation_step")
        
        for epoch in range(self.num_epochs):

            # Training
            for gene in self.training_index:
                raise NotImplementedError()

            # Validation
            for gene in self.validation_index:
                raise NotImplementedError()
            
        raise NotImplementedError()

    def predict(self):
        # Basically this is model testing.
        test_index_df = pd.read_csv(self.test_index)
        for i, r in tqdm(test_index_df.iterrows(), total=test_index_df.shape[0], desc="Predicting"):
            gene_path = os.path.join(self.gene_dir, r["chr"], r["gene"])
            dataloader = preprocessing_kmer(gene_path, self.tokenizer)
        raise NotImplementedError()

if __name__ == "__main__":
    baseline = Baseline(default_baseline_training_config)
    baseline.train_and_val()
    baseline.predict()

