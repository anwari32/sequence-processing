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
from utils.seqlab import Index_Dictionary, preprocessing_gene_kmer, preprocessing_kmer
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from models.rnn import RNN_BiGRU, RNN_BiLSTM, RNN_Config
from utils.metrics import Metrics
from utils.seqlab import NUM_LABELS
import pandas as pd
import pathlib
from datetime import datetime
from models import default_bilstm_config, default_bigru_config

class Baseline_Config:
    def __init__(self, dict):
        self.name = dict.get("name")
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
        self.gene_training_index = [os.path.join(self.gene_dir, p) for p in self.training_index]
        self.gene_validation_index = [os.path.join(self.gene_dir, p) for p in self.validation_index]
        self.gene_test_index = [os.path.join(self.gene_dir, p) for p in self.test_index]

default_baseline_training_config_dict = {
    "name": "gidx01-adamw-lr3e-3",
    "pretrained": "pretrained\\3-new-12w-0",
    "training_index": "index\\gene_train_index.csv",
    "validation_index": "index\\gene_validation_index.csv",
    "test_index": "index\\gene_test_index.csv",
    "gene_dir": "data\\gene_dir",
    "num_epochs": 5,
    "batch_size": 32,
    "optimizer": {
        "name": "adamw",
        "learning_rate": 3e-5,
        "epsilon": 1e-8,
        "beta1": 0.9,
        "beta2": 0.98,
        "weight_decay": 0.01
    }
}

default_baseline_training_config = Baseline_Config(default_baseline_training_config_dict)

class Baseline:
    def __init__(self, config: Baseline_Config, model_config: RNN_Config):
        self.config = config
        self.config_name = self.config.name
        if model_config.rnn == "bilstm":
            self.model = RNN_BiLSTM(model_config)
        elif model_config.rnn == "bigru":
            self.model = RNN_BiGRU(model_config)
        else:
            raise ValueError("Model config not acceptable.")
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
        self.gene_training_index = self.config.gene_training_index
        self.gene_validation_index = self.config.gene_validation_index
        self.gene_test_index = self.config.gene_test_index
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.training_log_path = None
        self.validation_log_path = None
        self.test_log_path = None
        self.tokenizer = BertTokenizer.from_pretrained(self.config.tokenizer_path)
        self.loss_function = "crossentropy"
        self.project_name = "baseline"
        self.mode = "release"

    def train_and_val(self, device, device_list=[]):
        # training and validation.
        if not self.model:
            raise ValueError(f"model {self.model} is None")
        if not self.optimizer:
            raise ValueError(f"optimizer {self.optimizer} is None")
        
        device_names = [torch.cuda.get_device_name(i) for i in device_list]
        run_id = wandb.util.generate_id()
        cur_date = datetime.now()
        runname = f"{self.config_name}"
        run = wandb.init(project=self.project_name, entity="anwari32", config={
            "device": device,
            "device_list": device_names,
            "model_config": self.config_name,
            "training_data": len(self.gene_training_index),
            "validation_data": len(self.gene_validation_index),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "training_date": cur_date,
            "initial_learning_rate": self.learning_rate,
            "use_weighted_loss": self.use_weighted_loss,
        }, reinit=True, resume='allow', id=run_id, name=runname) 

        # metrics
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("training_step")
        wandb.define_metric("training/*", step_metric="training_step")
        wandb.define_metric("validation_step")
        wandb.define_metric("validation/*", step_metric="validation_step")

        # set model to device.
        self.model.to(device)

        # steps.
        training_step = 0
        validation_step = 0
        
        for epoch in range(self.num_epochs):

            # training
            self.model.train()
            for gene in self.gene_training_index:
                dataloader = preprocessing_gene_kmer(gene, self.tokenizer, self.batch_size)
                loss_weight = create_loss_weight(gene)
                self.loss_function = torch.nn.CrossEntropyLoss(loss_weight)
                for step, batch in enumerate(dataloader):
                    input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(self.device) for t in batch)
                    with autocast():
                        prediction, hidden_output = self.model(input_ids, attention_mask)
                        batch_loss = self.loss_function(prediction.view(-1, 8), batch_labels.view(-1))

                    batch_loss.backward()
                    self.optimizer.step()
                    lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "training/learning_rate": lr,
                        "training/loss": batch_loss.item(),
                        "training_step": training_step
                    })
                    prediction_values, prediction_indices = torch.argmax(prediction.view(-1, 8))
                    metrics = Metrics(prediction_indices, batch_labels.view(-1))
                    for n in range(NUM_LABELS):
                        token_label = Index_Dictionary[n]
                        wandb.log({
                            f"training/precision-{token_label}": metrics.precision(n),
                            f"training/recall-{token_label}": metrics.recall(n),
                            f"training/f1_score-{token_label}": metrics.f1_score(n),
                            "training_step": training_step
                        })
                    training_step += 1
            
            self.scheduler.step()

            # validation
            self.model.eval()
            for gene in self.gene_validation_index:
                loss_weight = create_loss_weight(gene)
                self.loss_function = torch.nn.CrossEntropyLoss(loss_weight)
                with torch.no_grad():
                    for step, batch in enumerate(dataloader):
                        input_ids, attention_mask, token_type_ids, labels = tuple(t.to(self.device) for t in batch)
                        prediction, hidden_output = self.model(input_ids, attention_mask)
                        loss = self.loss_function(prediction.view(-1, 8), labels.view(-1))
                        wandb.log({
                            "validation/loss": loss.item(),
                            "validation_step": validation_step
                        })
                        prediction_values, prediction_indices = torch.argmax(prediction.view(-1, 8))
                        metrics = Metrics(prediction_indices, labels.view(-1))
                        for n in range(NUM_LABELS):
                            token_label = Index_Dictionary[n]
                            wandb.log({
                                f"validation/precision-{token_label}": metrics.precision(n),
                                f"validation/recall-{token_label}": metrics.recall(n),
                                f"validation/f1_score-{token_label}": metrics.f1_score(n),
                                f"validation_step": validation_step
                            })
                        validation_step += 1

    def predict(self):
        # Basically this is model testing.
        test_index_df = pd.read_csv(self.test_index)
        for i, r in tqdm(test_index_df.iterrows(), total=test_index_df.shape[0], desc="Predicting"):
            gene_path = os.path.join(self.gene_dir, r["chr"], r["gene"])
            dataloader = preprocessing_kmer(gene_path, self.tokenizer)
        raise NotImplementedError()

if __name__ == "__main__":
    bilstm_training = Baseline(default_baseline_training_config, default_bilstm_config)
    bilstm_training.mode = "debugging"
    bigru_training = Baseline(default_baseline_training_config, default_bigru_config)
    bigru_training.mode = "debugging"
    print(bilstm_training.model)
    print(bigru_training.model)

