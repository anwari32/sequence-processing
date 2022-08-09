"""
This training utilizes a gene as an instance instead of contig as found in run_gene_labelling.py.
Command line accepts several input; one of which is train_genes and validation_genes.
"""

from hashlib import new
import sys
import pandas as pd
import os
import json
import torch
import wandb

from getopt import getopt
from pathlib import Path, PureWindowsPath
from sched import scheduler
from utils.cli import parse_args
from models.genlab import DNABERT_RNN
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch import autocast, no_grad
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.seqlab import preprocessing_kmer
from utils.utils import create_loss_weight
from utils.metrics import accuracy_and_error_rate

def train(model, optimizer, scheduler, gene_dir, training_index_path, validation_index_path, tokenizer, save_dir, num_epochs=1, start_epoch = 0, batch_size=1, use_weighted_loss=False):
    training_index_df = pd.read_csv(training_index_path)
    training_genes = []
    for i, r in training_index_df.iterrows():
        chr_dir = r["chr"]
        g_file = r["gene"]
        training_genes.append(os.path.join(gene_dir, chr_dir, g_file))

    validation_index_df = pd.read_csv(validation_index_path)
    validation_genes = []
    for i, r in validation_index_df.iterrows():
        chr_dir = r["chr"]
        g_file = r["gene"]
        validation_genes.append(os.path.join(gene_dir, chr_dir, g_file))
    
    training_log_path = os.path.join(save_dir, "training_log.csv")
    validation_log_path = os.path.join(save_dir, "validation_log.csv")
    training_log = open(training_log_path, "x")
    training_log.write("epoch,step,loss\n")
    validation_log = open(validation_log_path, "x")
    validation_log.write("epoch,step,sequence,prediction,target,accuracy,error_rate\n")

    num_labels = model.num_labels
    model.to(device)
    for epoch in tqdm(range(start_epoch, num_epochs), total=(num_epochs-start_epoch), desc="Training "):
        model.train()
        for training_gene_file in training_genes:
            dataloader = preprocessing_kmer(training_gene_file, tokenizer, batch_size, disable_tqdm=True)
            loss_weight = None
            if use_weighted_loss:
                loss_weight = create_loss_weight(training_gene_file)
                loss_weight = loss_weight.to(device)
            criterion = CrossEntropyLoss(weight=loss_weight)
            hidden_output = None
            for step, batch in enumerate(dataloader):
                input_ids, attention_masks, token_type_ids, labels = tuple(t.to(device) for t in batch)
                with autocast():
                    prediction, hidden_output = model(input_ids, attention_masks, hidden_output)
                    loss = criterion(prediction.view(-1, num_labels), labels.view(-1))
                training_log.write(f"{epoch},{step},{loss.item()}")
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        
        model.eval()
        for validation_gene_file in validation_genes:
            dataloader = preprocessing_kmer(validation_gene_file, tokenizer, batch_size, disable_tqdm=True)
            hidden_output = None
            gene_labelling = []
            for step, batch in enumerate(dataloader):
                input_ids, attention_masks, token_type_ids, labels = tuple(t.to(device) for t in batch)
                with no_grad():
                    prediction, hidden_output = model(input_ids, attention_masks, hidden_output)
                values, indices = torch.max(prediction, 2)
                for i, j, k in zip(input_ids, indices, labels):
                    ilist = i.tolist()
                    jlist = j.tolist()
                    klist = k.tolist()
                    accuracy, error_rate = accuracy_and_error_rate(i, j, k)
                    validation_log.write(f"{epoch},{step},{ilist},{klist},{jlist},{accuracy},{error_rate}\n")

        checkpoint_path = os.path.join(save_dir, "latest", "checkpoint.pth")
        save_path = os.path.join(save_dir, f"checkpoint-{epoch}.pth")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, save_path)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    training_config = json.load(open(args.get("training-config", False), "r"))
    batch_size = args.get("batch-size", training_config.get("batch_size", 1))
    num_epochs = args.get("num-epochs", training_config.get("num_epochs", 1))
    gene_dir = training_config.get("gene_dir", False)
    training_index = training_config.get("training_index", False)
    validation_index = training_config.get("validation_index", False)
    test_index = training_config.get("test_index", False)
    model_config_dir = args.get("model-config-dir", False)
    model_config_names = args.get("model-config-names", [])
    optimizer_config = training_config.get("optimizer", {})
    learning_rate = optimizer_config.get("learning_rate", 4e-4)
    epsilon = optimizer_config.get("epsilon", 1e-6)
    beta1 = optimizer_config.get("beta1", 0.98)
    beta2 = optimizer_config.get("beta2", 0.9)
    weight_decay = optimizer_config.get("weight_decay", 0.01)
    device = args.get("device", "cuda:0")
    device_list = args.get("device-list", [])
    use_weighted_loss = args.get("use-weighted-loss", False)
    run_name = args.get("run-name", "genlab")
    resume_run_ids = args.get("resume-run_ids", [None for a in model_config_names])
    project_name = args.get("project-name", "pilot-project")

    gene_dirpath = str(Path(PureWindowsPath(gene_dir)))
    training_index_path = str(Path(PureWindowsPath(training_index)))
    validation_index_path = str(Path(PureWindowsPath(validation_index)))
    test_index_path = str(Path(PureWindowsPath(test_index)))

    pretrained = str(Path(PureWindowsPath(training_config.get("pretrained", os.path.join("pretrained", "3-new-12w-0")))))
    bert_for_masked_lm = BertForMaskedLM.from_pretrained(pretrained)
    bert = bert_for_masked_lm.bert
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    for config_name, resume_run_id in zip(model_config_names, resume_run_ids):
        config_path = os.path.join(model_config_dir, config_name)
        config = json.load(open(config_path, "r"))
        model = DNABERT_RNN(bert, config)
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        run_id = resume_run_id if resume_run_id else wandb.util.generate_id()
        runname = f"{run_name}-{config_name}-{run_id}"
        save_dir = os.path.join("run", runname)
        save_checkpoint_dir = os.path.join(save_dir, "latest")
        for d in [save_dir, save_checkpoint_dir]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
        run = wandb.init(id=run_id, project=project_name, resume='allow', reinit=True, name=runname)
        start_epoch = 0
        if run.resumed:
            checkpoint_path = os.path.join("run", runname, "latest", "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(torch.load(checkpoint.get("model")))
                optimizer.load_state_dict(torch.load(checkpoint.get("optimizer")))
                scheduler.load_state_dict(torch.load(checkpoint.get("scheduler")))
                epoch = checkpoint.get("epoch")
                start_epoch = epoch + 1

        train(model, optimizer, scheduler, gene_dir, training_index_path, validation_index_path, tokenizer, save_dir, num_epochs, start_epoch, batch_size, use_weighted_loss)
        run.finish()


