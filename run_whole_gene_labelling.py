"""
This training utilizes a gene as an instance instead of contig as found in run_gene_labelling.py.
Command line accepts several input; one of which is train_genes and validation_genes.
"""

from datetime import datetime
from hashlib import new
import sys
import pandas as pd
import os
import json
import torch
import numpy as np

import wandb
from getopt import getopt
from pathlib import Path, PureWindowsPath
from sched import scheduler
from utils import parse_args
from models.genlab import DNABERT_RNN
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch import no_grad
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.seqlab import NUM_LABELS, Index_Dictionary, Label_Dictionary, preprocessing_kmer, preprocessing_whole_sequence
from utils.utils import create_loss_weight
from utils.metrics import Metrics, accuracy_and_error_rate

def train(model, optimizer, scheduler, gene_dir, training_index_path, validation_index_path, tokenizer, save_dir, wandb, num_epochs=1, start_epoch = 0, batch_size=1, use_weighted_loss=False, accumulate_gradient=False, preprocessing_mode="sparse", device_list=[]):
    if len(device_list) > 0:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    
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
    training_log = open(training_log_path, "x") if not os.path.exists(training_log_path) else open(training_log_path, "w")
    training_log.write("epoch,step,loss,accuracy,error_rate\n")
    
    num_labels = model.num_labels
    num_training_genes = len(training_genes)
    num_validation_genes = len(validation_genes)
    sequential_labels = [k for k in Label_Dictionary.keys() if Label_Dictionary[k] >= 0]
    sequential_label_indices = [k for k in range(NUM_LABELS)]
    model.to(device)
    wandb.define_metric("epoch")
    # for epoch in tqdm(range(start_epoch, num_epochs), total=(num_epochs-start_epoch), desc="Training "):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # for training_gene_file in training_genes:
        for training_gene_file in tqdm(training_genes, total=num_training_genes, desc=f"Training at Epoch {epoch + 1}/{num_epochs}"):
            # TODO: Enable preprocessing sparse and dense.
            # currently works on certain formatted sequence.
            # dataloader = preprocessing_kmer(training_gene_file, tokenizer, batch_size, disable_tqdm=True)
            dataloader = preprocessing_whole_sequence(training_gene_file, tokenizer, batch_size, dense=(preprocessing_mode == "dense"))
            chr_name = os.path.basename(os.path.dirname(training_gene_file))
            gene_name = '.'.join(os.path.basename(training_gene_file).split('.')[:-1])
            
            # training metrics
            # wandb.define_metric(f"training-{chr_name}-{gene_name}/loss", step_metric="epoch")
            # for k in sequential_labels:
            #     wandb.define_metric(f"training-{chr_name}-{gene_name}/precision-{k}", step_metric="epoch")
            #     wandb.define_metric(f"training-{chr_name}-{gene_name}/recall-{k}", step_metric="epoch")
            # wandb.define_metric(f"training-{chr_name}-{gene_name}/accuracy", step_metric="epoch")
            # wandb.define_metric(f"training-{chr_name}-{gene_name}/error_rate", step_metric="epoch")

            # loss weight
            loss_weight = None
            if use_weighted_loss:
                loss_weight = create_loss_weight(training_gene_file, ignorance_level=2, kmer=3)
                loss_weight = loss_weight.to(device)
            criterion = CrossEntropyLoss(weight=loss_weight)
            hidden_output = None
            y_prediction = []
            y_target = []
            for step, batch in enumerate(dataloader):
                input_ids, attention_masks, token_type_ids, labels = tuple(t.to(device) for t in batch)
                with autocast():
                    prediction, hidden_output = model(input_ids, attention_masks, hidden_output)
                    loss = criterion(prediction.view(-1, num_labels), labels.view(-1))
                prediction_values, prediction_indices = torch.max(prediction, 2)
                y_prediction_at_step = []
                y_target_at_step = []
                for p, t in zip(prediction_indices, labels):
                    plist = p.tolist()
                    tlist = t.tolist()
                    plist = plist[1:] # Remove CLS token.
                    tlist = tlist[1:] # Remove CLS token.
                    tlist = [a for a in tlist if a >= 0] # Remove special tokens.
                    plist = plist[0:len(tlist)] # Remove special tokens.

                    # Y for whole gene.
                    y_prediction = np.concatenate((y_prediction, plist))
                    y_target = np.concatenate((y_target, tlist))

                    # Y for this step.
                    y_prediction_at_step = np.concatenate((y_prediction_at_step, plist))
                    y_target_at_step = np.concatenate((y_target_at_step, tlist))

                # metric at step
                metric_at_step = Metrics(y_prediction_at_step, y_target_at_step)
                metric_at_step.calculate()
                for k in sequential_label_indices:
                    token_label = Index_Dictionary[k]
                    wandb.log({f"precision-{token_label}": metric_at_step.precision(k, True)})
                    wandb.log({f"recall-{token_label}": metric_at_step.recall(k, True)})
                acc_at_step, error_rate_at_step = metric_at_step.accuracy_and_error_rate()
                training_log.write(f"{epoch},{step},{loss.item()},{acc_at_step},{error_rate_at_step}")
                wandb.log({"loss": loss.item()})
                wandb.log({"accuracy": acc_at_step})
                wandb.log({"error_rate": error_rate_at_step})
                loss.backward()
                optimizer.step()
                if not accumulate_gradient:
                    optimizer.zero_grad() # Reset gradient if not accumulate gradient.
            optimizer.zero_grad() # Automatically reset gradient if a gene has finished.

            # metric at gene
            # metric_at_gene = Metrics(y_prediction, y_target)
            # metric_at_gene.calculate()
            # gene_acc, gene_error_rate = metric_at_gene.accuracy_and_error_rate()
            
            # for k in sequential_label_indices:
            #     token_label = Index_Dictionary[k]
            #     wandb.log({
            #         f"training-{chr_name}-{gene_name}/precision-{token_label}": metric_at_step.precision(k, True),
            #         f"training-{chr_name}-{gene_name}/recall-{token_label}": metric_at_step.recall(k, True),
            #         "epoch":epoch
            #     })
            # wandb.log({
            #     f"training-{chr_name}-{gene_name}/accuracy": gene_acc, 
            #     f"training-{chr_name}-{gene_name}/error_rate": gene_error_rate,
            #     "epoch": epoch
            # })

        scheduler.step()
        
        # validation.
        model.eval()
        validation_log_path = os.path.join(save_dir, f"validation_log.{epoch}.csv")
        if os.path.exists(validation_log_path):
            os.remove(validation_log_path)
        validation_log = open(validation_log_path, "x")
        validation_log.write("epoch,step,chr,gene,sequence,prediction,target,accuracy,loss\n")
        # for validation_gene_file in validation_genes:
        for validation_gene_file in tqdm(validation_genes, total=num_validation_genes, desc=f"Validating at Epoch {epoch + 1}/{num_epochs}"):
            dataloader = preprocessing_whole_sequence(validation_gene_file, tokenizer, batch_size, dense=(preprocessing_mode == "dense"))
            chr_name = os.path.basename(os.path.dirname(validation_gene_file))
            gene_name = '.'.join(os.path.basename(validation_gene_file).split('.')[:-1])
            
            # TODO: detailed metrics are too much, better get an overview.
            for k in sequential_labels:
                wandb.define_metric(f"validation-{chr_name}-{gene_name}/precision-{k}", step_metric="epoch")
                wandb.define_metric(f"validation-{chr_name}-{gene_name}/recall-{k}", step_metric="epoch")
            wandb.define_metric(f"validation-{chr_name}-{gene_name}/accuracy", step_metric="epoch")
            wandb.define_metric(f"validation-{chr_name}-{gene_name}/error_rate", step_metric="epoch")
            hidden_output = None
            y_target = []
            y_pred = []
            for step, batch in enumerate(dataloader):
                input_ids, attention_masks, token_type_ids, labels = tuple(t.to(device) for t in batch)
                with no_grad():
                    prediction, hidden_output = model(input_ids, attention_masks, hidden_output)
                values, indices = torch.max(prediction, 2)
                for i, j, k in zip(input_ids, indices, labels):
                    ilist = i.tolist()
                    ilist_str = ' '.join([str(a) for a in ilist])
                    jlist = j.tolist()
                    jlist_str = ' '.join([str(a) for a in jlist])
                    klist = k.tolist()
                    klist_str = ' '.join([str(a) for a in klist])
                    accuracy, error_rate = accuracy_and_error_rate(i, j, k)
                    validation_log.write(f"{epoch},{step},{chr_name},{gene_name},{ilist_str},{klist_str},{jlist_str},{accuracy},{error_rate}\n")
                    klist = klist[1:] # Remove CLS token.
                    jlist = jlist[1:] # Remove CLS token.
                    klist = [e for e in klist if e >= 0] # Remove other special tokens.
                    jlist = jlist[0:len(klist)] # Remove other special tokens.
                    
                    y_target = np.concatenate((y_target, klist))
                    y_pred = np.concatenate((y_pred, jlist))

            # metrics.
            metrics = Metrics(y_pred, y_target)
            metrics.calculate()
            # print(f"label counts {metrics.get_label_counts()}")
            for e in sequential_label_indices:
                token_label = Index_Dictionary[e]
                precision = metrics.precision(e, True)
                recall = metrics.recall(e, True)
                wandb.log({
                    f"validation-{chr_name}-{gene_name}/precision-{token_label}": precision,
                    f"validation-{chr_name}-{gene_name}/recall-{token_label}": recall,
                    "epoch": epoch
                })
            wandb.log({
                f"validation-{chr_name}-{gene_name}/accuracy": accuracy,
                f"validation-{chr_name}-{gene_name}/error_rate": error_rate,
                "epoch": epoch
            })

        validation_log.close()
        checkpoint_path = os.path.join(save_dir, "latest", "checkpoint.pth")
        save_path = os.path.join(save_dir, f"checkpoint-{epoch}.pth")
        checkpoint = {
            "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, save_path)
        wandb.save(checkpoint_path)
        wandb.save(save_path)
        wandb.save(validation_log_path)


    training_log.close()

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
    device_names = "None" if device_list == [] else ', '.join([torch.cuda.get_device_name(k) for k in device_list])
    use_weighted_loss = args.get("use-weighted-loss", False)
    run_name = args.get("run-name", "genlab")
    resume_run_ids = args.get("resume-run-ids", [None for a in model_config_names])
    project_name = args.get("project-name", "whole-gene-labelling")
    accumulate_gradient = args.get("accumulate-gradient", False)
    preprocessing_mode = args.get("preprocessing-mode", "sparse") # Either dense or sparse.

    gene_dirpath = str(Path(PureWindowsPath(gene_dir)))
    training_index_path = str(Path(PureWindowsPath(training_index)))
    validation_index_path = str(Path(PureWindowsPath(validation_index)))
    test_index_path = str(Path(PureWindowsPath(test_index)))

    pretrained = str(Path(PureWindowsPath(training_config.get("pretrained", os.path.join("pretrained", "3-new-12w-0")))))
    bert_for_masked_lm = BertForMaskedLM.from_pretrained(pretrained)
    bert = bert_for_masked_lm.bert
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    os.environ["WANDB_MODE"] = "offline" if args.get("offline", False) else "online"
    n_train_data = pd.read_csv(training_index_path).shape[0]
    n_validation_data = pd.read_csv(validation_index_path).shape[0]
    cur_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"~~~~~Training Whole Gene Sequential Labelling~~~~~")
    print(f"# Training Data {n_train_data}")
    print(f"# Validation Data {n_validation_data}")
    print(f"Device {torch.cuda.get_device_name(device)}")
    print(f"Device List {device_names}")
    print(f"Project Name {project_name}")
    print(f"Model Configs {model_config_names}")
    print(f"Epochs {num_epochs}")
    print(f"Use weighted loss {use_weighted_loss}")
    print(f"Accumulate Gradients {accumulate_gradient}")
    print(f"Preprocessing Mode {preprocessing_mode}")

    for config_name, resume_run_id in zip(model_config_names, resume_run_ids):
        config_path = os.path.join(model_config_dir, f"{config_name}.json",)
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
        run = wandb.init(id=run_id, project=project_name, resume='allow', reinit=True, name=runname, config={
            "device": device,
            "device_list": device_names,
            "model_config": config_name,
            "training_data": n_train_data,
            "validation_data": n_validation_data,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "training_date": cur_date
        })
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

        train(model, optimizer, scheduler, gene_dirpath, training_index_path, validation_index_path, tokenizer, save_dir, wandb, num_epochs, start_epoch, batch_size, use_weighted_loss, accumulate_gradient, preprocessing_mode, device_list=device_list)
        run.finish()


