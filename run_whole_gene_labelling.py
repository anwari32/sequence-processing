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
import wandb

from getopt import getopt
from pathlib import Path, PureWindowsPath
from sched import scheduler
from utils.cli import parse_args
from models.genlab import DNABERT_RNN
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch import no_grad
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.seqlab import Label_Dictionary, preprocessing_kmer
from utils.utils import create_loss_weight
from utils.metrics import Metrics, accuracy_and_error_rate

def train(model, optimizer, scheduler, gene_dir, training_index_path, validation_index_path, tokenizer, save_dir, wandb, num_epochs=1, start_epoch = 0, batch_size=1, use_weighted_loss=False, accumulate_gradient=False, packed_preprocessing=False):
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
    training_log = open(training_log_path, "x") if not os.path.exists(training_log_path) else open(training_log_path, "w")
    training_log.write("epoch,step,loss\n")
    validation_log = open(validation_log_path, "x") if not os.path.exists(validation_log_path) else open(validation_log_path, "w")
    validation_log.write("epoch,step,sequence,prediction,target,accuracy,loss\n")

    num_labels = model.num_labels
    num_training_genes = len(training_genes)
    num_validation_genes = len(validation_genes)
    model.to(device)
    wandb.define_metric("epoch")
    # for epoch in tqdm(range(start_epoch, num_epochs), total=(num_epochs-start_epoch), desc="Training "):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # for training_gene_file in training_genes:
        for training_gene_file in tqdm(training_genes, total=num_training_genes, desc=f"Training at Epoch {epoch + 1}/{num_epochs}"):
            dataloader = preprocessing_kmer(training_gene_file, tokenizer, batch_size, disable_tqdm=True)
            loss_weight = None
            if use_weighted_loss:
                loss_weight = create_loss_weight(training_gene_file, ignorance_level=2)
                loss_weight = loss_weight.to(device)
            criterion = CrossEntropyLoss(weight=loss_weight)
            hidden_output = None
            for step, batch in enumerate(dataloader):
                input_ids, attention_masks, token_type_ids, labels = tuple(t.to(device) for t in batch)
                with autocast():
                    prediction, hidden_output = model(input_ids, attention_masks, hidden_output)
                    loss = criterion(prediction.view(-1, num_labels), labels.view(-1))
                training_log.write(f"{epoch},{step},{loss.item()}")
                wandb.log({"loss": loss.item()})
                loss.backward()
                optimizer.step()
                if not accumulate_gradient:
                    optimizer.zero_grad() # Reset gradient if not accumulate gradient.
            optimizer.zero_grad() # Automatically reset gradient if a gene has finished.
        scheduler.step()
        
        model.eval()
        sequential_labels = [k for k in Label_Dictionary.keys() if Label_Dictionary[k] >= 0]
        sequential_label_indices = [k for k in range(8)]
        # for validation_gene_file in validation_genes:
        for validation_gene_file in tqdm(validation_genes, total=num_validation_genes, desc=f"Validating at Epoch {epoch + 1}/{num_epochs}"):
            dataloader = preprocessing_kmer(validation_gene_file, tokenizer, batch_size, disable_tqdm=True)
            gene_name = '.'.join(os.path.basename(validation_gene_file).split('.')[:-1])
            for k in sequential_labels:
                wandb.define_metric(f"validation-{gene_name}/{k}", step_metric="epoch")
            wandb.define_metric(f"validation-{gene_name}/accuracy", step_metric="epoch")
            wandb.define_metric(f"validation-{gene_name}/error_rate", step_metric="epoch")
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
                    print(f"{epoch},{step},{ilist},{klist},{jlist},{accuracy},{error_rate}\n")
                    validation_log.write(f"{epoch},{step},{ilist},{klist},{jlist},{accuracy},{error_rate}\n")
                    klist = k[1:] # Remove CLS token.
                    jlist = j[1:] # Remove CLS token.
                    klist = [e for e in klist if e >= 0] # Remove other special tokens.
                    jlist = jlist[0:len(klist)] # Remove other special tokens.
                    print(klist)
                    print(jlist)
                    
                    # metrics.
                    metrics = Metrics(jlist, klist)
                    metrics.calculate()
                    for e in sequential_label_indices:
                        precision = metrics.precission(e, True)
                        wandb.log({
                            f"validation-{gene_name}/{Label_Dictionary[e]}": precision
                        })
                    wandb.log({
                        f"validation-{gene_name}/accuracy": accuracy,
                        f"validation-{gene_name}/error_rate": error_rate
                    })


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

    training_log.close()
    validation_log.close()

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
    project_name = args.get("project-name", "pilot-project")
    accumulate_gradient = args.get("accumulate-gradient", False)
    packed_preprocessing = args.get("packed-preprocessing", False)

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
    print(f"Packed Preprocessing {packed_preprocessing}")

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

        train(model, optimizer, scheduler, gene_dirpath, training_index_path, validation_index_path, tokenizer, save_dir, wandb, num_epochs, start_epoch, batch_size, use_weighted_loss, accumulate_gradient)
        run.finish()


