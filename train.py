from pickletools import optimize
from sched import scheduler
from torch import no_grad
from torch.optim import AdamW
from torch.nn import DataParallel, CrossEntropyLoss
from torch import autocast, cuda
from transformers import get_polynomial_decay_schedule_with_warmup
from utils.seqlab import preprocessing
from utils.tokenizer import get_default_tokenizer
from tqdm import tqdm
import torch
import os
from utils.utils import save_checkpoint

def train_by_sequence(model, optimizer, train_dataloader, num_epochs, device, device_list: list, batch_size, wandb, save_dir, eval_dataloader=None, **kwargs):
    # Log properties.
    if wandb != None:
        wandb.define_metric("train/epoch")
        wandb.define_metric("train/loss", step_metric="train/epoch")
        wandb.define_metric("validation/epoch")
        wandb.define_metric("validation/loss", step_metric="validation/epoch")
        wandb.define_metric("validation/accuracy", step_metric="validation/epoch")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    loss_fn = kwargs.get("loss_fn", CrossEntropyLoss())
    loss_strategy = kwargs.get("loss_strategy", "sum")

    model.to(device)
    if len(device_list) > 1:
        model = DataParallel(model, device_ids=device_list)
    
    scaler = cuda.amp.GradScaler()
    best_accuracy = 0
    for epoch in range(num_epochs):
        epoch_loss = 0

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training [{epoch + 1}/{num_epochs}]"):
            input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
            batch_loss = 0
            with autocast():
                predictions = model(input_ids, attention_mask)
                for pred, label in zip(predictions, batch_labels):
                    loss = loss_fn(pred, label)
                    batch_loss += loss
                
            batch_loss = batch_loss / predictions.shape[0] if loss_strategy == "average" else 1
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss
            
        if wandb != None:
            wandb.log({
                "training/loss": epoch_loss.item(),
                "training/epoch": epoch
            })
        
        # After an epoch, eval model is evaluation data is available.
        if eval_dataloader != None:
            model.eval()
            eval_accuracy = 0
            eval_loss = 0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"Evaluating [{epoch + 1}/{num_epochs}]"):
                    input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
                    predictions = model(input_ids, attention_mask)
                    batch_loss = 0
                    batch_accuracy = 0
                    for pred, label in zip(predictions, batch_labels):
                        loss = loss_fn(pred, label)
                        batch_loss += loss
                        accuracy = 0
                        scores, indices = torch.max(pred, 1)
                        for idx, l in zip(indices, label):
                            accuracy += 1 if idx == l else 0
                        accuracy = accuracy / len(indices)
                        batch_accuracy += accuracy

                    batch_loss = batch_loss / predictions.shape[0] if loss_strategy == "average" else 1
                    eval_loss += batch_loss
                    batch_accuracy = accuracy / predictions.shape[0]
                    eval_accuracy += accuracy
                
                eval_accuracy = eval_accuracy / len(eval_dataloader)
                eval_loss = eval_loss / len(eval_dataloader)
            
            if wandb != None:
                wandb.log({
                    "validation/accuracy": eval_accuracy,
                    "validation/loss": eval_loss.item(),
                    "validation/epoch": epoch
                })
            
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                _model = model
                if isinstance(model, DataParallel):
                    _model = model.module
                save_checkpoint(_model, optimizer, {
                    "epoch": epoch
                }, os.path.join(save_dir, f"checkpoint-{epoch}.pth"))

    return model, optimizer

def train_by_genes(model, train_genes, num_epochs, device, device_list: list, batch_size, wandb, save_dir, eval_genes=None, **kwargs):

    # Log properties.
    if wandb != None:
        wandb.define_metric("train/epoch")
        wandb.define_metric("train/loss", step_metric="train/epoch")
        wandb.define_metric("validation/epoch")
        wandb.define_metric("validation/loss", step_metric="validation/epoch")
        wandb.define_metric("validation/accuracy", step_metric="validation/epoch")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Basic optimizer parameters.
    optim = kwargs.get("optimizer", "adamw")
    learning_rate = kwargs.get("learning_rate", 1e-4)
    beta1 = kwargs.get("beta1", 0.98)
    beta2 = kwargs.get("beta2", 0.9)
    warmup = kwargs.get("warmup", 0)
    epsilon = kwargs.get("epsilon", 1e-6)
    weight_decay = kwargs.get("weight_decay", 0.01)

    # Basic training config.
    loss_fn = kwargs.get("loss_fn", CrossEntropyLoss())
    loss_strategy = kwargs.get("loss_strategy", "sum")

    # Init tokenizer
    tokenizer = get_default_tokenizer()

    model.to(device)
    if len(device_list) > 1:
        model = DataParallel(model, device_ids=device_list)

    best_accuracy = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        model.train()
        # @param `genes` is an array containing gene files in which gene sequences are stored.
        for gene_file in train_genes:
            train_dataloader = preprocessing(gene_file, tokenizer, batch_size, do_kmer=False)

            steps = len(train_dataloader) * num_epochs
        
            optimizer = None
            if optim == "adamw":
                optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)

            scheduler = None
            if optimizer != None:
                scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup, steps)

            scaler = cuda.amp.GradScaler()

            gene_loss = 0
            for step, batch in enumerate(train_dataloader):
                input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
                with autocast():
                    predictions = model(input_ids, attention_mask)
                    batch_loss = 0
                    for pred, label in zip(predictions, batch_labels):
                        loss = loss_fn(pred, label)
                        batch_loss += loss
                    
                    if loss_strategy == "average":
                        batch_loss = batch_loss / predictions.shape[0]
                    
                gene_loss += batch_loss
                
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            
            epoch_loss += gene_loss
        
        if eval_genes != None:
            model.eval()
            eval_accuracy = 0
            eval_loss = 0
            with no_grad():
                for gene_file in tqdm(eval_genes, total=len(eval_genes), desc=f"Evaluating [{epoch + 1}/{num_epochs}]"):
                    eval_dataloader = preprocessing(gene_file, tokenizer, 1, do_kmer=False)
                    gene_accuracy = 0
                    gene_loss = 0
                    for step, batch in enumerate(eval_dataloader):
                        input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
                        predictions = model(input_ids, attention_mask)
                        batch_loss = 0
                        batch_accuracy = 0
                        for pred, label in zip(predictions, batch_labels):
                            loss = loss_fn(pred, label)
                            batch_loss += loss
                            accuracy = 0
                            scores, indices = torch.max(pred, 1)
                            for p, l in zip(indices, label):
                                accuracy += 1 if p == l else 0
                            accuracy = accuracy / len(indices)
                            batch_accuracy += accuracy
                        
                        batch_accuracy = batch_accuracy / predictions.shape[0]
                        batch_loss = batch_loss / predictions.shape[0]

                    gene_accuracy = batch_accuracy 
                    gene_loss = batch_loss

                eval_accuracy = gene_accuracy / len(eval_genes) # Average accuracy over all genes.
                eval_loss = gene_loss / len(eval_genes) # Average loss over all genes.

            if eval_accuracy > best_accuracy:
                # Save best model.
                best_accuracy = eval_accuracy
                save_checkpoint(model, optimizer, {
                    "epoch": epoch,
                    "accuracy": eval_accuracy,
                    "loss": eval_loss.item()
                }, os.path.join(save_dir, f"checkpoint-{epoch}.pth"))
            
    return model, optimizer
