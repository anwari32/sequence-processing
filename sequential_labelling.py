from concurrent.futures import process
from genericpath import exists
from msilib import sequence
import torch
from torch import tensor
from torch.nn import NLLLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertForMaskedLM, get_linear_schedule_with_warmup
import os
import pandas as pd
from tqdm import tqdm
import json
from utils.utils import save_model_state_dict, load_model_state_dict, load_checkpoint, save_checkpoint
from data_preparation import str_kmer

def train_iter(args):
    for epoch in args.num_epoch:
        model = train(args.model, args.optimizer, args.scheduler, args.batch_size, args.log)

def train_and_eval(model, train_dataloader, valid_dataloader, device="cpu"):
    model.to(device)
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
        output = model(input_ids, attn_mask)
    return model

def train(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', remove_old_model=False, training_counter=0, resume_from_checkpoint=None, resume_from_optimizer=None, grad_accumulation_step=1, loss_function=NLLLoss(), loss_strategy="sum"):
    """
    @param  model: BERT derivatives.

    @param  optimizer: optimizer
    @param  scheduler:
    @param  train_dataloader:
    @param  validation_dataloader:
    @param  epoch_size:
    @param  batch_size:
    @param  log_path (string):
    @param  save_model_path (string): where to save model for each epoch.
    @param  device (string) | None -> 'cpu': Default value is 'cpu', can be changed into 'cuda', 'cuda:0' for first cuda-compatible device, 'cuda:1' for second device, etc.
    @return model after training.
    """
    # Make directories if directories does not exist.
    if not os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # If log file exists, quit training.
    if os.path.exists(log_path):
        os.remove(log_path)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Save training configuration.
    training_config = {
        "num_epochs": epoch_size,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_step,
        "log": log_path,
        "save_model_path": save_model_path,
        "device": device,
        "training_counter": training_counter,
    }
    config_save_path = os.path.join(save_model_path, "config.json")
    if os.path.exists(config_save_path):
        os.remove(config_save_path)
    if not os.path.exists(os.path.dirname(config_save_path)):
        os.makedirs(os.path.dirname(config_save_path))
    config_file = open(config_save_path, 'x')
    json.dump(training_config, config_file, indent=4)
    config_file.close()

    # Writing training log.
    log_file = open(log_path, 'x')
    log_file.write('epoch,step,loss,learning_rate\n')

    # Do training.
    model.to(device)
    model.train()
    for i in range(epoch_size):
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)
            pred = model(input_ids, attention_mask, input_type_ids)
            loss_batch = None
            for p, t in zip(pred, label):
                loss = loss_function(p, t)
                if loss_batch == None:
                    loss_batch = loss
                else:
                    loss_batch += loss
            if loss_strategy == "average":
                loss_batch = loss_batch / batch_size
            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{i+training_counter},{step},{loss_batch},{lr}\n")
            epoch_loss += loss_batch
            loss_batch.backward()

            if (step + 1) % grad_accumulation_step == 0 or  (step + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            torch.cuda.empty_cache()
        #endfor batch

        # After an epoch, save model state.
        save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
        save_model_state_dict(optimizer, save_model_path, "optimizer-{}.pth".format(i+training_counter))
        save_checkpoint(model, optimizer, {
            "loss": epoch_loss.item(),
            "epoch": i + training_counter,
            "grad_accumulation_steps": grad_accumulation_step,
            "device": device,
            "batch_size": batch_size
        }, os.path.join(save_model_path, f"checkpoint-{i + training_counter}.pth"), replace=remove_old_model)
        #if remove_old_model:
        #    if i + training_counter > 0:
        #        old_model_path = os.path.join(save_model_path, os.path.basename("checkpoint-{}.pth".format(i + training_counter-1)))
        #        os.remove(old_model_path)

    #endfor epoch
    log_file.close()
    return model

def evaluate(model: DNABERTSeq2Seq, validation_dataloader: DataLoader, log=None, batch_size=1, device='cpu'):
    # TODO:
    # Implements how model evaluate model.
    model.to(device)
    model.eval()
    correct_pred = 0
    incorrect_pred = 1
    for step, batch in tqdm(enumerate(validation_dataloader, total=len(validation_dataloader))):
        input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)
        pred = model(input_ids, attention_mask, input_type_ids)
        for p, z in zip(pred, label):
            p_score, p_index = torch.max(p, 1)
            if p_index == z:
                correct_pred += 1
            else:
                incorrect_pred += 1

        
        
        

def convert_pred_to_label(pred, label_dict=Label_Dictionary):
    """
    @param      pred: tensor (<seq_length>, <dim>)
    @param      label_dict: 
    @return     array: []
    """
    return []


# def train_using_gene(model, tokenizer, optimizer, scheduler, num_epoch, batch_size, train_genes, loss_function, grad_accumulation_step="1", device="cpu"):
def train_using_genes(model, tokenizer, optimizer, scheduler, train_genes, loss_function, num_epoch=1, batch_size=1, grad_accumulation_step="1", device="cpu", resume_checkpoint=None, save_path=None):
    """
    @param  model
    @param  tokenizer
    @param  optimizer
    @param  scheduler
    @param  num_epoch (int | None -> 1)
    @param  batch_size (int | None -> 1)
    @param  train_genes (list<string>) : list of gene file path.
    @param  loss_function
    @param  grad_accumulation_step (int | None -> 1)
    @param  device (str | None -> ``cpu``)
    @return ``model``
    """
    resume_training_counter = 0
    if resume_checkpoint != None:
        model, optimizer, config = load_checkpoint(resume_checkpoint, model, optimizer)
        resume_training_counter = config["epoch"]
    
    num_training_genes = len(train_genes)
    for epoch in range(num_epoch):
        epoch_loss = None
        for i in range(num_training_genes):
            
            gene = train_genes[i]
            gene_dataloader = create_dataloader(gene, batch_size, tokenizer) # Create dataloader for this gene.
            gene_loss = None # This is loss computed from single gene.
            len_dataloader = len(gene_dataloader)
            total_training_instance = len_dataloader * batch_size # How many small sequences are in training.
            for step, batch in tqdm(enumerate(gene_dataloader), total=len_dataloader):
                input_ids, attn_mask, token_type_ids, label = tuple(t.to(device) for t in batch)

                pred = model(input_ids, attn_mask, token_type_ids)
                batch_loss = loss_function(pred, label)
                gene_loss = batch_loss if gene_loss == None else gene_loss + batch_loss
            #endfor

            avg_gene_loss = gene_loss / total_training_instance
            epoch_loss = avg_gene_loss if epoch_loss == None else epoch_loss + avg_gene_loss
            avg_gene_loss.backward()

            if i % grad_accumulation_step == 0 or (i + 1) == num_training_genes:
                optimizer.step()
                scheduler.step()

        #endfor
        save_checkpoint(model, optimizer, {
            'epoch': epoch + 1 + resume_training_counter,
            'loss': epoch_loss
        }, save_path)
    #endfor

    return model