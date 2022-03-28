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
from utils.utils import save_model_state_dict, load_checkpoint, save_checkpoint
from data_preparation import str_kmer
from models.seqlab import DNABERTSeqLab
from utils.seqlab import _create_dataloader
from datetime import datetime

def __train__(model, batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels, loss_function, loss_strategy="sum"):
    prediction = model(batch_input_ids, batch_attn_mask, batch_token_type_ids)

    # Since loss function can only works without batch dimension, I need to loop the loss for each tokens in batch dimension.
    batch_loss = 0
    for p, l in zip(prediction, batch_labels):
        loss = loss_function(p, l)
        batch_loss += loss
    if loss_strategy == "average":
        batch_loss = batch_loss/batch_input_ids.shape[0]
    return batch_loss

def train(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', remove_old_model=False, training_counter=0, resume_from_checkpoint=None, resume_from_optimizer=None, grad_accumulation_steps=1, loss_function=NLLLoss(), loss_strategy="sum"):
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
    print("=====BEGIN TRAINING=====")
    start_time = datetime.now()
    print(f"Start Time {start_time}")
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
        "grad_accumulation_steps": grad_accumulation_steps,
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
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch [{i + 1 + training_counter}/{epoch_size}]"):
            input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)    
            loss_batch = __train__(model, input_ids, attention_mask, input_type_ids, label, loss_function, loss_strategy)
            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{i+training_counter},{step},{loss_batch},{lr}\n")
            loss_batch = (loss_batch / grad_accumulation_steps)
            epoch_loss += loss_batch
            loss_batch.backward()

            if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        
        torch.cuda.empty_cache()
        
        # After an epoch, save model state.
        save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
        save_model_state_dict(optimizer, save_model_path, "optimizer-{}.pth".format(i+training_counter))
        save_checkpoint(model, optimizer, {
            "loss": epoch_loss.item(),
            "epoch": i + training_counter,
            "grad_accumulation_steps": grad_accumulation_steps,
            "device": device,
            "batch_size": batch_size,
            "training_config": training_config,
        }, os.path.join(save_model_path, f"checkpoint-{i + training_counter}.pth"), replace=remove_old_model)
        #if remove_old_model:
        #    if i + training_counter > 0:
        #        old_model_path = os.path.join(save_model_path, os.path.basename("checkpoint-{}.pth".format(i + training_counter-1)))
        #        os.remove(old_model_path)

    #endfor epoch
    log_file.close()
    end_time = datetime.now()
    print(f"Finished Time {end_time}")
    print(f"Training Time {end_time - start_time}")
    print("=====END TRAINING=====")
    return model

def __eval__(model, input_ids, attention_mask, input_type_ids, label, device="cpu"):
    correct_token_pred, incorrect_token_pred = 0, 0
    model.to(device)
    model.eval()
    with torch.not_grad():
        pred = model(input_ids, attention_mask, input_type_ids)
        for p, z in zip(pred, label):
            p_score, p_index = torch.max(p, 1)
            if p_index == z:
                correct_token_pred += 1
            else:
                incorrect_token_pred += 1

    return correct_token_pred, incorrect_token_pred

def do_evaluate(model: DNABERTSeqLab, validation_dataloader: DataLoader, log=None, batch_size=1, device='cpu'):
    # TODO:
    # Implements how model evaluate model.
    model.to(device)
    model.eval()

    # Enable logging if log exists.
    log_file = {}
    if log:
        if os.path.exists(log):
            os.remove(log)
        os.makedirs(os.path.dirname(log))
        log_file = open(log, 'x')
        log_file.write(f"step,scores,correct,incorrect\n")

    correct_scores = []
    for step, batch in tqdm(enumerate(validation_dataloader, total=len(validation_dataloader))):
        input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)
        correct_token_pred, incorrect_token_pred = __eval__(model, input_ids, attention_mask, input_type_ids, label)
        average_score = correct_token_pred / input_type_ids.shape[1]
        correct_scores.append(average_score)
        if log_file != {}:
            log_file.write(f"{step},{average_score},{correct_token_pred},{incorrect_token_pred}\n")

    if log_file != {}:
        log_file.close()
    return True

def evaluate(model, validation_csv, device="cpu", batch_size=1, log=None):
    from utils.seqlab import preprocessing
    from utils.utils import get_default_tokenizer
    dataloader = preprocessing(validation_csv, get_default_tokenizer(), batch_size=batch_size)
    if not do_evaluate(model, dataloader, device=device, log=log):
        print("Evaluation failed.")


# def train_using_gene(model, tokenizer, optimizer, scheduler, num_epoch, batch_size, train_genes, loss_function, grad_accumulation_step="1", device="cpu"):
def train_using_genes(model, tokenizer, optimizer, scheduler, train_genes, loss_function, num_epoch=1, batch_size=1, grad_accumulation_steps="1", device="cpu", resume_checkpoint=None, save_path=None):
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
            gene_dataloader = _create_dataloader(gene, batch_size, tokenizer) # Create dataloader for this gene.
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

            if i % grad_accumulation_steps == 0 or (i + 1) == num_training_genes:
                optimizer.step()
                scheduler.step()

        #endfor
        save_checkpoint(model, optimizer, {
            'epoch': epoch + 1 + resume_training_counter,
            'loss': epoch_loss
        }, save_path)
    #endfor

    return model