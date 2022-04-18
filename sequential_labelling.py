from sklearn.metrics import accuracy_score
import torch
from torch import tensor
from torch.nn import NLLLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup
import os
import pandas as pd
from tqdm import tqdm
import json
from utils.utils import save_model_state_dict, load_checkpoint, save_checkpoint
from data_preparation import merge_kmer
from models.seqlab import DNABERTSeqLab
from datetime import datetime
import wandb
from utils.seqlab import preprocessing_kmer, convert_ids_to_tokens
from utils.tokenizer import get_default_tokenizer


def __forward_sequence__(model, batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels, loss_function, device, loss_strategy="sum"):
    # Make sure model and data are in the same device.
    model.to(device)
    batch_input_ids.to(device)
    batch_attn_mask.to(device)
    batch_token_type_ids.to(device)
    batch_labels.to(device)

    prediction = model(batch_input_ids, batch_attn_mask, batch_token_type_ids)

    # Since loss function can only works without batch dimension, I need to loop the loss for each tokens in batch dimension.
    batch_loss = None
    for p, l in zip(prediction, batch_labels):
        loss = loss_function(p, l)
        if batch_loss == None:
            batch_loss = loss
        else:
            batch_loss += loss
    if loss_strategy == "average":
        batch_loss = batch_loss/batch_input_ids.shape[0]
    return batch_loss

def __forward_gene_non_overlap__(model, dataloader: DataLoader, device: str, loss_function=None):
    """
    This function utilizes non-overlapping sequence.
    """

    # Make sure model and data are in the same device.
    model.train()
    model.to(device)
    contig_predicted_labels = []
    contig_target_labels = []
    for step, batch in enumerate(dataloader):
        input_ids, attn_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
        prediction = model(input_ids, attn_mask, token_type_ids)
        for pred, label in zip(prediction, labels): # Iterate through batch dimension.
            contig_predicted_labels.append(pred)
            contig_target_labels.append(label)
        # contig_predicted_labels, contig_target_labels = __forward_gene__(model, batch_input_ids, batch_attn_mask, batch_input_ids, batch_labels)
    #endfor

    # ``contig_predicted_labels`` is array of tensors (512, dim), first token and last token are special token hence they need to be removed.
    contig_predicted_labels = [t[1:511] for t in contig_predicted_labels] # Convert each tensor(510, dim) into array of 510 element.
    # ``contig_target_labels`` is array of tensors (512), first token and last token are special token hence they need to be removed.
    contig_target_labels = [t[1:511] for t in contig_target_labels] # Each element in ``contig_target_labels`` is a tensor with 510 element.
    
    # print(contig_predicted_labels, contig_predicted_labels[0].shape)
    # print(contig_target_labels, contig_target_labels[0].shape)

    # We need to merge contigs in ``contig_predicted_labels`` into single contig. First we convert those tensor-label sequence into label token.
    # and also merge target label in ``contig_target_labels`` into single contig.
    predicted_assembly = contig_predicted_labels[0]
    target_assembly = contig_target_labels[0]
    for pred, target in zip(contig_predicted_labels[1:], contig_target_labels[1:]):
        predicted_assembly = torch.concat((predicted_assembly, pred), 0)
        target_assembly = torch.concat((target_assembly, target), 0)

    gene_loss = None
    if loss_function:
        gene_loss = loss_function(predicted_assembly, target_assembly)

    # print(predicted_assembly)
    # print(target_assembly)
    return gene_loss, predicted_assembly, target_assembly

def __eval_sequence__(model, input_ids, attention_mask, input_type_ids, label, device):
    """
    Evaluate model in a sequence represented as ``input_ids``, ``attention_mask``, and ``input_type_ids`` against ``label``.
    @param  model:
    @param  input_ids:
    @param  attention_mask:
    @param  input_type_ids:
    @param  label:
    @param  device:
    @return (correct_token_pred, incorrect_token_pred, pred_labels, target_labels): tuple
    """

    # Make sure model and data are in the same device.
    model.to(device)
    input_ids.to(device)
    attention_mask.to(device)
    input_type_ids.to(device)
    label.to(device)

    correct_token_pred, incorrect_token_pred = 0, 0
    model.eval()
    pred_labels = []
    target_labels = []
    with torch.no_grad():
        pred = model(input_ids, attention_mask, input_type_ids)
        for p, z in zip(pred, label): # Batch
            p_score, p_index = torch.max(p, 1)
            for pi, zi in zip(p_index, z):
                if pi.item() == zi.item():
                    correct_token_pred += 1
                else:
                    incorrect_token_pred += 1
                pred_labels.append(pi.item())
                target_labels.append(zi.item())

    return correct_token_pred, incorrect_token_pred, pred_labels, target_labels

def __eval_gene__(model, dataloader, device):
    model.to(device)
    model.eval()
    correct_label, incorrect_label = 0, 0
    predicted_label_token, target_label_token = [], []
    with torch.no_grad():
        gene_loss, predicted_label_tensor, target_label_tensor = __forward_gene_non_overlap__(model, dataloader, device)
        values, indices = torch.max(predicted_label_tensor, 1)
        for p, q in zip(indices, target_label_tensor):
            if p.item() == q.item():
                correct_label += 1
            else:
                incorrect_label += 1
        predicted_label_token = [p.item() for p in list(indices)]
        target_label_token = [p.item() for p in list(target_label_tensor)]

        predicted_label_token = convert_ids_to_tokens(predicted_label_token)
        target_label_token = convert_ids_to_tokens(target_label_token)
    
    accuracy_score = correct_label / (correct_label + incorrect_label) * 100
    incorrect_score = incorrect_label / (correct_label + incorrect_label) * 100

    return accuracy_score, incorrect_score, predicted_label_token, target_label_token

def evaluate_genes(model, eval_genes, device, eval_log, epoch):
    model.eval()
    eval_logfile = {}
    if not os.path.exists(eval_log):
        eval_logfile = open(eval_log, "x")
        eval_logfile.write(f"epoch,gene,accuracy,error,predicted_label,target_label\n")
    else:
        eval_logfile = open(eval_log, "a")
    
    for gene in eval_genes:
        dataloader = preprocessing_kmer(gene, get_default_tokenizer(), 1)
        accuracy_score, incorrect_score, predicted_label_token, target_label_token = __eval_gene__(model, dataloader, device)
        eval_logfile.write(f"{epoch},{os.path.basename(gene).split('.')[0]},{accuracy_score},{incorrect_score},{' '.join(predicted_label_token)},{' '.join(target_label_token)}\n")
    #endfor
    eval_logfile.close()
    return None

def train_by_sequences(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', training_counter=0, grad_accumulation_steps=1, loss_function=NLLLoss(), loss_strategy="sum", wandb=None):
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
            loss_batch = __forward_sequence__(model, input_ids, attention_mask, input_type_ids, label, loss_function, loss_strategy)
            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{i+training_counter},{step},{loss_batch},{lr}\n")
            loss_batch = (loss_batch / grad_accumulation_steps)
            epoch_loss += loss_batch

            if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                model.zero_grad()
                optimizer.step()
                scheduler.step()

            if wandb != None:
                wandb.log({"epoch_loss": epoch_loss})
                wandb.log({"batch_loss": loss_batch})

                # Optional
                wandb.watch(model)
        #torch.cuda.empty_cache()
        
        # After an epoch, save model state.
        save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
        save_model_state_dict(optimizer, save_model_path, "optimizer-{}.pth".format(i+training_counter))
        save_checkpoint(model, optimizer, {
            "epoch_loss": epoch_loss.item(),
            "epoch": i + training_counter,
        }, os.path.join(save_model_path, f"checkpoint-{i + training_counter}.pth"))

    #endfor epoch
    log_file.close()
    end_time = datetime.now()
    print(f"Finished Time {end_time}")
    print(f"Training Time {end_time - start_time}")
    print("=====END TRAINING=====")
    return model

def train_by_genes(model: DNABERTSeqLab, tokenizer: BertTokenizer, optimizer, scheduler, train_genes: list, loss_function, num_epoch=1, batch_size=1, grad_accumulation_steps=1, device="cpu", save_path=None, log_file_path=None, training_counter=0, wandb=None, eval_genes=None, device_list=[]):
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

    if wandb:
        wandb.watch(model)

    n_gpu = len(device_list)
    if n_gpu > 1:
        print(f"Enabling DataParallel")
        model = torch.nn.DataParallel(model, device_list)
    
    from torch.cuda.amp import autocast, GradScaler                
    scaler = GradScaler()
        

    # Initialize log.
    logfile = open(log_file_path, "x")
    logfile.write("epoch,gene,gene_loss,epoch_loss\n")

    num_training_genes = len(train_genes)
    for epoch in range(num_epoch):
        epoch_loss = None
        for i in range(num_training_genes):
            
            gene = train_genes[i]
            gene_dataloader = preprocessing_kmer(gene, tokenizer, batch_size)
            # gene_loss = None # This is loss computed from single gene.
            with autocast():
                gene_loss, predicted_label, target_label = __forward_gene_non_overlap__(model, gene_dataloader, device, loss_function=loss_function)
                
            epoch_loss = gene_loss if epoch_loss == None else epoch_loss + gene_loss
            
            # Write gene training log.
            logfile.write(f"{epoch},{os.path.basename(gene).split('.')[0]},{gene_loss.item()},{epoch_loss.item()}")

            # Record log in the cloud.
            if wandb:
                wandb.log({"gene_loss": gene_loss.item()})

            # gene_loss.backward()
            scaler.scale(gene_loss).backward()

            # Check of gradient must be cleared or not.
            if i % grad_accumulation_steps == 0 or (i + 1) == num_training_genes:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        #endfor

        # Record epoch loss.
        if wandb:
            wandb.log({"epoch_loss": epoch_loss.item()})

        # Eval model if eval_genes is available.
        if eval_genes:
            eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
            evaluate_genes(model, eval_genes, device, eval_log, epoch)

        # Save trained model after this epoch is finished.
        save_checkpoint(model, optimizer, {
            'epoch': epoch + training_counter,
            'loss': epoch_loss
        }, os.path.join(save_path, f"checkpoint-{epoch}.pth"))
        torch.cuda.empty_cache()
    #endfor
    logfile.close()
    return model