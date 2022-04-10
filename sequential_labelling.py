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
from data_preparation import str_kmer
from models.seqlab import DNABERTSeqLab
from datetime import datetime
import wandb
from utils.seqlab import preprocessing, convert_ids_to_tokens
from utils.utils import get_default_tokenizer


def __train__(model, batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_labels, loss_function, device, loss_strategy="sum"):
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

def __eval__(model, input_ids, attention_mask, input_type_ids, label, device):
    # Make sure model and data are in the same device.
    model.to(device)
    input_ids.to(device)
    attention_mask.to(device)
    input_type_ids.to(device)
    label.to(device)

    correct_token_pred, incorrect_token_pred = 0, 0
    model.eval()
    pred_labels = []
    actual_labels = []
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
                actual_labels.append(zi.item())

    return correct_token_pred, incorrect_token_pred, pred_labels, actual_labels

def evaluate_genes(model, eval_genes, device="cpu"):
    model.eval()
    for gene in eval_genes:
        dataloader = preprocessing(gene, get_default_tokenizer())
        gene_chunk_accuracy, gene_chunk_error, token_preds, tokens = do_evaluate_gene(model, dataloader, device)

    return None

def do_evaluate_gene(model, dataloader, device):
    """
    Evaluate single gene.
    @param  model
    @param  dataloader
    @param  device
    """
    gene_accuracy_score = 0
    label_preds = []
    labels = []
    gene_chunk_accuracy = []
    gene_chunk_error = []
    for step, batch in dataloader: # Evaluating each chunk.
        input_ids, attn_mask, token_type_ids, label = tuple(t.to(device) for t in batch)
        correct_token_pred, incorrect_token_pred, pred_labels, target_labels = __eval__(model, input_ids, attn_mask, token_type_ids, label, device)
        label_preds.append(pred_labels)
        labels.append(target_labels)
        gene_chunk_accuracy.append(correct_token_pred / len(pred_labels))
        gene_chunk_error.append(incorrect_token_pred / len(pred_labels))

    # Since there is two additional tokens for BERT processing, we remove those two tokens leaving 510 token only.
    label_preds = [p[1:511] for p in label_preds]
    labels = [p[1:511] for p in labels]

    tokens_preds = [convert_ids_to_tokens(p) for p in label_preds]
    tokens = [convert_ids_to_tokens(p) for p in labels]

    return gene_chunk_accuracy, gene_chunk_error, tokens_preds, tokens


def train(model, optimizer, scheduler, train_dataloader, epoch_size, batch_size, log_path, save_model_path, device='cpu', training_counter=0, grad_accumulation_steps=1, loss_function=NLLLoss(), loss_strategy="sum", wandb=None):
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
            loss_batch = __train__(model, input_ids, attention_mask, input_type_ids, label, loss_function, loss_strategy)
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
        os.makedirs(os.path.dirname(log), exist_ok=True)
        log_file = open(log, 'x')
        log_file.write(f"step,scores,correct,incorrect,pred_labels,actual_labels\n")

    correct_scores = []
    for step, batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader)):
        input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)
        correct_token_pred, incorrect_token_pred, pred_labels, actual_labels = __eval__(model, input_ids, attention_mask, input_type_ids, label, device)
        average_score = correct_token_pred / input_type_ids.shape[1]
        correct_scores.append(average_score)
        if log_file != {}:
            log_file.write(f"{step},{average_score},{correct_token_pred},{incorrect_token_pred},{' '.join(str(v) for v in pred_labels)},{' '.join(str(v) for v in actual_labels)}\n")

    if log_file != {}:
        log_file.close()
    return sum(correct_scores)/len(correct_scores)

def evaluate(model, validation_csv, device="cpu", batch_size=1, log=None):
    dataloader = preprocessing(validation_csv, get_default_tokenizer(), batch_size=batch_size)
    return do_evaluate(model, dataloader, device=device, log=log)
        

# def train_using_gene(model, tokenizer, optimizer, scheduler, num_epoch, batch_size, train_genes, loss_function, grad_accumulation_step="1", device="cpu"):
def train_using_genes(model: DNABERTSeqLab, tokenizer: BertTokenizer, optimizer, scheduler, train_genes: list, loss_function, num_epoch=1, batch_size=1, grad_accumulation_steps=1, device="cpu", save_path=None, log_file_path=None, training_counter=0, wandb=None, eval_genes=None):
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

    # Initialize log.
    logfile = open(log_file_path, "x")
    logfile.write("epoch,gene,step,batch_loss,gene_loss,epoch_loss\n")

    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    import torch.profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1
        ), 
        on_trace_ready=trace_handler, 
        with_stack=True) as profiler:
            num_training_genes = len(train_genes)
            for epoch in range(num_epoch):
                epoch_loss = None
                for i in range(num_training_genes):
                    
                    gene = train_genes[i]
                    gene_dataloader = preprocessing(gene, tokenizer, batch_size, do_kmer=True)
                    gene_loss = None # This is loss computed from single gene.
                    len_dataloader = len(gene_dataloader)
                    total_training_instance = len_dataloader * batch_size # How many small sequences are in training.
                    # for step, batch in tqdm(enumerate(gene_dataloader), total=len_dataloader, desc=f"Epoch {epoch + 1}/{num_epoch} Gene {i+1}/{num_training_genes} {gene}"):
                    for step, batch in enumerate(gene_dataloader):
                        print(f"Training Epoch {epoch} {os.path.basename(gene)} {step + 1}/{len_dataloader} {30 * ' '}", end="\r")

                        input_ids, attn_mask, token_type_ids, label = tuple(t.to(device) for t in batch)
                        batch_loss = __train__(model, input_ids, attn_mask, token_type_ids, label, loss_function, device)
                        gene_loss = batch_loss if gene_loss == None else gene_loss + batch_loss

                        logfile.write(f"{epoch},{os.path.basename(gene)},{step},{batch_loss},{gene_loss},{epoch_loss}\n")

                        if wandb != None:
                            wandb.log({"batch_loss": batch_loss})
                            wandb.log({"gene_loss": gene_loss})

                            # Optional
                            wandb.watch(model)

                        torch.cuda.empty_cache()
                    #endfor

                    gene_loss = gene_loss / total_training_instance
                    epoch_loss = gene_loss if epoch_loss == None else epoch_loss + gene_loss
                    gene_loss.backward()

                    if i % grad_accumulation_steps == 0 or (i + 1) == num_training_genes:
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                #endfor

                # Eval model if eval_genes is available.
                if eval_genes:
                    eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
                    eval_genes()

                # Save trained model after this epoch is finished.
                save_checkpoint(model, optimizer, {
                    'epoch': epoch + training_counter,
                    'loss': epoch_loss
                }, os.path.join(save_path, f"checkpoint-{epoch}.pth"))
                torch.cuda.empty_cache()
            #endfor
    logfile.close()
    return model