import json
import traceback
import torch
from torch.nn import DataParallel
from torch import nn
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import os
import sys

import wandb
from utils.utils import save_checkpoint, save_model_state_dict
from data_preparation import str_kmer
from pathlib import Path, PureWindowsPath

from models.mtl import MTModel, PolyAHead, PromoterHead, SpliceSiteHead

def init_model_mtl(pretrained_path, config: json, device="cpu"):
    bert = BertForMaskedLM.from_pretrained(pretrained_path).bert
    model = MTModel(bert, config)
    model.to(device)
    return model

def __train__(model: MTModel, input_ids, attention_mask, label_prom, label_ss, label_polya, loss_fn_prom=nn.BCELoss(), loss_fn_ss=nn.CrossEntropyLoss(), loss_fn_polya=nn.CrossEntropyLoss()):
    output = model(input_ids, attention_mask)
    pred_prom = output["prom"]
    pred_ss = output["ss"]
    pred_polya = output["polya"]

    loss_prom, loss_ss, loss_polya = 0, 0, 0
    num_prom_labels, num_ss_labels, num_polya_labels = 0, 0, 0
    _model = model
    if isinstance(model, DataParallel):
        _model = model.module
    if _model.promoter_layer.num_labels == 1:
        loss_prom = loss_fn_prom(pred_prom, label_prom.float().reshape(-1, 1))
    else:
        loss_prom = loss_fn_prom(pred_prom, label_prom)    
    
    if _model.splice_site_layer.num_labels == 1:
        loss_ss = loss_fn_ss(pred_ss, label_ss.float().reshape(-1, 1))
    else:
        loss_ss = loss_fn_ss(pred_ss, label_ss)

    if _model.polya_layer.num_labels == 1:
        loss_polya = loss_fn_polya(pred_polya, label_polya.float().reshape(-1, 1))
    else:
        loss_polya = loss_fn_polya(pred_polya, label_polya)
    
    return loss_prom, loss_ss, loss_polya

def __eval__(model: MTModel, input_ids, attention_mask, label_prom, label_ss, label_polya, device):
    model.to(device)
    input_ids.to(device)
    attention_mask.to(device)
    label_prom.to(device)
    label_ss.to(device)
    label_polya.to(device)

    with torch.no_grad():
        # Forward.
        output = model(input_ids, attention_mask)
        pred_prom = output['prom']
        pred_ss = output['ss']
        pred_polya = output['polya']

        _model = model
        if isinstance(model, DataParallel):
            _model = model.module

        # Prediction.
        predicted_prom, prom_eval = 0, 0
        if _model.promoter_layer.num_labels == 1:
            predicted_prom = torch.round(pred_prom).item()
            prom_eval = (predicted_prom == label_prom.item())
        else:
            prom_val, predicted_prom_index = torch.max(pred_ss, 1)
            predicted_prom = predicted_prom_index.item()
            prom_eval = (predicted_prom_index == label_prom.item())    

        predicted_ss, ss_eval = 0, 0
        if _model.splice_site_layer.num_labels == 1:
            predicted_ss = torch.round(pred_ss).item()
            ss_eval = (predicted_ss == label_ss.item())
        else:
            ss_val, predicted_ss_index = torch.max(pred_ss, 1)
            predicted_ss = predicted_ss_index.item()
            ss_eval = (predicted_ss_index == label_ss.item())

        predicted_polya, polya_eval = 0, 0
        if _model.polya_layer.num_labels == 1:
            predicted_polya = torch.round(pred_polya).item()
            polya_eval = (predicted_polya == label_polya.item())
        else:
            polya_val, predicted_polya_index = torch.max(pred_polya, 1)
            predicted_polya = predicted_polya_index.item()
            polya_eval = (predicted_polya_index == label_polya.item())
        
    return (prom_eval, predicted_prom, label_prom.item(), 
        ss_eval, predicted_ss, label_ss.item(), 
        polya_eval, predicted_polya, label_polya.item())

def evaluate(model, dataloader, log_path, device, cur_epoch, wandb: wandb=None):
    log_dir = os.path.dirname(log_path)
    log = {}
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_path):
        log = open(log_path, "x")
        log.write("epoch,step,prom_eval,prom_predict,prom_label,ss_eval,ss_predict,ss_label,polya_eval,polya_predict,polya_label\n")
    else:
        log = open(log_path, "a")
    model.eval()
    model.to(device)
    count_prom_correct = 0
    count_ss_correct = 0
    count_polya_correct = 0
    prom_accuracy = 0
    ss_accuracy = 0
    polya_accuracy = 0
    prom_evals = []
    prom_predicts = []
    prom_actuals = []
    ss_evals = []
    ss_predicts = []
    ss_actuals = []
    polya_evals = []
    polya_predicts = []
    polya_actuals = []
    len_dataloader = len(dataloader)
    for step, batch in tqdm(enumerate(dataloader), total=len_dataloader, desc=f"Evaluating"):
        input_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)

        prom_eval, prom_predict, prom_label, ss_eval, ss_predict, ss_label, polya_eval, polya_predict, polya_label = __eval__(model, input_ids, attn_mask, label_prom, label_ss, label_polya, device)
        prom_evals.append(prom_eval)
        prom_predicts.append(prom_predicts)
        prom_actuals.append(prom_label)
        ss_evals.append(ss_eval)
        ss_predicts.append(ss_predict)
        ss_actuals.append(ss_label)
        polya_evals.append(polya_eval)
        polya_predicts.append(polya_predict)
        polya_actuals.append(polya_label)

        log.write(f"{cur_epoch},{step},{1 if prom_eval else 0},{prom_predict},{prom_label},{1 if ss_eval else 0},{ss_predict},{ss_label},{1 if polya_eval else 0},{polya_predict},{polya_label}\n")
    #endfor
    log.close()
    # Compute average accuracy.
    count_prom_correct = len([p for p in prom_evals if p])
    count_ss_correct = len([p for p in ss_evals if p])
    count_polya_correct = len([p for p in polya_evals if p])
    prom_accuracy = count_prom_correct / len(prom_evals) * 100
    ss_accuracy = count_ss_correct / len(ss_evals) * 100
    polya_accuracy = count_polya_correct / len(polya_evals) * 100

    if wandb:
        wandb.log({"validation/prom_accuracy": prom_accuracy, "train/epoch": cur_epoch})
        wandb.log({"validation/ss_accuracy": ss_accuracy, "train/epoch": cur_epoch})
        wandb.log({"validation/polya_accuracy": polya_accuracy, "train/epoch": cur_epoch})

    return prom_accuracy, ss_accuracy, polya_accuracy

def train(dataloader: DataLoader, model: MTModel, loss_fn, optimizer, scheduler, batch_size: int, epoch_size: int, log_file_path: str, device='cpu', save_model_path=None, training_counter=0, loss_strategy="sum", grad_accumulation_steps=1, wandb=None, eval_dataloader=None, device_list=[]):
    """
    @param      dataloader:
    @param      model:
    @param      loss_fn:
    @param      optimizer:
    @param      scheduler:
    @param      batch_size:
    @param      epoch_size:
    @param      log_file_path:
    @param      device:
    @param      save_model_path (string | None = None): dir path to save model per epoch. Inside this dir will be generated a dir for each epoch. If this path is None then model will not be saved.
    @param      grad_accumulation_steps (int | None = 1): After how many step backward is computed.
    """
    log_file = {}    
    model.to(device)
    model.train()
    model.zero_grad()

    if save_model_path != None:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path, exist_ok=True)
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.mkdir(os.path.dirname(log_file_path))
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    _cols = ['epoch','batch','loss_prom','loss_ss','loss_polya','lr']
    log_file = open(log_file_path, 'x')
    log_file.write("{}\n".format(','.join(_cols)))
    start_time = datetime.now()
    len_dataloader = len(dataloader)
    try:
        # Last best accuracy.
        best_accuracy = 0 
        if wandb != None:
            wandb.define_metric("train/epoch")
            wandb.define_metric("train/*", step_metric="train/epoch")
            wandb.define_metric("validation/epoch")
            wandb.define_metric("validation/*", step_metric="train/epoch")
        
        n_gpu = len(device_list)
        if n_gpu > 1:
            print(f"Enabling DataParallel")
            model = torch.nn.DataParallel(model, device_list)
        
        from torch.cuda.amp import autocast, GradScaler                
        scaler = GradScaler()
        
        for i in range(epoch_size):
            epoch_start_time = datetime.now()
            model.train()
            model.zero_grad()
            epoch_loss = 0
            avg_prom_loss = 0
            avg_ss_loss = 0
            avg_polya_loss = 0
            for step, batch in tqdm(enumerate(dataloader), total=len_dataloader, desc="Training Epoch [{}/{}]".format(i + 1 + training_counter, epoch_size)):
                in_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
                with autocast():
                    loss_prom, loss_ss, loss_polya = __train__(model, in_ids, attn_mask, label_prom, label_ss, label_polya, loss_fn_prom=loss_fn["prom"], loss_fn_ss=loss_fn["ss"], loss_fn_polya=loss_fn["polya"])
                    
                    # Accumulate promoter, splice site, and poly-A loss.
                    avg_prom_loss += loss_prom
                    avg_ss_loss += loss_ss
                    avg_polya_loss += loss_polya

                    # Following MTDNN (Liu et. al., 2019), loss is summed.
                    loss = (loss_prom + loss_ss + loss_polya) / (3 if loss_strategy == "average" else 1)

                    # Log loss values and learning rate.
                    lr = optimizer.param_groups[0]['lr']
                    log_file.write("{},{},{},{},{},{}\n".format(i+training_counter, step, loss_prom.item(), loss_ss.item(), loss_polya.item(), lr))


                # Update parameters and learning rate for every batch.
                # Since this training is based on batch, then for every batch optimizer.step() and scheduler.step() are called.
                loss = loss / grad_accumulation_steps

                # Accumulate loss in this batch.
                epoch_loss += loss

                # Wandb.
                if wandb != None:
                    wandb.log({"loss": loss.item()})
                    wandb.log({"prom_loss": loss_prom.item()})
                    wandb.log({"ss_loss": loss_ss.item()})
                    wandb.log({"polya_loss": loss_polya.item()})
                    wandb.log({"learning_rate": lr})

                # Backpropagation.
                # loss.backward(retain_graph=True)                
                scaler.scale(loss).backward()

                if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    
                    # Clip gradient.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update learning rate and scheduler.
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Reset model gradients.
                    # model.zero_grad()
                    optimizer.zero_grad()
            # endfor batch.

            # Calculate average loss for promoter, splice site, and poly-A.
            # Log losses with wandb.
            avg_prom_loss = avg_prom_loss / len_dataloader
            avg_ss_loss = avg_ss_loss / len_dataloader
            avg_polya_loss = avg_polya_loss / len_dataloader
            avg_epoch_loss = epoch_loss / len_dataloader
            log_entry = {
                "train/prom_loss": avg_prom_loss.item(),
                "train/ss_loss": avg_ss_loss.item(),
                "train/polya_loss": avg_polya_loss.item(),
                "train/loss": avg_epoch_loss.item(),
                "train/epoch": i
            }
            wandb.log(log_entry)

            epoch_duration = datetime.now() - epoch_start_time
            # Log epoch loss. Epoch loss equals to average of epoch loss over steps.
            if wandb:
                wandb.define_metric("epoch/loss", step_metric="train/epoch")
                wandb.log({"epoch/loss": epoch_loss.item() / len_dataloader, "train/epoch": i})
                wandb.watch(model)

            # After an epoch, eval model if eval_dataloader is given.
            prom_accuracy, ss_accuracy, polya_accuracy = 0, 0, 0
            if eval_dataloader:
                eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
                prom_accuracy, ss_accuracy, polya_accuracy = evaluate(model, eval_dataloader, eval_log, device, i + training_counter, wandb=wandb)
                avg_accuracy = (prom_accuracy + ss_accuracy + prom_accuracy) / 3
                if wandb:
                    wandb.log({"validation/prom_accuracy": prom_accuracy, "train/epoch": i} )
                    wandb.log({"validation/ss_accuracy": ss_accuracy, "train/epoch": i})
                    wandb.log({"validation/polya_accuracy": polya_accuracy, "train/epoch": i})
                    wandb.log({"validation/avg_accuracy": avg_accuracy, "train/epoch": i})

            # Calculate epoch loss over len(dataloader)
            epoch_loss = epoch_loss / len(dataloader)

            # Save model with best validation score.
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy

                # Save checkpoint.
                save_checkpoint(model, optimizer, {
                    "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                    "epoch": (i + training_counter),
                    "batch_size": batch_size,
                    "device": device,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "prom_accuracy": prom_accuracy,
                    "ss_accuracy": ss_accuracy,
                    "polya_accuracy": polya_accuracy
                }, os.path.join(save_model_path, f"checkpoint-{i + training_counter}.pth"))

                # Remove previous model.
                old_model_path = os.path.join(save_model_path, os.path.basename(f"checkpoint-{i + training_counter - 1}.pth"))
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
        # endfor epoch.
    except ImportError: 
        raise ImportError("Error importing autocase or GradScaler")
    except Exception as e:
        log_file.close()
        print(traceback.format_exc())
        print(e)

    log_file.close()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Start Time: {start_time}, End Time: {end_time}, Training Duration {elapsed_time}")
    return model, optimizer

def get_sequences(csv_path: str, n_sample=10, random_state=1337):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
    @param      csv_path (string): path to csv file.
    @param      n_sample (int): how many instance are retrieved from CSV located in `csv_path`.
    @param      random_state (int): random seed for randomly retriving `n_sample` instances.
    @return     (list, list, list, list): sequence, label_prom, label_ss, label_polya.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File {} not found.".format(csv_path))
    df = pd.read_csv(csv_path)
    if (n_sample > 0):
        df = df.sample(n=n_sample, random_state=random_state)
    sequence = list(df['sequence'])
    sequence = [str_kmer(s, 3) for s in sequence]
    label_prom = list(df['label_prom'])
    label_ss = list(df['label_ss'])
    label_polya = list(df['label_polya'])

    return sequence, label_prom, label_ss, label_polya

def prepare_data(data, tokenizer: BertTokenizer):
    """
    Preprocessing for pretrained BERT.
    @param      data (array of string): array of string, each string contains kmers separated by spaces.
    @param      tokenizer (Tokenizer): tokenizer initialized from pretrained values.
    @return     input_ids, attention_masks (tuple of torch.Tensor): tensor of token ids to be fed to model,
                tensor of indices (a bunch of 'indexes') specifiying which token needs to be attended by model.
    """
    input_ids = []
    attention_masks = []
    _count = 0
    _len_data = len(data)
    for sequence in tqdm(data, total=_len_data, desc="Preparing Data"):
        """
        Sequence is 512 characters long.
        """
        _count += 1
        #if _count < _len_data:
        #    print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data), end='\r')
        #else:
        #    print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data))
        encoded_sent = tokenizer.encode_plus(
            text=sequence,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert input_ids and attention_masks to tensor.
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

def preprocessing(csv_file: str, pretrained_tokenizer_path: str, batch_size=2000, n_sample=0, random_state=1337):
    """
    @return dataloader (torch.utils.data.DataLoader)
    """
    csv_file = PureWindowsPath(csv_file)
    csv_file = str(Path(csv_file))
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
        sys.exit(2)
    _start_time = datetime.now()

    bert_path = PureWindowsPath(pretrained_tokenizer_path)
    bert_path = str(Path(bert_path))
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state)
    arr_input_ids, arr_attn_mask = prepare_data(sequences, tokenizer)
    prom_labels_tensor = tensor(prom_labels)
    ss_labels_tensor = tensor(ss_labels)
    polya_labels_tensor = tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Preparing Dataloader duration {}".format(_elapsed_time))
    return dataloader

