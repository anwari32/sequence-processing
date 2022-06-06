import traceback
import torch
from torch.nn import DataParallel
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import os
from torch.cuda.amp import autocast, GradScaler                

import wandb
from models.mtl import DNABERT_MTL
from utils.utils import save_checkpoint
from data_preparation import str_kmer
from pathlib import Path, PureWindowsPath

def forward(model: DNABERT_MTL, input_ids, attention_mask, label_prom, label_ss, label_polya, loss_fn_prom=nn.BCELoss(), loss_fn_ss=nn.CrossEntropyLoss(), loss_fn_polya=nn.CrossEntropyLoss()):
    output = model(input_ids, attention_mask)
    pred_prom = output["prom"] # Tensor
    pred_ss = output["ss"] # Tensor
    pred_polya = output["polya"] # Tensor

    prom_loss, ss_loss, polya_loss = 0, 0, 0
    _model = model
    if isinstance(model, DataParallel):
        _model = model.module

    if _model.promoter_layer.num_labels == 1:
        predicted_prom = torch.round(pred_prom).item() if len(pred_prom) == 1 else [torch.round(p).item() for p in pred_prom]
        pred_eval = (predicted_prom == label_prom.item()) if len(predicted_prom) == 1 else sum([1 for p, q in zip(predicted_prom, label_prom) if p == q])
        prom_loss = loss_fn_prom(pred_prom, label_prom.float().reshape(-1, 1)) # Tensor
    else:
        prom_val, predicted_prom_index = torch.max(pred_prom, 1)
        predicted_prom = predicted_prom_index.item() if len(predicted_prom_index) == 1 else [p.item() for p in predicted_prom_index]
        prom_eval = (predicted_prom == label_prom.item()) if len(label_prom) == 1 else sum([1 for p, q in zip(predicted_prom, label_prom) if p == q])
        prom_loss = loss_fn_prom(pred_prom, label_prom) # Tensor   
    
    predicted_ss, ss_eval, ss_loss = 0, 0, 0
    if _model.splice_site_layer.num_labels == 1:
        predicted_ss = torch.round(pred_ss).item() if len(pred_ss) == 1 else [torch.round(p).item() for p in pred_ss]
        ss_eval = (predicted_ss == label_ss.item()) if len(predicted_ss) == 1 else sum([1 for p, q in zip(predicted_ss, label_ss) if p == q])
        ss_loss = loss_fn_ss(pred_ss, label_ss.float().reshape(-1, 1)) # Tensor
    else:
        ss_val, predicted_ss_index = torch.max(pred_ss, 1)
        predicted_ss = predicted_ss_index.item() if len(predicted_ss_index) == 1 else [p.item() for p in predicted_ss_index]
        ss_eval = (predicted_ss == label_ss.item()) if len(label_ss) == 1 else sum([1 for p, q in zip(predicted_ss, label_ss) if p == q])
        ss_loss = loss_fn_ss(pred_ss, label_ss) # Tensor

    predicted_polya, polya_eval, polya_loss = 0, 0, 0
    if _model.polya_layer.num_labels == 1:
        predicted_polya = torch.round(pred_prom).item() if len(pred_polya) == 1 else [torch.round(p).item() for p in pred_polya]
        polya_eval = (predicted_polya == label_prom.item()) if len(predicted_polya) == 1 else sum([1 for p, q in zip(predicted_polya, label_polya) if p == q])
        polya_loss = loss_fn_polya(pred_polya, label_polya.float().reshape(-1, 1)) # Tensor
    else:
        polya_val, predicted_polya_index = torch.max(pred_polya, 1)
        predicted_polya = predicted_polya_index.item() if len(predicted_polya_index) == 1 else [p.item() for p in predicted_polya_index]
        polya_eval = (predicted_polya == label_polya.item()) if len(label_polya) == 1 else sum([1 for p, q in zip(predicted_polya, label_polya) if p == q])
        polya_loss = loss_fn_polya(pred_polya, label_polya) # Tensor
    
    # return prom_loss, ss_loss, polya_loss
    return (
        prom_eval, predicted_prom, label_prom, prom_loss,
        ss_eval, predicted_ss, label_ss, ss_loss,
        polya_eval, predicted_polya, label_polya, polya_loss
    )

def __eval__(model: DNABERT_MTL, input_ids, attention_mask, label_prom, label_ss, label_polya, device, loss_fn):
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
        predicted_prom, prom_eval, prom_loss = 0, 0, 0
        if _model.promoter_layer.num_labels == 1:
            predicted_prom = torch.round(pred_prom).item()
            prom_eval = (predicted_prom == label_prom.item())
            prom_loss = loss_fn["prom"](pred_prom, label_prom.float().reshape(-1, 1))
        else:
            prom_val, predicted_prom_index = torch.max(pred_prom, 1)
            predicted_prom = predicted_prom_index.item()
            prom_eval = (predicted_prom_index == label_prom.item())
            prom_loss = loss_fn["prom"](pred_prom, label_prom)

        predicted_ss, ss_eval, ss_loss = 0, 0, 0
        if _model.splice_site_layer.num_labels == 1:
            predicted_ss = torch.round(pred_ss).item()
            ss_eval = (predicted_ss == label_ss.item())
            ss_loss = loss_fn["ss"](pred_ss, label_ss.float().reshape(-1, 1))
        else:
            ss_val, predicted_ss_index = torch.max(pred_ss, 1)
            predicted_ss = predicted_ss_index.item()
            ss_eval = (predicted_ss_index == label_ss.item())
            ss_loss = loss_fn["ss"](pred_ss, label_ss)

        predicted_polya, polya_eval, polya_loss = 0, 0, 0

        if _model.polya_layer.num_labels == 1:
            predicted_polya = torch.round(pred_polya).item()
            polya_eval = (predicted_polya == label_polya.item())
            polya_loss = loss_fn["polya"](pred_polya, label_polya.float().reshape(-1, 1))
        else:
            polya_val, predicted_polya_index = torch.max(pred_polya, 1)
            predicted_polya = predicted_polya_index.item()
            polya_eval = (predicted_polya_index == label_polya.item())
            polya_loss = loss_fn["polya"](pred_polya, label_polya)
        
    return (
        prom_eval, predicted_prom, label_prom.item(), prom_loss.item(),
        ss_eval, predicted_ss, label_ss.item(), ss_loss.item(),
        polya_eval, predicted_polya, label_polya.item(), polya_loss.item()
        )

def evaluate(model, dataloader, log_path, device, cur_epoch, loss_fn: dict, wandb: wandb=None):
    log_dir = os.path.dirname(log_path)
    log = {}
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_path):
        log = open(log_path, "x")
        log.write("epoch,step,prom_eval,prom_predict,prom_label,prom_loss,ss_eval,ss_predict,ss_label,ss_loss,polya_eval,polya_predict,polya_label,polya_loss\n")
    else:
        log = open(log_path, "a")
    model.eval()
    model.to(device)
    count_prom_correct, count_ss_correct, count_polya_correct = 0, 0, 0
    prom_accuracy, ss_accuracy, polya_accuracy = 0, 0, 0
    prom_evals, prom_predicts, prom_actuals, prom_losses = [], [], [], []
    ss_evals, ss_predicts, ss_actuals, ss_losses = [], [], [], []
    polya_evals, polya_predicts, polya_actuals, polya_losses = [], [], [], []

    len_dataloader = len(dataloader)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len_dataloader, desc=f"Evaluating"):
            input_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
            # prom_eval, prom_predict, prom_label, prom_loss, ss_eval, ss_predict, ss_label, ss_loss, polya_eval, polya_predict, polya_label, polya_loss = __eval__(model, input_ids, attn_mask, label_prom, label_ss, label_polya, device, loss_fn)
            prom_eval, prom_predict, prom_label, prom_loss, ss_eval, ss_predict, ss_label, ss_loss, polya_eval, polya_predict, polya_label, polya_loss = forward(model, input_ids, attn_mask, label_prom, label_ss, label_polya, loss_fn_prom=loss_fn["prom"], loss_fn_ss=loss_fn["ss"], loss_fn_polya=loss_fn["polya"])
            prom_evals.append(prom_eval)
            prom_predicts.append(prom_predict)
            prom_actuals.append(prom_label.item())
            prom_losses.append(prom_loss.item())
            ss_evals.append(ss_eval)
            ss_predicts.append(ss_predict)
            ss_actuals.append(ss_label.item())
            ss_losses.append(ss_loss.item())
            polya_evals.append(polya_eval)
            polya_predicts.append(polya_predict)
            polya_actuals.append(polya_label.item())
            polya_losses.append(polya_loss.item())
            log.write(f"{cur_epoch}, {step},{1 if prom_eval else 0},{prom_predict},{prom_label.item()},{prom_loss.item()},{1 if ss_eval else 0},{ss_predict},{ss_label.item()},{ss_loss.item()},{1 if polya_eval else 0},{polya_predict},{polya_label.item()},{polya_loss.item()}\n")
    #endfor
    log.close()
    # Compute average accuracy and loss.
    count_prom_correct = len([p for p in prom_evals if p])
    count_ss_correct = len([p for p in ss_evals if p])
    count_polya_correct = len([p for p in polya_evals if p])
    prom_accuracy = count_prom_correct / len(prom_evals) * 100
    ss_accuracy = count_ss_correct / len(ss_evals) * 100
    polya_accuracy = count_polya_correct / len(polya_evals) * 100
    avg_prom_loss = sum(prom_losses) / len(prom_losses)
    avg_ss_loss = sum(ss_losses)/ len(ss_losses)
    avg_polya_loss = sum(polya_losses) / len(polya_losses)

    return prom_accuracy, prom_loss, ss_accuracy, ss_loss, polya_accuracy, polya_loss

def train(dataloader: DataLoader, model: DNABERT_MTL, loss_fn: dict, optimizer, scheduler, batch_size: int, epoch_size: int, device='cpu', save_dir=None, training_counter=0, loss_strategy="sum", grad_accumulation_steps=1, wandb=None, eval_dataloader=None, device_list=[]):
    """
    @param      dataloader:
    @param      model:
    @param      loss_fn:
    @param      optimizer:
    @param      scheduler:
    @param      batch_size:
    @param      epoch_size:
    @param      device:
    @param      save_dir (string | None = None): dir path to save model per epoch. Inside this dir will be generated a dir for each epoch. If this path is None then model will not be saved.
    @param      grad_accumulation_steps (int | None = 1): After how many step backward is computed.
    """

    assert wandb != None, f"wandb not initialized."

    # Assign model to device.
    model.to(device)

    # Setup logging.
    log_file = None

    for t in ["train", "validation"]:
        wandb.define_metric(f"{t}/epoch")

    training_metrics = [
        "prom_loss", 
        "ss_loss", 
        "polya_loss", 
        "avg_prom_loss", 
        "avg_ss_loss", 
        "avg_polya_loss"]
    for t in training_metrics:
        wandb.define_metric(f"train/{t}", step_metric="train/epoch")

    validation_metrics = [
        "avg_prom_loss", 
        "avg_ss_loss", 
        "avg_polya_loss", 
        "avg_loss",
        "prom_loss", 
        "ss_loss", 
        "polya_loss", 
        "loss", 
        "avg_loss",
        "prom_accuracy",
        "ss_accuracy",
        "polya_accuracy", 
        "prom_error_rate", 
        "ss_error_rate", 
        "polya_error_rate"]
    for v in validation_metrics:
        wandb.define_metric(f"validation/{v}", step_metric="validation/epoch")

    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "log.csv")
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
        
        n_gpu = len(device_list)
        if n_gpu > 1:
            print(f"Enabling DataParallel")
            model = torch.nn.DataParallel(model, device_list)
        
        scaler = GradScaler()
        
        for i in range(epoch_size):
            epoch_start_time = datetime.now()
            model.train()
            model.zero_grad()
            epoch_loss = 0
            accumulate_prom_loss = 0
            accumulate_ss_loss = 0
            accumulate_polya_loss = 0
            avg_prom_loss = 0
            avg_ss_loss = 0
            avg_polya_loss = 0
            avg_loss = 0
            for step, batch in tqdm(enumerate(dataloader), total=len_dataloader, desc="Training Epoch [{}/{}]".format(i + 1 + training_counter, epoch_size)):
                in_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
                with autocast():
                    (prom_eval, predicted_prom, label_prom, prom_loss,
                    ss_eval, predicted_ss, label_ss, ss_loss,
                    polya_eval, predicted_polya, label_polya, polya_loss) = forward(model, in_ids, attn_mask, label_prom, label_ss, label_polya, loss_fn_prom=loss_fn["prom"], loss_fn_ss=loss_fn["ss"], loss_fn_polya=loss_fn["polya"])
                    # loss_prom, loss_ss, loss_polya = forward(model, in_ids, attn_mask, label_prom, label_ss, label_polya, loss_fn_prom=loss_fn["prom"], loss_fn_ss=loss_fn["ss"], loss_fn_polya=loss_fn["polya"])
                    
                    # Accumulate promoter, splice site, and poly-A loss.
                    accumulate_prom_loss += prom_loss
                    accumulate_ss_loss += ss_loss
                    accumulate_polya_loss += polya_loss

                    # Following MTDNN (Liu et. al., 2019), loss is summed.
                    loss = (prom_loss + ss_loss + polya_loss) / (3 if loss_strategy == "average" else 1)

                    # Log loss values and learning rate.
                    lr = optimizer.param_groups[0]['lr']
                    log_file.write("{},{},{},{},{},{}\n".format(i+training_counter, step, prom_loss.item(), ss_loss.item(), polya_loss.item(), lr))


                # Update parameters and learning rate for every batch.
                # Since this training is based on batch, then for every batch optimizer.step() and scheduler.step() are called.
                loss = loss / grad_accumulation_steps

                # Accumulate loss in this batch.
                epoch_loss += loss

                # Backpropagation.
                scaler.scale(loss).backward()

                if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    
                    # Clip gradient.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update learning rate and scheduler.
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Reset model gradients.
                    # model.zero_grad()
                    optimizer.zero_grad()
            # endfor batch.

            # Call scheduler in epoch loop.
            scheduler.step()

            # Calculate average loss for promoter, splice site, and poly-A.
            # Log losses with wandb.
            avg_prom_loss = accumulate_prom_loss / len_dataloader
            avg_ss_loss = accumulate_ss_loss / len_dataloader
            avg_polya_loss = accumulate_polya_loss / len_dataloader
            avg_epoch_loss = epoch_loss / len_dataloader

            epoch_duration = datetime.now() - epoch_start_time
            # Log epoch loss. Epoch loss equals to average of epoch loss over steps.
            log_entry = {
                "train/avg_prom_loss": avg_prom_loss.item(),
                "train/avg_ss_loss": avg_ss_loss.item(),
                "train/avg_polya_loss": avg_polya_loss.item(),
                "train/avg_loss": avg_epoch_loss.item(),
                "train/prom_loss": accumulate_prom_loss.item(),
                "train/ss_loss": accumulate_ss_loss.item(),
                "train/polya_loss": accumulate_polya_loss.item(),
                "train/loss": epoch_loss.item(),
                "train/epoch": i + training_counter
            }
            wandb.log(log_entry)
            wandb.watch(model)

            # After an epoch, eval model if eval_dataloader is given.
            prom_accuracy, ss_accuracy, polya_accuracy = 0, 0, 0
            if eval_dataloader:
                eval_log = os.path.join(save_dir, "eval_log.csv")
                prom_accuracy, prom_loss, ss_accuracy, ss_loss, polya_accuracy, polya_loss = evaluate(model, eval_dataloader, eval_log, device, i + training_counter, loss_fn, wandb=wandb)
                val_avg_accuracy = (prom_accuracy + ss_accuracy + prom_accuracy) / 3
                val_avg_loss = (prom_loss + ss_loss + polya_loss) / 3
                wandb.log({
                    "validation/prom_accuracy": prom_accuracy, 
                    "validation/ss_accuracy": ss_accuracy, 
                    "validation/polya_accuracy": polya_accuracy, 
                    "validation/avg_accuracy": val_avg_accuracy,
                    "validation/prom_loss": prom_loss.item(),
                    "validation/ss_loss": ss_loss.item(),
                    "validation/polya_loss": polya_loss.item(),
                    "validation/avg_loss": val_avg_loss.item(),
                    "validation/epoch": i + training_counter
                    })

            # Save model with best validation score.
            if val_avg_accuracy > best_accuracy:
                best_accuracy = val_avg_accuracy

                # Save checkpoint.
                checkpoint_path = os.path.join(save_dir, f"checkpoint-{i + training_counter}.pth")
                info = {
                    "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                    "epoch": (i + training_counter),
                    "batch_size": batch_size,
                    "device": device,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "prom_accuracy": prom_accuracy,
                    "ss_accuracy": ss_accuracy,
                    "polya_accuracy": polya_accuracy,
                    "prom_loss": prom_loss.item(),
                    "ss_loss": ss_loss.item(),
                    "polya_loss": polya_loss.item(),
                }
                save_checkpoint(model, optimizer, scheduler, info, checkpoint_path)

                # Had to save BERT layer separately because unknown error miskey match.
                _model = model
                if isinstance(model, DataParallel):
                    _model = model.module
                current_bert_layer = _model.shared_layer
                current_bert_layer.save_pretrained(save_dir)

                # Remove previous model.
                old_model_path = os.path.join(save_dir, os.path.basename(f"checkpoint-{i + training_counter - 1}.pth"))
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
    return model, optimizer, scheduler

def train_by_steps(dataloader: DataLoader, model: DNABERT_MTL, loss_fn: dict, optimizer, scheduler, max_steps: int, batch_size: int, save_dir: str, device: str='cpu', training_counter: int=0, loss_strategy: str="sum", grad_accumulation_steps=1, wandb=None, eval_dataloader=None, device_list=[]):
    num_epochs = max_steps // len(dataloader) + (1 if max_steps % len(dataloader) > 0 else 0)

    log_file_path = os.path.join(save_dir, "log.csv")
    save_model_path = os.path.join(save_dir)
    trained_model, trained_optimizer = train(dataloader, model, loss_fn, optimizer, scheduler, batch_size, num_epochs, log_file_path, device, save_model_path, training_counter=training_counter, loss_strategy=loss_strategy, grad_accumulation_steps=grad_accumulation_steps, wandb=wandb, eval_dataloader=eval_dataloader, device_list=device_list)

    return trained_model, trained_optimizer

def get_sequences(csv_path: str, n_sample=10, random_state=1337, do_kmer=False):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
    @param      csv_path (string): path to csv file.
    @param      n_sample (int): how many instance are retrieved from CSV located in `csv_path`.
    @param      random_state (int): random seed for randomly retriving `n_sample` instances.
    @param      do_kmer (bool): determine whether do kmer for sequences.
    @return     (list, list, list, list): sequence, label_prom, label_ss, label_polya.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File {} not found.".format(csv_path))
    df = pd.read_csv(csv_path)
    if (n_sample > 0):
        df = df.sample(n=n_sample, random_state=random_state)
    sequence = list(df['sequence'])
    sequence = [str_kmer(s, 3) for s in sequence] if do_kmer == True else sequence
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

def preprocessing(csv_file: str, pretrained_tokenizer_path: str, batch_size=2000, n_sample=0, random_state=1337, do_kmer=False):
    """
    @return dataloader (torch.utils.data.DataLoader)
    """
    csv_file = PureWindowsPath(csv_file)
    csv_file = str(Path(csv_file))
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
    _start_time = datetime.now()

    bert_path = PureWindowsPath(pretrained_tokenizer_path)
    bert_path = str(Path(bert_path))
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state, do_kmer=do_kmer)
    arr_input_ids, arr_attn_mask = prepare_data(sequences, tokenizer)
    prom_labels_tensor = torch.tensor(prom_labels)
    ss_labels_tensor = torch.tensor(ss_labels)
    polya_labels_tensor = torch.tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Preparing Dataloader duration {}".format(_elapsed_time))
    return dataloader

def preprocessing_batches(csv_file: str, pretrained_tokenizer_path: str, batch_sizes=[], n_sample=0, random_state=1337, do_kmer=False):
    if len(batch_sizes) == 0:
        raise ValueError(f"Batch sizes cannot be {batch_sizes}")
    csv_file = PureWindowsPath(csv_file)
    csv_file = str(Path(csv_file))
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
    _start_time = datetime.now()

    bert_path = PureWindowsPath(pretrained_tokenizer_path)
    bert_path = str(Path(bert_path))
    # tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state, do_kmer=do_kmer)
    arr_input_ids, arr_attn_mask = prepare_data(sequences, tokenizer)
    prom_labels_tensor = torch.tensor(prom_labels)
    ss_labels_tensor = torch.tensor(ss_labels)
    polya_labels_tensor = torch.tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloaders = []
    for batch_size in batch_sizes:
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print(f"Preparing Dataloaders duration {_elapsed_time}")
    return dataloaders
    
