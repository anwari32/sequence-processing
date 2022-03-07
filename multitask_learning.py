import traceback
import torch
from torch import cuda
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn
from torch.optim import AdamW
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime
import pandas as pd
import os
import sys
from utils import save_model_state_dict

from models.mtl import MTModel, PolyAHead, PromoterHead, SpliceSiteHead

def init_model_mtl(pretrained_path, device="cpu"):
    polya_head = PolyAHead()
    promoter_head = PromoterHead()
    splice_head = SpliceSiteHead()

    dnabert_3_pretrained = pretrained_path
    shared_parameter = BertForMaskedLM.from_pretrained(dnabert_3_pretrained).bert
    model = MTModel(shared_parameters=shared_parameter, promoter_head=promoter_head, polya_head=polya_head, splice_site_head=splice_head)
    return model

def restore_model(model_path, device="cpu"):
    """
    @return     model
    """
    model = torch.load(model_path)
    return model.to(device)

def evaluate(dataloader, model, log_path, device='cpu'):
    model.eval()
    model.to(device)    
    count_prom_correct = 0
    count_ss_correct = 0
    count_polya_correct = 0
    prom_accuracy = 0
    ss_accuracy = 0
    polya_accuracy = 0
    len_dataloader = len(dataloader)
    for step, batch in enumerate(dataloader, total=len_dataloader, desc="Inference"):
        input_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)

        # Compute logits.
        with torch.no_grad():
            # Forward.
            output = model(input_ids, attn_mask)
            pred_prom = output['prom']
            pred_ss = output['ss']
            pred_polya = output['polya']

            # Prediction.
            predicted_prom = torch.round(pred_prom).item()
            actual_prom = label_prom.float().item()
            if (predicted_prom == actual_prom):
                count_prom_correct += 1
            print(pred_prom, label_prom, predicted_prom)

            predicted_ss, predicted_ss_index = torch.max(pred_ss, 1)
            predicted_ss = predicted_ss.item()
            predicted_ss_index = predicted_ss_index.item()
            if (predicted_ss_index == label_ss):
                count_ss_correct += 1

            predicted_polya, predicted_polya_index = torch.max(pred_polya, 1)
            predicted_polya = predicted_polya.item()
            predicted_polya_index = predicted_polya_index.item()
            if (predicted_polya_index == label_polya):
                count_polya_correct += 1
    #endfor
    # Compute average accuracy.
    prom_accuracy = count_prom_correct / len_dataloader * 100
    ss_accuracy = count_ss_correct / len_dataloader * 100
    polya = count_polya_correct / len_dataloader * 100
    return prom_accuracy, ss_accuracy, polya_accuracy

def train(dataloader: DataLoader, model: MTModel, loss_fn, optimizer, scheduler, batch_size: int, epoch_size: int, log_file_path, device='cpu', save_model_path=None, remove_old_model=False, training_counter=0, loss_strategy="sum", grad_accumulation_steps=1):
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
    @param      grad_accumulation_steps (int | None = 0): After how many step backward is computed.
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
    _start_time = datetime.now()
    _len_dataloader = len(dataloader)
    try:
        for i in range(epoch_size):
            for step, batch in tqdm(enumerate(dataloader), total=_len_dataloader, desc="Epoch [{}/{}]".format(i+1, epoch_size)):
                in_ids, attn_mask, label_prom, label_ss, label_polya = tuple(t.to(device) for t in batch)
                output = model(in_ids, attn_mask)
                pred_prom = output['prom']
                pred_ss = output['ss']
                pred_polya = output['polya']    
                loss_prom = model.promoter_loss_function(pred_prom, label_prom.float().reshape(-1, 1))
                loss_ss = model.splice_site_loss_function(pred_ss, label_ss)
                loss_polya = model.polya_loss_function(pred_polya, label_polya)

                # Following MTDNN (Liu et. al., 2019), loss is summed.
                loss = (loss_prom + loss_ss + loss_polya) / (3 if loss_strategy == "average" else 1)

                # Log loss values and learning rate.
                lr = optimizer.param_groups[0]['lr']
                log_file.write("{},{},{},{},{},{}\n".format(i, step, loss_prom, loss_ss, loss_polya, lr))

                # Update parameters and learning rate for every batch.
                # Since this training is based on batch, then for every batch optimizer.step() and scheduler.step() are called.
                if grad_accumulation_steps > 1:
                    loss = loss / grad_accumulation_steps

                # Backpropagation.
                loss.backward()


                if (step + 1) % grad_accumulation_steps == 0:
                    # Update learning rate and scheduler.
                    optimizer.step()
                    scheduler.step()

                    # Reset model gradients.
                    model.zero_grad()

                    # Empty cuda cache to save memory.
                    torch.cuda.empty_cache()

                # Just print something so terminal doesn't look so boring. (-_-)'
                # print("Epoch {}, Step {}".format(i, step), end='\r')

            # endfor batch.

            # After and epoch, Save the model if `save_model_path` is not None.
            save_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_model_state_dict(model, save_model_path, "epoch-{}.pth".format(i+training_counter))
            # torch.save(model.state_dict(), os.path.join(save_model_path, "epoch-{}.pth".format(i+training_counter)))
            if remove_old_model:
                old_model_path = os.path.join(save_model_path, os.path.basename("epoch-{}.pth".format(i+training_counter-1)))
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
        # endfor epoch.
    except Exception as e:
        log_file.close()
        print(traceback.format_exc())
        print(e)

    log_file.close()
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Start Time: {}, End Time: {}, Training Duration {}".format(_start_time, _end_time, _elapsed_time))
    return model

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
    label_prom = list(df['label_prom'])
    label_ss = list(df['label_ss'])
    label_polya = list(df['label_polya'])

    return sequence, label_prom, label_ss, label_polya

def preprocessing(data, tokenizer: BertTokenizer):
    """
    Preprocessing for pretrained BERT.
    @param      data (array of string): array of string, each string contains kmers separated by spaces.
    @param      tokenizer (Tokenizer): tokenizer initialized from pretrained values.
    @return     input_ids, attention_masks (tuple of torch.Tensor): tensor of token ids to be fed to model,
                tensor of indices (a bunch of 'indexes') specifiying which token needs to be attended by model.
    """
    input_ids = []
    attention_masks = []
    _start_time = datetime.now()
    _count = 0
    _len_data = len(data)
    for sequence in data:
        """
        Sequence is 512 characters long.
        """
        _count += 1
        if _count < _len_data:
            print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data), end='\r')
        else:
            print("Seq length = {} [{}/{}]".format(len(sequence.split(' ')), _count, _len_data))
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
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Preprocessing Duration {}".format(_elapsed_time))
    return input_ids, attention_masks

def prepare_data(csv_file: str, pretrained_tokenizer_path: str, batch_size=2000, n_sample=0, random_state=1337):
    """
    @return dataloader (torch.utils.data.DataLoader)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
        sys.exit(2)
    _start_time = datetime.now()
    tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state)
    arr_input_ids, arr_attn_mask = preprocessing(sequences, tokenizer)
    prom_labels_tensor = tensor(prom_labels)
    ss_labels_tensor = tensor(ss_labels)
    polya_labels_tensor = tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Preparing Dataloader duration {}".format(_elapsed_time))
    return dataloader

