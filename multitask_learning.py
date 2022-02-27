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

_device = "cuda" if cuda.is_available() else "cpu"
_device
"""
Create simple multitask learning architecture with three task.
1. Promoter detection.
2. Splice-site detection.
3. poly-A detection.
"""

def _get_adam_optimizer(parameters, lr=0, eps=0, beta=0):
    return AdamW(parameters, lr=lr, eps=eps, betas=beta)

class PromoterHead(nn.Module):
    """
    Network configuration can be found in DeePromoter (Oubounyt et. al., 2019).
    Classification is done by using Sigmoid. Loss is calculated by CrossEntropyLoss.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, out_features=128, device=device), # Adapt 768 unit from BERT to 128 unit for DeePromoter's fully connected layer.
            nn.ReLU(), # Asssume using ReLU.
            nn.Dropout(),
            nn.Linear(128, 1, device=device),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.stack(x)
        x = self.activation(x)
        return x

class SpliceSiteHead(nn.Module):
    """
    Network configuration can be found in Splice2Deep (Albaradei et. al., 2020).
    Classification layer is using Softmax function and loss is calculated by ???.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, out_features=512, device=device),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2, device=device)
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stack(x)
        x = self.activation(x)
        return x

class PolyAHead(nn.Module):
    """
    Network configuration can be found in DeeReCT-PolyA (Xia et. al., 2018).
    Loss function is cross entropy and classification is done by using Softmax.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(768, 64, device=device), # Adapt from BERT layer which provide 768 outputs.
            nn.ReLU(), # Assume using ReLU.
            nn.Dropout(),
            nn.Linear(64, 2, device=device),
        )
        self.activation = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.stack(x)
        x = self.activation(x)
        return x

class MTModel(nn.Module):
    """
    Core architecture. This architecture consists of input layer, shared parameters, and heads for each of multi-tasks.
    """
    def __init__(self, shared_parameters, promoter_head, splice_site_head, polya_head):
        super().__init__()
        self.shared_layer = shared_parameters
        self.promoter_layer = promoter_head
        self.splice_site_layer = splice_site_head
        self.polya_layer = polya_head
        self.promoter_loss_function = nn.BCELoss()
        self.splice_site_loss_function = nn.CrossEntropyLoss()
        self.polya_loss_function = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_masks):
        x = self.shared_layer(input_ids=input_ids, attention_mask=attention_masks)
        x = x[0][:, 0, :]
        x1 = self.promoter_layer(x)
        x2 = self.splice_site_layer(x)
        x3 = self.polya_layer(x)
        return {'prom': x1, 'ss': x2, 'polya': x3}

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

def forward(model, input_ids, attention_mask):
    pred_prom, pred_ss, pred_polya = model(input_ids, attention_mask)
    return pred_prom, pred_ss, pred_polya

def evaluate(dataloader, model, loss_fn, log, device='cpu'):
    model.eval()
    model.to(device)
    val_prom_acc = []
    val_prom_loss = []
    val_ss_acc = []
    val_ss_loss = []
    val_polya_acc = []
    val_polya_loss = []

    if os.path.exists(log):
        os.remove(log)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    log_file = open(log, 'x')
    log_file.write('step,loss_prom,loss_ss,loss_polya,acc_prom,acc_ss,acc_polya\n')
    for step, batch in enumerate(dataloader):
        b_input_ids, b_attn_mask, b_label_prom, b_label_ss, b_label_polya = tuple(t.to(device) for t in batch)

        # Compute logits.
        with torch.no_grad():
            # Forward.
            pred_prom, pred_ss, pred_polya = forward(model, b_input_ids, b_attn_mask)

            # Get loss function.
            prom_loss_function = loss_fn['prom']
            ss_loss_function = loss_fn['ss']
            polya_loss_function = loss_fn['polya']

            # Compute loss.
            prom_loss = prom_loss_function(pred_prom, b_label_prom.float().reshape(-1,1)).cpu().numpy().mean() * 100
            ss_loss = ss_loss_function(pred_ss, b_label_ss).cpu().numpy().mean() * 100
            polya_loss = polya_loss_function(pred_polya, b_label_polya).cpu().numpy().mean() * 100
            val_prom_loss.append(prom_loss)
            val_ss_loss.append(ss_loss)
            val_polya_loss.append(polya_loss)

            # Prediction.
            preds_prom = torch.argmax(pred_prom, dim=1).flatten()
            preds_ss = torch.argmax(pred_ss, dim=1).flatten()
            preds_polya = torch.argmax(pred_polya, dim=1).flatten()

            # Accuracy
            prom_acc = (preds_prom == b_label_prom).cpu().numpy().mean() * 100
            ss_acc = (preds_ss == b_label_ss).cpu().numpy().mean() * 100
            polya_acc = (preds_polya == b_label_polya).cpu().numpy().mean() * 100
            val_prom_acc.append(prom_acc)
            val_ss_acc.append(ss_acc)
            val_polya_acc.append(polya_acc)

            log_file.write('{},{},{},{},{},{},{}\n'.format(
                step,
                prom_loss,
                ss_loss,
                polya_loss,
                prom_acc,
                ss_acc,
                polya_acc
            ))
    #endfor

    log_file.close()
    # Compute average acc and loss.
    avg_prom_acc = np.mean(val_prom_acc)
    avg_ss_acc = np.mean(val_ss_acc)
    avg_polya_acc = np.mean(val_polya_acc)
    avg_prom_loss = np.mean(val_prom_loss)
    avg_ss_loss = np.mean(val_ss_loss)
    avg_polya_loss = np.mean(val_polya_loss)

    return avg_prom_acc, avg_ss_acc, avg_polya_acc, avg_prom_loss, avg_ss_loss, avg_polya_loss

def train(dataloader, model, loss_fn, optimizer, scheduler, batch_size, epoch_size, log_file_path, device='cpu', loss_strategy='sum', save_model_path=None, remove_old_model=False):
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
    @param      loss_strategy:
    @param      save_model_path (string | None = None): dir path to save model per epoch. Inside this dir will be generated a dir for each epoch. If this path is None then model will not be saved.
    """
    log_file = {}    
    model.to(device)
    model.train()
    batch_counts = 0
    batch_loss = 0
    batch_loss_prom, batch_loss_ss, batch_loss_polya = 0, 0, 0

    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    _cols = ['epoch','batch','loss_prom','loss_ss','loss_polya']
    log_file = open(log_file_path, 'x')
    log_file.write("{}\n".format(','.join(_cols)))
    _start_time = datetime.now()
    for i in range(epoch_size):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_counts += 1
            # print("Epoch {}, Step {}".format(i, step), end='\r')

            # Load batch to device.
            b_input_ids, b_attn_masks, b_labels_prom, b_labels_ss, b_labels_polya = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients.
            model.zero_grad()
            
            # Perform forward pass.
            outputs = model(b_input_ids, b_attn_masks)

            # Define loss function
            prom_loss_function = loss_fn['prom'] if loss_fn['prom'] != None else model.promoter_loss_function
            ss_loss_function = loss_fn['ss'] if loss_fn['ss'] != None else model.splice_sites_loss_function
            polya_loss_function = loss_fn['polya'] if loss_fn['polya'] != None else model.polya_loss_function
            
            # Compute error.
            loss_prom = prom_loss_function(outputs['prom'], b_labels_prom.float().reshape(-1, 1))
            loss_ss = ss_loss_function(outputs['ss'], b_labels_ss)
            loss_polya = polya_loss_function(outputs['polya'], b_labels_polya)

            # Following MTDNN (Liu et. al., 2019), loss is summed.
            if loss_strategy == 'average':
                loss = (loss_prom + loss_ss + loss_polya)/3
            else:
                loss = loss_prom + loss_ss + loss_polya

            # Compute this batch error.
            batch_loss_prom += loss_prom
            batch_loss_ss += loss_ss
            batch_loss_polya += loss_polya
            batch_loss += loss

            # Log loss values.
            log_file.write("{},{},{},{},{}\n".format(i, batch_counts, loss_prom, loss_ss, loss_polya))

            # Backpropagation.
            loss.backward()

            # Update parameters and learning rate.
            optimizer.step()
            scheduler.step()

            # Reset accumulation loss.
            if (step % batch_size == 0 and step != 0) or (step == len(dataloader) - 1):
                batch_loss = 0
                batch_loss_prom = 0
                batch_loss_ss = 0
                batch_loss_polya = 0
                batch_counts = 0

            # Empty cuda cache to save memory.
            torch.cuda.empty_cache()
        # endfor batch.

        # After and epoch, Save the model if `save_model_path` is not None.
        if save_model_path != None:
            save_path = os.path.join(save_model_path)
            if os.path.exists(save_path):
                if not os.path.isdir(save_path):
                    print('Save path is not path to directory.')
                    sys.exit(2)
            else:
                os.makedirs(save_path)
            save_path_model = os.path.join(save_path, 'epoch-{}'.format(i))
            torch.save(model, save_path_model)
            
            # Remove old model.
            if remove_old_model:
                if (i-1) >= 0:
                    os.remove(os.path.join(save_path, 'epoch-{}'.format(i-1)))
        
    # endfor epoch.

    log_file.close()
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Training Duration {}".format(_elapsed_time))
    return model

def get_sequences(csv_path, n_sample=10, random_state=1337):
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

def preprocessing(data, tokenizer):
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

def prepare_data(csv_file, pretrained_tokenizer_path, batch_size=2000, n_sample=0, random_state=1337):
    """
    @return dataloader (torch.utils.data.DataLoader)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError("File {} not found.".format(csv_file))
        sys.exit(2)
    tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path)
    sequences, prom_labels, ss_labels, polya_labels = get_sequences(csv_file, n_sample=n_sample, random_state=random_state)
    arr_input_ids, arr_attn_mask = preprocessing(sequences, tokenizer)
    prom_labels_tensor = tensor(prom_labels)
    ss_labels_tensor = tensor(ss_labels)
    polya_labels_tensor = tensor(polya_labels)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, prom_labels_tensor, ss_labels_tensor, polya_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


