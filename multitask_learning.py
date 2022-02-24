from torch import cuda
from transformers import BertForMaskedLM
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import os

_device = "cuda" if cuda.is_available() else "cpu"
_device
"""
Create simple multitask learning architecture with three task.
1. Promoter detection.
2. Splice-site detection.
3. poly-A detection.
"""
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import AdamW
from transformers import BertForMaskedLM
crossentropy_loss_func = CrossEntropyLoss()

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
        self.activation = nn.Softmax()

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
        self.activation = nn.Softmax()

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


    def forward(self, input_ids, attention_masks):
        x = self.shared_layer(input_ids=input_ids, attention_mask=attention_masks)
        x = x[0][:, 0, :]
        x1 = self.promoter_layer(x)
        x2 = self.splice_site_layer(x)
        x3 = self.polya_layer(x)
        return {'prom': x1, 'ss': x2, 'polya': x3}

def evaluate(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    model.to(device)
    val_prom_acc = []
    val_prom_loss = []
    val_ss_acc = []
    val_ss_loss = []
    val_polya_acc = []
    val_polya_loss = []

    for step, batch in enumerate(dataloader):
        b_input_ids, b_attn_masks, b_label_prom, b_label_ss, b_label_polya = tuple(t.to(device) for t in batch)

        # Compute logits.
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_masks)

            prom_logits = logits['prom']
            ss_logits = logits['ss']
            polya_logits = logits['polya']

            # Compute loss.
            prom_loss = loss_fn(prom_logits, b_label_prom).cpu().numpy().mean() * 100
            ss_loss = loss_fn(ss_logits, b_label_ss).cpu().numpy().mean() * 100
            polya_loss = loss_fn(polya_logits, b_label_polya).cpu().numpy().mean() * 100
            val_prom_loss.append(prom_loss)
            val_ss_loss.append(ss_loss)
            val_polya_loss.append(polya_loss)

            # Prediction.
            preds_prom = torch.argmax(prom_logits, dim=1).flatten()
            preds_ss = torch.argmax(ss_logits, dim=1).flatten()
            preds_polya = torch.argmax(polya_logits, dim=1).flatten()

            # Accuracy
            prom_acc = (preds_prom == b_label_prom).cpu().numpy().mean() * 100
            ss_acc = (preds_ss == b_label_ss).cpu().numpy().mean() * 100
            polya_acc = (preds_polya == b_label_polya).cpu().numpy().mean() * 100
            val_prom_acc.append(prom_acc)
            val_ss_acc.append(ss_acc)
            val_polya_acc.append(polya_acc)

    # Compute average acc and loss.
    avg_prom_acc = np.mean(val_prom_acc)
    avg_ss_acc = np.mean(val_ss_acc)
    avg_polya_acc = np.mean(val_polya_acc)
    avg_prom_loss = np.mean(val_prom_loss)
    avg_ss_loss = np.mean(val_ss_loss)
    avg_polya_loss = np.mean(val_polya_loss)

    return avg_prom_acc, avg_ss_acc, avg_polya_acc, avg_prom_loss, avg_ss_loss, avg_polya_loss

def train(dataloader, model, loss_fn, optimizer, scheduler, batch_size, epoch_size, log_file_path, device='cpu', eval=False, val_dataloader=None, loss_strategy='sum'):
    log_file = {}
    
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    batch_counts = 0
    batch_loss = 0
    batch_loss_prom, batch_loss_ss, batch_loss_polya = 0, 0, 0
    total_loss = 0
    total_loss_prom, total_loss_ss, total_loss_polya = 0, 0, 0
    _count = 0
    _cols = ['epoch','batch','loss_prom','loss_ss','loss_polya']

    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    log_file = open(log_file_path, 'x')
    log_file.write("{}\n".format(','.join(_cols)))
    _start_time = datetime.now()
    for i in range(epoch_size):
        for step, batch in enumerate(dataloader):
            batch_counts += 1
            print("Epoch {}, Step {}".format(i, step), end='\r')

            # Load batch to device.
            b_input_ids, b_attn_masks, b_labels_prom, b_labels_ss, b_labels_polya = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients.
            model.zero_grad()
            
            # Perform forward pass.
            outputs = model(b_input_ids, b_attn_masks)

            # Compute error.
            loss_prom = loss_fn(outputs['prom'], b_labels_prom)
            loss_ss = loss_fn(outputs['ss'], b_labels_ss)
            loss_polya = loss_fn(outputs['polya'], b_labels_polya)

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

            # Print training process.
            if (step % batch_size == 0 and step != 0) or (step == len(dataloader) - 1):
                batch_loss = 0
                batch_loss_prom = 0
                batch_loss_ss = 0
                batch_loss_polya = 0
                batch_counts = 0
        
            torch.cuda.empty_cache()
        # endfor batch.

        # Evaluate.
        if eval and val_dataloader:
            pa, ssa, pola, pl, ssl, poll = evaluate(model, val_dataloader, loss_fn, device)
            print('-----')
            print('prom acc: {}, prom loss: {}'.format(pa, pl))
            print('ss acc: {}, ss loss: {}'.format(ssa, ssl))
            print('polya acc: {}, polya loss: {}'.format(pola, poll))
            print('-----')
    # endfor epoch.
    log_file.close()
    _end_time = datetime.now()
    _elapsed_time = _end_time - _start_time
    print("Training time {}".format(_elapsed_time))
    return True
    
def test(dataloader, model, loss_fn, optimizer, batch_size, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Set model on evaluation model.
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test error: \n Accuracy: {(100*correct):>0.1f}% \n Avg Loss: {test_loss:>8f} \n")

def get_sequences(csv_path, n_sample=10, random_state=1337):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
    @param      csv_path (string): path to csv file.
    @param      n_sample (int): how many instance are retrieved from CSV located in `csv_path`.
    @param      random_state (int): random seed for randomly retriving `n_sample` instances.
    @return     (list, list, list, list): sequence, label_prom, label_ss, label_polya.
    """
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
    @param  data (string): string containing kmers separated by spaces.
    @param  tokenizer (Tokenizer): tokenizer initialized from pretrained values.
    @return input_ids (torch.Tensor): tensor of token ids to be fed to model.
    @return attention_masks (torch.Tensor): tensor of indices (a bunch of 'indexes') specifiying which token needs to be attended by model.
    """
    input_ids = []
    attention_masks = []

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

    return input_ids, attention_masks
