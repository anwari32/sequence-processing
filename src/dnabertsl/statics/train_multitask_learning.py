import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
import datetime
import os
import pandas as pd
from data_preparation import kmer
from transformers import BertTokenizer
from data_dir import workspace_dir, pretrained_3kmer_dir
from transformers import AdamW, get_linear_schedule_with_warmup, BertForMaskedLM
from multitask_learning import PolyAHead, PromoterHead, SpliceSiteHead, DNABERT_MTL, train


"""
Check if CUDA is supported.
"""
torch.cuda.is_available()
torch.device('cuda:0')
torch.cuda.get_device_name(0)
_device = torch.device('cuda:0')

def get_sequences(csv_path, n_sample=10, random_state=1337):
    r"""
    Get sequence from certain CSV. CSV has header such as 'sequence', 'label_prom', 'label_ss', 'label_polya'.
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

"""
Initialize tokenizer using BertTokenizer with pretrained weights from DNABert.
"""
tokenizer = BertTokenizer.from_pretrained('./pretrained/3-new-12w-0')

train_seq, train_label_prom, train_label_ss, train_label_polya = get_sequences('{}/train.expanded.csv'.format(workspace_dir))
validation_seq, val_label_prom, val_label_ss, val_label_polya = get_sequences('{}/validation.expanded.csv'.format(workspace_dir))

"""
Create dataloader.
"""
BATCH_SIZE = 2000 # DNABERT is trained this way.
EPOCH_SIZE = 4 # Experiment

_device = torch.device('cuda:0')
train_label_prom = torch.tensor(train_label_prom, device=_device)
train_label_ss = torch.tensor(train_label_ss, device=_device)
train_label_polya = torch.tensor(train_label_polya, device=_device)

train_inputs_ids, train_masks = preprocessing(train_seq, tokenizer)
train_data = TensorDataset(train_inputs_ids, train_masks, train_label_prom, train_label_ss, train_label_polya)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_label_prom = torch.tensor(val_label_prom, device=_device)
val_label_ss = torch.tensor(val_label_ss, device=_device)
val_label_polya = torch.tensor(val_label_polya, device=_device)

val_input_ids, val_masks = preprocessing(validation_seq, tokenizer)
val_data = TensorDataset(val_input_ids, val_masks, val_label_prom, val_label_ss, val_label_polya)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

print('# of training data: {}'.format(len(train_seq)))  
print('# of validation data: {}'.format(len(validation_seq)))

"""
Constructing model.
"""
polya_head = PolyAHead(_device)
promoter_head = PromoterHead(_device)
splice_head = SpliceSiteHead(_device)

dnabert_3_pretrained = pretrained_3kmer_dir
shared_parameter = BertForMaskedLM.from_pretrained(dnabert_3_pretrained).bert

model = DNABERT_MTL(shared_parameters=shared_parameter, promoter_head=promoter_head, polya_head=polya_head, splice_site_head=splice_head).to(_device)
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
training_steps = len(train_dataloader) * EPOCH_SIZE
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_steps)
loss_fn = CrossEntropyLoss()

_now = datetime.datetime.now()
_log_folder = "./logs/{}".format(_now.strftime("%y-%m-%d"))
if not os.path.exists(_log_folder):
    os.mkdir(_log_folder)
_log_name = "{}.csv".format(_now.strftime("%y-%m-%d-%H-%M-%S"))
_log_path = "{}/{}".format(_log_folder, _log_name)
print('Saving training log at {}.'.format(_log_path))
training_status = train(train_dataloader, model, loss_fn, optimizer, scheduler, BATCH_SIZE, EPOCH_SIZE, _log_path, _device, eval=True, val_dataloader=val_dataloader)
if training_status:
    _now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    store_model_result_bert_only_path = './result/gpu/bert/{}'.format(_now_str)
    store_model_result_all_path = './result/gpu/all/{}'.format(_now_str)
    for _path in [store_model_result_bert_only_path, store_model_result_all_path]:
        if not os.path.exists(_path):
            os.makedirs(_path)
    print("Saving BERT Layer at {}".format(store_model_result_bert_only_path))
    model.shared_layer.save_pretrained(store_model_result_bert_only_path)
    print("Saving Whole Model at {}".format(store_model_result_all_path))
    torch.save(model.state_dict(), store_model_result_all_path)