import torch
from torch.utils.data import TensorDataset, DataLoader
from data_dir import pretrained_3kmer_dir
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained(pretrained_3kmer_dir)

def create_dataloader_from_csv(src_file, tokenizer, batch_size=1):
    df = pd.read_csv(src_file)

    arr_input_ids = []
    arr_attn_mask = []
    arr_token_type_ids = []
    arr_label_prom = []
    arr_label_ss = []
    arr_label_polya = []
    for i, r in tqdm(df.iterrows(), total=df.shape[0]):
        sent = r["sequence"]
        encoded = tokenizer.encode_plus(
            text=sent, 
            padding="max_length", 
            max_length=512, 
            add_special_tokens=True, 
            return_token_type_ids=True, 
            return_attention_mask=True
        )
        input_ids = encoded.get("input_ids")
        attn_mask = encoded.get("attention_mask")
        token_type_ids = encoded.get("token_type_ids")

        arr_input_ids.append(input_ids)
        arr_attn_mask.append(attn_mask)
        arr_token_type_ids.append(token_type_ids)

        arr_label_prom.append(torch.tensor([int(r["label_prom"])]))
        arr_label_ss.append(torch.tensor([int(r["label_ss"])]))
        arr_label_polya.append(torch.tensor([int(r["label_polya"])]))

    arr_input_ids = torch.tensor(arr_input_ids)
    arr_attn_mask = torch.tensor(arr_attn_mask)
    arr_token_type_ids = torch.tensor(arr_token_type_ids)
    arr_label_prom = torch.tensor(arr_label_prom)
    arr_label_ss = torch.tensor(arr_label_ss)
    arr_label_polya = torch.tensor(arr_label_polya)

    dataset = TensorDataset(arr_input_ids, arr_attn_mask, arr_token_type_ids, arr_label_prom, arr_label_polya, arr_label_polya)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader