from transformers import BertTokenizer
from __models.genlab import DNABERT_LSTM
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils.seqlab import preprocessing_kmer, convert_ids_to_tokens
from utils.tokenizer import get_default_tokenizer

import torch
import wandb
import os

from utils.utils import save_checkpoint

def forward(model: DNABERT_LSTM, optimizer, dataloader: DataLoader, device: str, loss_function, gene_name: str=None, scaler: GradScaler=None, wandb: wandb = None, mode: str = "train", epoch: int=0, num_epoch: int=0):
    """
    This function utilizes non-overlapping sequence.
    """
    # Make sure model and data are in the same device.
    model.to(device)
    contig_predicted_labels = []
    contig_target_labels = []
    scaler = GradScaler()
    description = f"Training {gene_name} Epoch {epoch + 1}/{num_epoch}" if mode == "train" else f"Validating {gene_name} Epoch {epoch + 1}/{num_epoch}"

    hidden_units = None
    for step, batch in enumerate(dataloader):
        input_ids, attn_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
        contig_loss = None
        # accumulated_contig_loss = None
        prediction = None
        with autocast(enabled=True, cache_enabled=True):
            prediction, hidden_output = model(input_ids, attn_mask, hidden_units)
            hidden_output = hidden_units
            for pred, label in zip(prediction, labels): # Iterate through batch dimension.
                contig_predicted_labels.append(pred)
                contig_target_labels.append(label)
                contig_loss = loss_function(pred, label)
                wandb.log({
                    "contig_loss": contig_loss.item()
                })
            #endfor
        num_labels = 8
        batch_loss = loss_function(prediction.view(-1, num_labels), labels.view(-1))
        wandb.log({
            "loss": batch_loss.item()
        })

        if mode == "train":
            if scaler:
                # scaler.scale(accumulated_contig_loss).backward()
                scaler.scale(batch_loss).backward()
            else:
                # accumulated_contig_loss.backward()
                batch_loss.backward()

            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()
            optimizer.zero_grad()

    predicted_assembly = contig_predicted_labels[0]
    target_assembly = contig_target_labels[0]
    for pred, target in zip(contig_predicted_labels[1:], contig_target_labels[1:]):

        # Appending contigs.
        predicted_assembly = torch.concat((predicted_assembly, pred), 0)
        target_assembly = torch.concat((target_assembly, target), 0)

    gene_loss = None
    gene_accuracy = 0
    if loss_function:
        gene_loss = loss_function(predicted_assembly, target_assembly)
        wandb.log({
            "gene_loss": gene_loss.item()
        })

    return gene_loss, predicted_assembly, target_assembly, scaler

def evaluate(model, eval_genes, device, eval_log, epoch, num_epoch, loss_fn, wandb):
    model.eval()
    eval_logfile = {}
    if not os.path.exists(eval_log):
        eval_logfile = open(eval_log, "x")
        eval_logfile.write(f"epoch,gene,accuracy,error,loss,predicted_label,target_label\n")
    else:
        eval_logfile = open(eval_log, "a")

    # Sum accuracy, incorrect scores.
    accuracy_score_sum, incorrect_score_sum, gene_loss_sum = 0, 0, 0
        
    for gene in tqdm(eval_genes, desc=f"Validating Epoch {epoch + 1}/{num_epoch}", total=len(eval_genes)):
        gene_name = os.path.basename(gene).split('.')[0]
        gene_dir = os.path.dirname(gene)
        gene_dir, gene_chr = os.path.split(gene_dir)
        dataloader = preprocessing_kmer(gene, get_default_tokenizer(), 1, disable_tqdm=True)
        accuracy_score, incorrect_score, predicted_label_token, target_label_token, gene_loss = eval_gene(model, dataloader, device, loss_fn, gene_name=gene_name, wandb=wandb, at_epoch=epoch, num_epoch=num_epoch)
        accuracy_score_sum += accuracy_score
        incorrect_score_sum += incorrect_score
        gene_loss_sum += gene_loss.item()

        # EDIT 15 May 2022: Remove details for each gene since everything can be seen from eval log.
        # EDIT 5 June 2022: Gene evaluation details are logged.
        # Log accuracy and incorrect score for each gene after an epoch.
        wandb.define_metric("validation/epoch")
        wandb.define_metric(f"validation/{gene_chr}-{gene_name}/accuracy", step_metric="validation/epoch")
        wandb.define_metric(f"validation/{gene_chr}-{gene_name}/error", step_metric="validation/epoch")
        wandb.define_metric(f"validation/{gene_chr}-{gene_name}/loss", step_metric="validation/epoch")
        log_entry = {
            f"validation/{gene_chr}-{gene_name}/accuracy": accuracy_score,
            f"validation/{gene_chr}-{gene_name}/error": incorrect_score,
            f"validation/{gene_chr}-{gene_name}/loss": gene_loss.item(),
            f"validation/epoch": epoch
        }
        wandb.log(log_entry)

        eval_logfile.write(f"{epoch},{gene_chr}-{gene_name},{accuracy_score},{incorrect_score},{gene_loss.item()},{' '.join(predicted_label_token)},{' '.join(target_label_token)}\n")

    #endfor
    eval_logfile.close()
    n_eval_genes = len(eval_genes)
    avg_accuracy_score = accuracy_score_sum / n_eval_genes # Average accuracy over all genes.
    avg_incorrect_score = incorrect_score_sum / n_eval_genes # Average inaccuracy over all genes.
    avg_gene_loss_score = gene_loss_sum / n_eval_genes # Average loss over all genes.

    return avg_accuracy_score, avg_incorrect_score, avg_gene_loss_score

def eval_gene(model, dataloader, device, loss_fn, gene_name: str = None, wandb: wandb = None, at_epoch: int = 0, num_epoch: int = 0):
    model.to(device)
    model.eval()
    correct_label, incorrect_label = 0, 0
    predicted_label_token, target_label_token = [], []

    with torch.no_grad():
        gene_loss, predicted_label_tensor, target_label_tensor, scaler = forward(model, None, dataloader, device, loss_fn, gene_name=gene_name, mode="validation", wandb=wandb, epoch=at_epoch, num_epoch=num_epoch)
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

    return accuracy_score, incorrect_score, predicted_label_token, target_label_token, gene_loss

def train(model, tokenizer: BertTokenizer, optimizer, scheduler, train_genes: list, loss_function, num_epoch=1, batch_size=1, device="cpu", save_dir=None, training_counter=0, wandb=None, eval_genes=None, device_list=[]):
    n_gpu = len(device_list)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_list)
        print(f"Main Device {device}")
        print(f"Device List {device_list}")
    else:
        print(f"Device {device}")

    scaler = GradScaler()

    # Initialize log.
    log_file_path = os.path.join(save_dir, "log.csv")
    logfile = open(log_file_path, "x")
    logfile.write("epoch,gene,gene_loss,epoch_loss,lr\n")

    num_training_genes = len(train_genes)
    best_accuracy = 0

    TRAINING_EPOCH = "train/epoch"
    TRAINING_LOSS = "train/loss" # Accumulated gene losses.
    TRAINING_AVG_LOSS = "train/avg_loss" # Accumulated gene losses over all genes.
    TRAINING_LR = "train/learning_rate" # Training learning rate.

    VALIDATION_EPOCH = "validation/epoch"
    VALIDATION_AVG_ACC = "validation/average_accuracy"
    VALIDATION_AVG_INACC = "validation/average_inaccuracy"
    VALIDATION_AVG_LOSS = "validation/average_loss"

    
    wandb.define_metric(TRAINING_EPOCH)
    wandb.define_metric(TRAINING_LOSS, step_metric=TRAINING_EPOCH)
    wandb.define_metric(TRAINING_AVG_LOSS, step_metric=TRAINING_EPOCH)
    wandb.define_metric(TRAINING_LR, step_metric=TRAINING_LOSS)

    wandb.define_metric(VALIDATION_EPOCH)
    wandb.define_metric(VALIDATION_AVG_ACC, step_metric=VALIDATION_EPOCH) # Avaerage accuracy.
    wandb.define_metric(VALIDATION_AVG_INACC, step_metric=VALIDATION_EPOCH) # Average inaccuracy.
    wandb.define_metric(VALIDATION_AVG_LOSS, step_metric=VALIDATION_EPOCH) # Average gene loss.

    for epoch in range(training_counter, num_epoch):
        model.train()
        epoch_loss = None

        lr = 0 # Learning rate.
        for i in tqdm(range(num_training_genes), desc=f"Training Epoch {epoch + 1}/{num_epoch}", total=num_training_genes, unit="gene"):
            
            gene = train_genes[i]
            gene_name = os.path.basename(gene).split(".")[0]
            gene_dir = os.path.dirname(gene)
            gene_dir, gene_chr = os.path.split(gene_dir)
            gene_dataloader = preprocessing_kmer(gene, tokenizer, batch_size, disable_tqdm=True)

            # gene_loss = None # This is loss computed from single gene.
            gene_loss, predicted_label, target_label, scaler = forward(model, optimizer, gene_dataloader, device, loss_function=loss_function, wandb=wandb, gene_name=gene_name, scaler=scaler, epoch=epoch, num_epoch=num_epoch, mode="train")
            
            gene_loss = gene_loss
            epoch_loss = gene_loss if epoch_loss == None else epoch_loss + gene_loss

            # Get current learning rate and log it.
            lr = optimizer.param_groups[0]['lr']

            # Write gene training log.
            logfile.write(f"{epoch},{gene_chr}-{gene_name},{gene_loss.item()},{epoch_loss.item()},{lr}\n")            

            # Gradient is cleared after a gene has been processed.
            # Optimizer is reset after a gene is finised.
            # EDIT 11 May 2022: Moved gradient accumulation and clearance at forward function.
            scaler.step(optimizer)
            scaler.update()

            wandb.log({
                TRAINING_LR: lr,
                TRAINING_EPOCH: epoch,
            })
        
        # Moved scheduler to epoch loop.
        scheduler.step()

        # Record epoch loss.
        # Epoch loss is accumulation of all gene losses.
        wandb.log({
            TRAINING_LOSS: epoch_loss.item(), 
            TRAINING_AVG_LOSS: epoch_loss.item() / num_training_genes,
            TRAINING_EPOCH: epoch
        })            

        # Eval model if eval_genes is available.
        if eval_genes:
            eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
            avg_accuracy, avg_inaccuracy, avg_gene_loss = evaluate(model, eval_genes, device, eval_log, epoch, num_epoch, loss_function, wandb)

            validation_log = {
                VALIDATION_AVG_ACC: avg_accuracy,
                VALIDATION_AVG_INACC: avg_inaccuracy,
                VALIDATION_AVG_LOSS: avg_gene_loss,
                VALIDATION_EPOCH: epoch
            }
            wandb.log(validation_log)

            # Save trained model if this epoch produces better model.
            # EDIT: 5 June 2022: Just save every model.
            _model = model
            if isinstance(model, torch.nn.DataParallel):
                _model = model.module

            cur_config = {
                "epoch": epoch,
                "num_epochs": num_epoch,
                "batch_size": batch_size,
                "accuracy": avg_accuracy
            }
            save_checkpoint(_model, optimizer, scheduler, cur_config, os.path.join(save_dir, f"checkpoint-{epoch}"))

    logfile.close()
    return model, optimizer, scheduler