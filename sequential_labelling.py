import torch
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import os
from tqdm import tqdm
from utils.utils import save_model_state_dict, save_checkpoint
from models.seqlab import DNABERTSeqLab
from datetime import datetime
import wandb
from utils.seqlab import preprocessing_kmer, convert_ids_to_tokens
from utils.tokenizer import get_default_tokenizer
from torch.cuda.amp import autocast, GradScaler


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

def __forward_gene_non_overlap__(model: DNABERTSeqLab, dataloader: DataLoader, device: str, loss_function=None, gene_name: str=None, scaler: GradScaler=None, wandb: wandb = None, mode: str = "train", epoch=0, num_epoch=0):
    """
    This function utilizes non-overlapping sequence.
    """
    # Assertion
    assert mode == "train" or mode == "validation", f"Expected `train` or `validation` but found {mode}"

    # Make sure model and data are in the same device.
    model.to(device)
    contig_predicted_labels = []
    contig_target_labels = []
    scaler = GradScaler()
    description = f"Training {gene_name} Epoch {epoch + 1}/{num_epoch}" if mode == "train" else f"Validating {gene_name} Epoch {epoch + 1}/{num_epoch}"

    #if wandb != None:
    #    if mode == "train":
    #        wandb.define_metric(f"{gene_name}/train_step")
    #        wandb.define_metric(f"{gene_name}/train_contig_loss", step_metric=f"{gene_name}/train_step")
    #    if mode == "validation":
    #        wandb.define_metric(f"{gene_name}/validation_step")
    #        wandb.define_metric(f"{gene_name}/validation_contig_loss", step_metric=f"{gene_name}/validation_step")

    # for step, batch in tqdm(enumerate(dataloader), desc=description, total=len(dataloader)):
    for step, batch in enumerate(dataloader):
        input_ids, attn_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
        contig_loss = None
        with autocast(enabled=True, cache_enabled=True):
            # prediction = model(input_ids, attn_mask, token_type_ids)
            # Not using `token_type_ids` anymore.
            prediction = model(input_ids, attn_mask)
            for pred, label in zip(prediction, labels): # Iterate through batch dimension.
                contig_predicted_labels.append(pred)
                contig_target_labels.append(label)
                assert pred != None, f"Prediction must not be None, got {pred}"
                assert label != None, f"Label must not be None, got {label}"
                if contig_loss == None:
                    contig_loss = loss_function(pred, label)
                else:
                    contig_loss += loss_function(pred, label)
            #endfor
        
        #if wandb != None:
        #    if mode == "train":
        #        log_entry = {
        #            f"{gene_name}/train_contig_loss": contig_loss.item(),
        #            f"{gene_name}/train_step": step
        #        }
        #        wandb.log(log_entry)
        #    if mode == "validation":
        #        log_entry = {
        #            f"{gene_name}/validation_contig_loss": contig_loss.item(),
        #            f"{gene_name}/validation_step": step
        #        }
        #        wandb.log(log_entry)

        if mode == "train":
            if scaler:
                scaler.scale(contig_loss).backward()
            else:
                contig_loss.backward()

    # ``contig_predicted_labels`` is array of tensors (512, dim), first token and last token are special token hence they need to be removed.
    contig_predicted_labels = [t[1:511] for t in contig_predicted_labels] # Convert each tensor(510, dim) into array of 510 element.
    # ``contig_target_labels`` is array of tensors (512), first token and last token are special token hence they need to be removed.
    contig_target_labels = [t[1:511] for t in contig_target_labels] # Each element in ``contig_target_labels`` is a tensor with 510 element.
    
    # print(contig_predicted_labels, contig_predicted_labels[0].shape)
    # print(contig_target_labels, contig_target_labels[0].shape)

    # We need to merge contigs in ``contig_predicted_labels`` into single assembly. First we convert those tensor-label sequence into label token.
    # and also merge target label in ``contig_target_labels`` into single assembly.
    predicted_assembly = contig_predicted_labels[0]
    target_assembly = contig_target_labels[0]
    for pred, target in zip(contig_predicted_labels[1:], contig_target_labels[1:]):

        # Appending contigs.
        predicted_assembly = torch.concat((predicted_assembly, pred), 0)
        target_assembly = torch.concat((target_assembly, target), 0)

    gene_loss = None
    if loss_function:
        gene_loss = loss_function(predicted_assembly, target_assembly)

    return gene_loss, predicted_assembly, target_assembly, scaler

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

def __eval_gene__(model, dataloader, device, loss_fn, gene_name: str = None, wandb: wandb = None, at_epoch: int = 0, num_epoch: int = 0):
    model.to(device)
    model.eval()
    correct_label, incorrect_label = 0, 0
    predicted_label_token, target_label_token = [], []

    with torch.no_grad():
        gene_loss, predicted_label_tensor, target_label_tensor, scaler = __forward_gene_non_overlap__(model, dataloader, device, loss_fn, gene_name=gene_name, mode="validation", wandb=wandb, epoch=at_epoch, num_epoch=num_epoch)
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

def evaluate_genes(model, eval_genes, device, eval_log, epoch, num_epoch, loss_fn, wandb=None):
    model.eval()
    eval_logfile = {}
    if not os.path.exists(eval_log):
        eval_logfile = open(eval_log, "x")
        eval_logfile.write(f"epoch,gene,accuracy,error,loss,predicted_label,target_label\n")
    else:
        eval_logfile = open(eval_log, "a")

    # Sum accuracy, incorrect scores.
    accuracy_score_sum, incorrect_score_sum, gene_loss_sum = 0, 0, 0
        
    for gene in eval_genes:
        gene_name = os.path.basename(gene).split('.')[0]
        dataloader = preprocessing_kmer(gene, get_default_tokenizer(), 1)
        accuracy_score, incorrect_score, predicted_label_token, target_label_token, gene_loss = __eval_gene__(model, dataloader, device, loss_fn, gene_name=gene_name, wandb=wandb, at_epoch=epoch, num_epoch=num_epoch)
        accuracy_score_sum += accuracy_score
        incorrect_score_sum += incorrect_score
        gene_loss_sum += gene_loss.item()
    
        # Log accuracy and incorrect score for each gene after an epoch.
        if wandb != None:
            wandb.define_metric(f"{gene_name}/validation_accuracy", step_metric="epoch/epoch")
            wandb.define_metric(f"{gene_name}/validation_error", step_metric="epoch/epoch")
            wandb.define_metric(f"{gene_name}/validation_loss", step_metric="epoch/epoch")
            log_entry = {
                f"{gene_name}/validation_accuracy_at_epoch": accuracy_score,
                f"{gene_name}/validation_error_at_epoch": incorrect_score,
                f"{gene_name}/validation_loss_at_epoch": gene_loss.item(),
                f"epoch/epoch": epoch
            }
            wandb.log(log_entry)

        eval_logfile.write(f"{epoch},{os.path.basename(gene).split('.')[0]},{accuracy_score},{incorrect_score},{gene_loss.item()},{' '.join(predicted_label_token)},{' '.join(target_label_token)}\n")

        # After each gene is passed, hidden state and cell state are reset.
        if model.seqlab_head.lstm:
            model.seqlab_head.lstm.reset_hidden()

    #endfor
    eval_logfile.close()
    n_eval_genes = len(eval_genes)
    avg_accuracy_score = accuracy_score_sum / n_eval_genes
    avg_incorrect_score = incorrect_score_sum / n_eval_genes
    avg_gene_loss_score = gene_loss_sum / n_eval_genes

    return avg_accuracy_score, avg_incorrect_score, avg_gene_loss_score

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

    n_gpu = len(device_list)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_list)
    
    scaler = GradScaler()

    # Initialize log.
    logfile = open(log_file_path, "x")
    logfile.write("epoch,gene,gene_loss,epoch_loss\n")

    num_training_genes = len(train_genes)
    best_accuracy = 0

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = None
        wandb.define_metric("epoch/epoch")
        # for i in range(num_training_genes):
        for i in tqdm(range(num_training_genes), desc=f"Training Epoch {epoch + 1}/{num_epoch}", total=num_training_genes):
            
            gene = train_genes[i]
            gene_name = os.path.basename(gene).split(".")[0]
            gene_dir = os.path.dirname(gene)
            gene_dir, gene_chr = os.path.split(gene_dir)
            gene_dataloader = preprocessing_kmer(gene, tokenizer, batch_size)

            # gene_loss = None # This is loss computed from single gene.
            gene_loss, predicted_label, target_label, scaler = __forward_gene_non_overlap__(model, gene_dataloader, device, loss_function=loss_function, wandb=wandb, gene_name=gene_name, scaler=scaler, epoch=epoch, num_epoch=num_epoch, mode="train")
            
            gene_loss = gene_loss / grad_accumulation_steps
            epoch_loss = gene_loss if epoch_loss == None else epoch_loss + gene_loss
            
            # Write gene training log.
            logfile.write(f"{epoch},{gene_chr}-{gene_name},{gene_loss.item()},{epoch_loss.item()}\n")

            # Record log in the cloud.
            # Record gene loss in this epoch.
            if wandb != None:
                wandb.define_metric(f"{gene_chr}-{gene_name}/epoch_loss", step_metric="epoch/epoch")
                log_entry = {
                    f"{gene_chr}-{gene_name}/epoch_loss": gene_loss.item(),
                    "epoch/epoch": epoch
                }
                wandb.log(log_entry)

            # If model uses LSTM, reset hidden state and cell state if a gene has been processed.
            _model = model
            if isinstance(model, torch.nn.DataParallel):
                _model = model.module
            if _model.seqlab_head.lstm:
                _model.seqlab_head.lstm.reset_hidden()

            # Gradient is cleared after a gene has been processed.
            # Optimizer is reset after a gene is finised.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        #endfor

        # Record epoch loss.
        # Epoch loss is accumulation of all gene losses.
        if wandb != None:
            wandb.define_metric("epoch/training_loss", step_metric="epoch/epoch")
            wandb.log({"epoch/training_loss": epoch_loss.item(), "epoch/epoch": epoch})
            

        # Eval model if eval_genes is available.
        if eval_genes:
            eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
            avg_accuracy, avg_inaccuracy, avg_gene_loss = evaluate_genes(model, eval_genes, device, eval_log, epoch, num_epoch, loss_function, wandb)

            if wandb:
                wandb.define_metric("epoch/average_accuracy", step_metric="epoch/epoch")
                wandb.define_metric("epoch/average_inaccuracy", step_metric="epoch/epoch")
                wandb.define_metric("epoch/average_gene_loss", step_metric="epoch/epoch")
                validation_log = {
                    "epoch/average_accuracy": avg_accuracy,
                    "epoch/average_inaccuracy": avg_inaccuracy,
                    "epoch/average_gene_loss": avg_gene_loss,
                    "epoch/epoch": epoch
                }
                wandb.log(validation_log)

            # Save trained model if this epoch produces better model.
            if avg_accuracy > best_accuracy:
                save_checkpoint(model, optimizer, {
                    "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                    "epoch": (i + training_counter),
                    "batch_size": batch_size,
                }, os.path.join(save_path, f"checkpoint-{epoch + training_counter}.pth"))

                old_model_path = os.path.join(save_path, f"checkpoint-{epoch + training_counter - 1}.pth")
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

        torch.cuda.empty_cache()
    #endfor
    logfile.close()
    return model, optimizer