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


def __forward_sequence__(model, batch_input_ids, batch_attn_mask, batch_labels, loss_function, device, loss_strategy="sum"):
    # Make sure model and data are in the same device.
    model.to(device)
    batch_input_ids.to(device)
    batch_attn_mask.to(device)
    batch_labels.to(device)

    with autocast(enabled=True, cache_enabled=True):
        prediction = model(batch_input_ids, batch_attn_mask)

        # Since loss function can only works without batch dimension, I need to loop the loss for each tokens in batch dimension.
        batch_loss = None
        for pred, labels in zip(prediction, batch_labels):
            loss = loss_function(pred, labels)
            if batch_loss == None:
                batch_loss = loss
            else:
                batch_loss += loss
        if loss_strategy == "average":
            batch_loss = batch_loss/batch_input_ids.shape[0]
    return batch_loss

def __forward_gene_non_overlap__(model: DNABERTSeqLab, optimizer, scheduler, dataloader: DataLoader, device: str, loss_function=None, gene_name: str=None, scaler: GradScaler=None, wandb: wandb = None, mode: str = "train", epoch: int=0, num_epoch: int=0, grad_accumulation_steps: int=1):
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
        
        # EDIT 22 May 2022: wandb logging is done outside this function.
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
            if grad_accumulation_steps > 0:
                contig_loss = contig_loss / grad_accumulation_steps

            if scaler:
                scaler.scale(contig_loss).backward()
            else:
                contig_loss.backward()                
            
            if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # scaler.step(optimizer)
                # scaler.update()
                # scheduler.step()
                optimizer.zero_grad()


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

def __eval_gene__(model, dataloader, device, loss_fn, gene_name: str = None, wandb: wandb = None, at_epoch: int = 0, num_epoch: int = 0):
    model.to(device)
    model.eval()
    correct_label, incorrect_label = 0, 0
    predicted_label_token, target_label_token = [], []

    with torch.no_grad():
        gene_loss, predicted_label_tensor, target_label_tensor, scaler = __forward_gene_non_overlap__(model, None, None, dataloader, device, loss_fn, gene_name=gene_name, mode="validation", wandb=wandb, epoch=at_epoch, num_epoch=num_epoch)
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
        
    for gene in tqdm(eval_genes, desc=f"Validating Epoch {epoch + 1}/{num_epoch}", total=len(eval_genes)):
        gene_name = os.path.basename(gene).split('.')[0]
        gene_dir = os.path.dirname(gene)
        gene_dir, gene_chr = os.path.split(gene_dir)
        dataloader = preprocessing_kmer(gene, get_default_tokenizer(), 1)
        accuracy_score, incorrect_score, predicted_label_token, target_label_token, gene_loss = __eval_gene__(model, dataloader, device, loss_fn, gene_name=gene_name, wandb=wandb, at_epoch=epoch, num_epoch=num_epoch)
        accuracy_score_sum += accuracy_score
        incorrect_score_sum += incorrect_score
        gene_loss_sum += gene_loss.item()

        # EDIT 15 May 2022: Remove details for each gene since everything can be seen from eval log.
        # Log accuracy and incorrect score for each gene after an epoch.
        #if wandb != None:
        #    wandb.define_metric("validation/epoch")
        #    wandb.define_metric(f"validation/{gene_chr}-{gene_name}/accuracy", step_metric="validation/epoch")
        #    wandb.define_metric(f"validation/{gene_chr}-{gene_name}/error", step_metric="validation/epoch")
        #    wandb.define_metric(f"validation/{gene_chr}-{gene_name}/loss", step_metric="validation/epoch")
        #    log_entry = {
        #        f"validation/{gene_chr}-{gene_name}/accuracy": accuracy_score,
        #        f"validation/{gene_chr}-{gene_name}/error": incorrect_score,
        #        f"validation/{gene_chr}-{gene_name}/loss": gene_loss.item(),
        #        f"validation/epoch": epoch
        #    }
        #    wandb.log(log_entry)

        eval_logfile.write(f"{epoch},{gene_chr}-{gene_name},{accuracy_score},{incorrect_score},{gene_loss.item()},{' '.join(predicted_label_token)},{' '.join(target_label_token)}\n")

        # After each gene is passed, hidden state and cell state are reset.
        if model.seqlab_head.lstm:
            model.seqlab_head.lstm.reset_hidden()

    #endfor
    eval_logfile.close()
    n_eval_genes = len(eval_genes)
    avg_accuracy_score = accuracy_score_sum / n_eval_genes # Average accuracy over all genes.
    avg_incorrect_score = incorrect_score_sum / n_eval_genes # Average inaccuracy over all genes.
    avg_gene_loss_score = gene_loss_sum / n_eval_genes # Average loss over all genes.

    return avg_accuracy_score, avg_incorrect_score, avg_gene_loss_score

def evaluate_sequences(model, eval_dataloader, device, eval_log, epoch, num_epoch, loss_fn, loss_strategy, wandb=None):
    if wandb != None:
        VALIDATION_EPOCH = "validation/epoch"
        VALIDATION_LOSS = "validation/loss"
        VALIDATION_ACCURACY = "validation/accuracy"

        wandb.define_metric(VALIDATION_EPOCH)
        wandb.define_metric(VALIDATION_LOSS, step_metric=VALIDATION_EPOCH)
        wandb.define_metric(VALIDATION_ACCURACY, step_metric=VALIDATION_EPOCH)

    model.eval()
    avg_accuracy = 0
    avg_loss = 0
    eval_log_file = None
    if os.path.exists(eval_log):
        eval_log_file = open(eval_log, "a")
    else:
        eval_log_file = open(eval_log, "x")
        eval_log_file.write("epoch,step,accuracy,loss,prediction,target")
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"Evaluating {epoch + 1}/{num_epoch}"):
            input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
            predictions = model(input_ids, attention_mask)
            batch_loss = 0
            batch_accuracy = 0
            for pred, label in zip(predictions, batch_labels):
                loss = loss_fn(pred, label)
                batch_loss += loss
                accuracy = 0
                pscores, pindices = torch.max(pred, 1)
                for idx, lab in zip(pindices, label):
                    accuracy = accuracy + 1 if idx == lab else 0
                accuracy = accuracy / predictions.shape[1]
                batch_accuracy += accuracy
            avg_accuracy = batch_accuracy / predictions.shape[0]
            avg_loss = batch_loss / predictions.shape[0]

    if eval_log_file != None:
        eval_log_file.close()
        
    return avg_accuracy, avg_loss

def train_by_sequences(model, optimizer, scheduler, train_dataloader, epoch_size, log_path, save_model_path, device='cpu', training_counter=0, grad_accumulation_steps=1, loss_function=NLLLoss(), loss_strategy="sum", wandb=None, device_list=[], eval_dataloader=None):
    
    # Writing training log.
    log_file = open(log_path, 'x')
    log_file.write('epoch,step,batch_loss,epoch_loss,learning_rate\n')

    n_gpu = len(device_list)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_list)
    else:
        print(f"Device {device}")

    scaler = GradScaler()

    if wandb != None:
        TRAINING_EPOCH = "train/epoch"
        TRAINING_LOSS = "train/loss"

        wandb.define_metric(TRAINING_EPOCH)
        wandb.define_metric(TRAINING_LOSS, step_metric=TRAINING_EPOCH)

        VALIDATION_ACCURACY = "validation/accuracy"
        VALIDATION_LOSS = "validation/loss"
        VALIDATION_EPOCH = "validation/epoch"

        wandb.define_metric(VALIDATION_EPOCH)
        wandb.define_metric(VALIDATION_ACCURACY, step_metric=VALIDATION_EPOCH)
        wandb.define_metric(VALIDATION_LOSS, step_metric=VALIDATION_EPOCH)

    # Do training.
    model.to(device)
    model.train()
    best_accuracy = 0
    for epoch in range(epoch_size):
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch [{epoch + 1 + training_counter}/{epoch_size}]"):
            input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)    
            batch_loss = __forward_sequence__(model, input_ids, attention_mask, label, loss_function, device, loss_strategy)
            lr = optimizer.param_groups[0]['lr']
            batch_loss = (batch_loss / grad_accumulation_steps)
            epoch_loss += batch_loss

            if scaler:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()                

            if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if wandb != None:
                wandb.log({"epoch_loss": epoch_loss})
                wandb.log({"batch_loss": batch_loss})

                # Optional
                wandb.watch(model)

            log_file.write(f"{epoch + training_counter},{step},{batch_loss.item()},{epoch_loss.item()},{lr}\n")
        
        # After an epoch, evaluate.
        if eval_dataloader != None:
            eval_log = os.path.join(os.path.dirname(log_path), "eval_log.csv")
            # avg_accuracy, avg_inaccuracy, avg_gene_loss = evaluate_genes(model, eval_genes, device, eval_log, epoch, num_epoch, loss_function, wandb)
            avg_accuracy, avg_loss = evaluate_sequences(model, eval_dataloader, device, eval_log, epoch, epoch_size, loss_function, wandb)

            if wandb != None:
                validation_log = {
                    VALIDATION_ACCURACY: avg_accuracy,
                    VALIDATION_LOSS: avg_loss,
                    VALIDATION_EPOCH: epoch
                }
                wandb.log(validation_log)

            # Save trained model if this epoch produces better model.
            if avg_accuracy > best_accuracy:
                save_checkpoint(model, optimizer, {
                    "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                    "epoch": (epoch + training_counter),
                }, os.path.join(save_model_path, f"checkpoint-{epoch + training_counter}.pth"))

                # Had to save BERT layer separately because unknown error miskey match.
                current_bert_layer = model.bert
                current_bert_layer.save_pretrained(save_model_path)

                old_model_path = os.path.join(save_model_path, f"checkpoint-{epoch + training_counter - 1}.pth")
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
        
        torch.cuda.empty_cache()
    #endfor epoch
    log_file.close()
    return model, optimizer

def train_by_genes(model: DNABERTSeqLab, tokenizer: BertTokenizer, optimizer, scheduler, train_genes: list, loss_function, num_epoch=1, batch_size=1, grad_accumulation_steps=1, device="cpu", save_path=None, log_file_path=None, training_counter=0, wandb=None, eval_genes=None, device_list=[]):
    n_gpu = len(device_list)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_list)
    
    scaler = GradScaler()

    # Initialize log.
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

    if wandb != None:
        wandb.define_metric(TRAINING_EPOCH)
        wandb.define_metric(TRAINING_LOSS, step_metric=TRAINING_EPOCH)
        wandb.define_metric(TRAINING_AVG_LOSS, step_metric=TRAINING_EPOCH)
        wandb.define_metric(TRAINING_LR, step_metric=TRAINING_LOSS)

        wandb.define_metric(VALIDATION_EPOCH)
        wandb.define_metric(VALIDATION_AVG_ACC, step_metric=VALIDATION_EPOCH) # Avaerage accuracy.
        wandb.define_metric(VALIDATION_AVG_INACC, step_metric=VALIDATION_EPOCH) # Average inaccuracy.
        wandb.define_metric(VALIDATION_AVG_LOSS, step_metric=VALIDATION_EPOCH) # Average gene loss.

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = None

        lr = 0 # Learning rate.
        for i in tqdm(range(num_training_genes), desc=f"Training Epoch {epoch + 1}/{num_epoch}", total=num_training_genes):
            
            gene = train_genes[i]
            gene_name = os.path.basename(gene).split(".")[0]
            gene_dir = os.path.dirname(gene)
            gene_dir, gene_chr = os.path.split(gene_dir)
            gene_dataloader = preprocessing_kmer(gene, tokenizer, batch_size)

            # gene_loss = None # This is loss computed from single gene.
            gene_loss, predicted_label, target_label, scaler = __forward_gene_non_overlap__(model, optimizer, scheduler, gene_dataloader, device, loss_function=loss_function, wandb=wandb, gene_name=gene_name, scaler=scaler, epoch=epoch, num_epoch=num_epoch, mode="train", grad_accumulation_steps=grad_accumulation_steps)
            
            gene_loss = gene_loss
            epoch_loss = gene_loss if epoch_loss == None else epoch_loss + gene_loss

            # Get current learning rate and log it.
            lr = optimizer.param_groups[0]['lr']

            # Write gene training log.
            logfile.write(f"{epoch},{gene_chr}-{gene_name},{gene_loss.item()},{epoch_loss.item()},{lr}\n")            

            # Record log in the cloud.
            # Record gene loss in this epoch.
            # This will make a lot of charts.
            # if wandb != None:
            #    wandb.define_metric(f"{gene_chr}-{gene_name}/epoch_loss", step_metric="epoch/epoch")
            #    log_entry = {
            #        f"{gene_chr}-{gene_name}/epoch_loss": gene_loss.item(),
            #        "epoch/epoch": epoch,
            #        "learning_rate": lr,
            #    }
            #    wandb.log(log_entry)

            # If model uses LSTM, reset hidden state and cell state if a gene has been processed.
            _model = model
            if isinstance(model, torch.nn.DataParallel):
                _model = model.module
            if _model.seqlab_head.lstm:
                _model.seqlab_head.lstm.reset_hidden()

            # Gradient clipping. Max grad is set to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Gradient is cleared after a gene has been processed.
            # Optimizer is reset after a gene is finised.
            # EDIT 11 May 2022: Moved gradient accumulation and clearance at forward function.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # optimizer.zero_grad()

            if wandb != None:
                wandb.log({
                    TRAINING_LR: lr,
                    TRAINING_EPOCH: epoch
                })

        #endfor

        # Record epoch loss.
        # Epoch loss is accumulation of all gene losses.
        if wandb != None:
            wandb.log({
                TRAINING_LOSS: epoch_loss.item(), 
                TRAINING_AVG_LOSS: epoch_loss.item() / num_training_genes,
                TRAINING_EPOCH: epoch
            })            

        # Eval model if eval_genes is available.
        if eval_genes:
            eval_log = os.path.join(os.path.dirname(log_file_path), "eval_log.csv")
            avg_accuracy, avg_inaccuracy, avg_gene_loss = evaluate_genes(model, eval_genes, device, eval_log, epoch, num_epoch, loss_function, wandb)

            if wandb != None:
                validation_log = {
                    VALIDATION_AVG_ACC: avg_accuracy,
                    VALIDATION_AVG_INACC: avg_inaccuracy,
                    VALIDATION_AVG_LOSS: avg_gene_loss,
                    VALIDATION_EPOCH: epoch
                }
                wandb.log(validation_log)

            # Save trained model if this epoch produces better model.
            if avg_accuracy > best_accuracy:
                _model = model
                if isinstance(model, torch.nn.DataParallel):
                    _model = model.module

                torch.save()
                torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer.pth"))
                save_checkpoint(_model, optimizer, {
                    "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                    "epoch": (i + training_counter),
                    "batch_size": batch_size,
                }, os.path.join(save_path, f"checkpoint-{epoch + training_counter}.pth"))

                # Had to save BERT layer separately because unknown error miskey match.
                current_bert_layer = _model.bert
                current_bert_layer.save_pretrained(save_path)

                old_model_path = os.path.join(save_path, f"checkpoint-{epoch + training_counter - 1}.pth")
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

        torch.cuda.empty_cache()
    #endfor
    logfile.close()
    return model, optimizer