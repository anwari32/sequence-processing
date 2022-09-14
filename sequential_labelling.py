import torch
import os
from tqdm import tqdm
from utils.metrics import Metrics
from utils.utils import save_checkpoint
from utils.seqlab import NUM_LABELS, Label_Dictionary, Index_Dictionary
from torch.cuda.amp import autocast, GradScaler
import wandb
from models.seqlab import DNABERT_SL
import numpy as np

def evaluate_sequences(model, eval_dataloader, device, save_dir, epoch, num_epoch, loss_fn, wandb):
    for label_index in range(NUM_LABELS):
        label =Index_Dictionary[label_index]
        wandb.define_metric(f"validation/precision-{label}", step_metric="epoch")
        wandb.define_metric(f"validation/recall-{label}", step_metric="epoch")

    model.eval()
    avg_accuracy = 0
    avg_loss = 0
    eval_log = os.path.join(save_dir, f"validation_log.{epoch}.csv")
    eval_log_file = None
    if os.path.exists(eval_log):
        eval_log_file = open(eval_log, "a")
    else:
        eval_log_file = open(eval_log, "x")
        eval_log_file.write("epoch,step,accuracy,loss,prediction,target\n")
    y_pred = []
    y_target = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"Evaluating {epoch + 1}/{num_epoch}"): 
            input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)
            predictions, bert_output = model(input_ids, attention_mask)
            batch_loss = 0
            batch_accuracy = 0
            for pred, label in zip(predictions, batch_labels):
                loss = loss_fn(pred, label)
                batch_loss += loss
                accuracy = 0
                pscores, pindices = torch.max(pred, 1)
                pindices_str = [str(a) for a in pindices.tolist()]
                label_str = [str(a) for a in label.tolist()]
                for idx, lab in zip(pindices_str, label_str):
                    if idx == lab:
                        accuracy += 1
                accuracy = accuracy / predictions.shape[1] * 100
                batch_accuracy += accuracy
                eval_log_file.write(f"{epoch},{step},{accuracy},{loss.item()},{' '.join(pindices_str)},{' '.join(label_str)}\n")
                
                pindices = pindices.tolist()[1:] # Remove CLS
                label_indices = label.tolist()[1:] # Remove CLS
                filtered_target = [a for a in label_indices if a >= 0] # Remove special tokens.
                filtered_pred = pindices[0:len(filtered_target)] # Remove special tokens.
                
                y_pred = np.concatenate((y_pred, filtered_pred))
                y_target = np.concatenate((y_target, filtered_target))

            avg_accuracy = batch_accuracy / predictions.shape[0]
            avg_loss = batch_loss / predictions.shape[0]

    metrics = Metrics(y_pred, y_target)
    metrics.calculate()
    for label_index in range(NUM_LABELS):
        label = Index_Dictionary[label_index]
        precision = metrics.precision(label_index, percentage=True)
        recall = metrics.recall(label_index, percentage=True)
        wandb.log({
            f"validation/precision-{label}": precision,
            f"validation/recall-{label}": recall,
            "epoch": epoch
        })

    if eval_log_file != None:
        eval_log_file.close()
    wandb.save(eval_log)
        
    return avg_accuracy, avg_loss

def train(model: DNABERT_SL, optimizer, scheduler, train_dataloader, epoch_size, save_dir, loss_function, device='cpu', wandb=None, device_list=[], eval_dataloader=None, training_counter=0):

    # Writing training log.
    log_path = os.path.join(save_dir, "log.csv")
    log_file = open(log_path, 'x')
    log_file.write('epoch,step,batch_loss,epoch_loss\n')

    model.to(device)

    n_gpu = len(device_list)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_list)

    TRAINING_EPOCH = "train/epoch"
    TRAINING_LOSS = "train/loss"
    TRAINING_EPOCH_LOSS = "train/epoch_loss"
    wandb.define_metric(TRAINING_EPOCH)
    wandb.define_metric(TRAINING_LOSS, step_metric=TRAINING_EPOCH)
    wandb.define_metric(TRAINING_EPOCH_LOSS, step_metric=TRAINING_EPOCH)

    VALIDATION_ACCURACY = "validation/accuracy"
    VALIDATION_LOSS = "validation/loss"
    VALIDATION_EPOCH = "validation/epoch"
    wandb.define_metric(VALIDATION_EPOCH)
    wandb.define_metric(VALIDATION_ACCURACY, step_metric=VALIDATION_EPOCH)
    wandb.define_metric(VALIDATION_LOSS, step_metric=VALIDATION_EPOCH)
    
    # Clean up previous training, if any.
    # torch.cuda.empty_cache()

    # Do training.
    best_accuracy = 0
    for epoch in range(training_counter, epoch_size):
        epoch_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch [{epoch + 1}/{epoch_size}]"):
            optimizer.zero_grad()
            input_ids, attention_mask, input_type_ids, batch_labels = tuple(t.to(device) for t in batch)    
            with autocast():
                prediction, bert_output = model(input_ids, attention_mask)
                prediction_vals, prediction_indices = torch.max(prediction, 2)
                batch_loss = loss_function(prediction.view(-1, 8), batch_labels.view(-1))
            batch_loss.backward()
            epoch_loss += batch_loss
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            wandb.log({"batch_loss": batch_loss})
            log_file.write(f"{epoch},{step},{batch_loss.item()},{epoch_loss.item()},{lr}\n")

            y_prediction_at_step = []
            y_target_at_step = []
            for p, t in zip(prediction_indices, batch_labels):
                plist = p.tolist()
                tlist = t.tolist()
                plist = plist[1:] # remove CLS token.
                tlist = tlist[1:] # remove CLS token.
                tlist = [a for a in tlist if a >= 0] # remove special tokens.
                plist = plist[0:len(tlist)] # remove special tokens.
                y_prediction_at_step = np.concatenate((y_prediction_at_step, plist))
                y_target_at_step = np.concatenate((y_target_at_step, tlist))
            
            metrics_at_step = Metrics(y_prediction_at_step, y_target_at_step)
            metrics_at_step.calculate()
            for label_index in range(NUM_LABELS):
                label = Index_Dictionary[label_index]
                wandb.log({f"precision-{label}": metrics_at_step.precision(label_index)})
                wandb.log({f"recall-{label}": metrics_at_step.recall(label_index)})
        
        # Move scheduler to epoch loop.
        scheduler.step()
        wandb.log({
            TRAINING_EPOCH_LOSS: epoch_loss,
            TRAINING_EPOCH: epoch
        })

        # After an epoch, evaluate.
        if eval_dataloader != None:
            model.eval()
            avg_accuracy, avg_loss = evaluate_sequences(model, eval_dataloader, device, save_dir, epoch, epoch_size, loss_function, wandb)
            best_accuracy = best_accuracy if best_accuracy > avg_accuracy else avg_accuracy
            wandb.log({
                VALIDATION_ACCURACY: avg_accuracy,
                VALIDATION_LOSS: avg_loss.item(),
                VALIDATION_EPOCH: epoch
            })

            # Save trained model if this epoch produces better model.
            # EDIT 6 June 2022: Save for each epoch. 
            # EDIT 21 August 2022: Save just the latest epoch to save disk space.
            _model = model.module if isinstance(model, torch.nn.DataParallel) else model
            latest_dir = os.path.join(save_dir, "latest")
            latest_model = os.path.join(latest_dir, "checkpoint.pth")
            if not os.path.exists(latest_dir):
                os.makedirs(latest_dir, exist_ok=True)
            torch.save({
                "model": _model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "accuracy": avg_accuracy,
                "run_id": wandb.run.id
            }, latest_model)
            wandb.save(latest_dir)
        
        # torch.cuda.empty_cache()
    #endfor epoch
    log_file.close()
    return model, optimizer, scheduler

if __name__ == "__main__":
    import getopt
    import sys
    from pathlib import Path, PureWindowsPath
    from transformers import BertForMaskedLM, BertTokenizer
    from utils.seqlab import preprocessing
    from torch.nn import CrossEntropyLoss

    opts, args = getopt.getopt(sys.argv[1:], "m:", ["mode="])
    outputs = {}
    for o, a in opts:
        if o in ["-m", "--mode"]:
            outputs["mode"] = str(a)

    if outputs["mode"] == "eval":        
        path = str(Path(PureWindowsPath("pretrained\\3-new-12w-0")))
        bert = BertForMaskedLM.from_pretrained(path).bert
        model = DNABERT_SL(bert, None)

        dataloader = preprocessing(
            str(Path(PureWindowsPath("workspace\\seqlab\\seqlab.strand-positive.kmer.stride-510.from-index\\sample.csv"))), 
            BertTokenizer.from_pretrained(path), 
            1,
            do_kmer=False)
        avg_acc, avg_loss = evaluate_sequences(
            model, 
            dataloader, 
            "cpu", 
            "seqlab_eval_log.csv", 
            0, 
            1, 
            CrossEntropyLoss(), 
            "sum")
        print(avg_acc, avg_loss)
        


