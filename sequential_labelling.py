import torch
import os
from tqdm import tqdm
from utils.utils import save_checkpoint
import utils.seqlab
from torch.cuda.amp import autocast, GradScaler
import wandb
from models.seqlab import DNABERT_SL

def forward(model, batch_input_ids, batch_attn_mask, batch_labels, loss_function, device):
    # Make sure model and data are in the same device.
    model.to(device)
    batch_input_ids.to(device)
    batch_attn_mask.to(device)
    batch_labels.to(device)

    with autocast(enabled=True, cache_enabled=True):
        prediction, bert_output = model(batch_input_ids, batch_attn_mask)

        # Since loss function can only works without batch dimension, I need to loop the loss for each tokens in batch dimension.
        batch_loss = None
        #for pred, labels in zip(prediction, batch_labels):
        #    loss = loss_function(pred, labels)
        #    if batch_loss == None:
        #        batch_loss = loss
        #    else:
        #        batch_loss += loss
        #if loss_strategy in ["average", "avg"]:
        #    batch_loss = batch_loss/batch_input_ids.shape[0]
        num_labels = utils.seqlab.NUM_LABELS
        #if isinstance(model, torch.nn.DataParallel):
        #    num_labels = model.module.seqlab_head.num_labels
        #else:
        #    num_labels = model.seqlab.num_labels
        batch_loss = loss_function(prediction.view(-1, num_labels), batch_labels.view(-1))
    return batch_loss

def evaluate_sequences(model, eval_dataloader, device, eval_log, epoch, num_epoch, loss_fn, wandb=None):

    model.eval()
    avg_accuracy = 0
    avg_loss = 0
    eval_log_file = None
    if os.path.exists(eval_log):
        eval_log_file = open(eval_log, "a")
    else:
        eval_log_file = open(eval_log, "x")
        eval_log_file.write("epoch,step,accuracy,loss,prediction,target\n")
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
                #print(pscores)
                #print(pindices)
                #print(label)
                pindices_str = [str(a) for a in pindices.tolist()]
                label_str = [str(a) for a in label.tolist()]
                for idx, lab in zip(pindices_str, label_str):
                    if idx == lab:
                        accuracy += 1
                accuracy = accuracy / predictions.shape[1] * 100
                batch_accuracy += accuracy
                eval_log_file.write(f"{epoch},{step},{accuracy},{loss.item()},{' '.join(pindices_str)},{' '.join(label_str)}\n")
            avg_accuracy = batch_accuracy / predictions.shape[0]
            avg_loss = batch_loss / predictions.shape[0]

    if eval_log_file != None:
        eval_log_file.close()
        
    return avg_accuracy, avg_loss

def train(model: DNABERT_SL, optimizer, scheduler, train_dataloader, epoch_size, save_dir, loss_function, device='cpu', wandb=None, device_list=[], eval_dataloader=None, training_counter=0):

    # Writing training log.
    log_path = os.path.join(save_dir, "log.csv")
    log_file = open(log_path, 'x')
    log_file.write('epoch,step,batch_loss,epoch_loss,learning_rate\n')

    model.to(device)

    n_gpu = len(device_list)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_list)
        print(f"Main Device {device}")
        print(f"Device List {device_list}")
    else:
        print(f"Device {device}")

    scaler = GradScaler()

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
            input_ids, attention_mask, input_type_ids, label = tuple(t.to(device) for t in batch)    
            batch_loss = forward(model, input_ids, attention_mask, label, loss_function, device)
            lr = optimizer.param_groups[0]['lr']
            epoch_loss += batch_loss

            if scaler:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()                

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            wandb.log({"batch_loss": batch_loss})
            log_file.write(f"{epoch},{step},{batch_loss.item()},{epoch_loss.item()},{lr}\n")
        
        # Move scheduler to epoch loop.
        scheduler.step()
        wandb.log({
            TRAINING_EPOCH_LOSS: epoch_loss,
            TRAINING_EPOCH: epoch
        })

        # After an epoch, evaluate.
        if eval_dataloader != None:
            model.eval()
            eval_log = os.path.join(os.path.dirname(log_path), "eval_log.csv")
            avg_accuracy, avg_loss = evaluate_sequences(model, eval_dataloader, device, eval_log, epoch, epoch_size, loss_function, wandb)

            best_accuracy = best_accuracy if best_accuracy > avg_accuracy else avg_accuracy
            validation_log = {
                VALIDATION_ACCURACY: avg_accuracy,
                VALIDATION_LOSS: avg_loss.item(),
                VALIDATION_EPOCH: epoch
            }
            wandb.log(validation_log)

            # Save trained model if this epoch produces better model.
            # EDIT 6 June 2022: Save for each epoch. 
            _model = model.module if isinstance(model, torch.nn.DataParallel) else model
            save_checkpoint(_model, optimizer, scheduler, {
                "loss": epoch_loss.item(), # Take the value only, not whole tensor structure.
                "epoch": epoch,
                "avg_accuracy": avg_accuracy,
                "avg_loss": avg_loss.item(),
                "best_accuracy": best_accuracy,
            }, os.path.join(save_dir, f"checkpoint-{epoch}"))
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
        


