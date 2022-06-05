import torch
import os
from tqdm import tqdm
from utils.utils import save_checkpoint
from torch.cuda.amp import autocast, GradScaler
import wandb

def forward(model, batch_input_ids, batch_attn_mask, batch_labels, loss_function, device, loss_strategy="sum"):
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

def train(model, optimizer, scheduler, train_dataloader, epoch_size, save_dir, loss_function, device='cpu', loss_strategy="sum", wandb=None, device_list=[], eval_dataloader=None):
    
    # Writing training log.
    log_path = os.path.join(save_dir, "log.csv")
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
            batch_loss = forward(model, input_ids, attention_mask, label, loss_function, device, loss_strategy)
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
