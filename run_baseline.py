import json
import torch
import sys
import os
import wandb
from utils.cli import parse_args
from utils.utils import create_loss_weight
from models.baseline import Baseline
from utils.seqlab import preprocessing_kmer
from transformers import BertTokenizer

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, batch_size, num_epochs, device, save_dir, wandb, start_epoch=0, device_list=[], loss_weight=None):
    model.to(device)
    num_labels = model.num_labels
    if len(device_list) > 0:
        torch.nn.DataParallel(device_list)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "latest")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_weight:
        loss_weight = loss_weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask = tuple(t.to(device) for t in batch)
            with torch.amp.auto_grad():
                pred = model(input_ids)
                loss = criterion(pred.view(-1, num_labels), target_label.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            input_ids, attention_mask, target_label = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                pred = model(input_ids)
                loss = criterion(pred.view(-1, num_labels), target_label.view(-1))
        
        torch.save({
            "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }, checkpoint_dir)
            

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    training_config_path = args.get("training-config", False)
    device = args.get("device", False)
    resume_run_ids = args.get("resume-run-ids", False)
    model_config_dir = args.get("model-config-dir", False)
    model_config_names = args.get("model-config-names", False)
    batch_size = args.get("batch-size", 1)
    num_epochs = args.get("num-epochs", 1)
    run_name = args.get("run-name", "baseline")
    device_list = args.get("device-list", [])
    project_name = args.get("project-name", "baseline")
    use_weighted_loss = args.get("use-weighted-loss", False)

    model = Baseline()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.01)
    training_config = json.load(open(training_config_path, "r"))
    training_data_path = training_config.get("training_data", False)
    validation_data_path = training_config.get("validation_data", False)
    pretrained_path = training_config.get("pretrained", os.path.join("pretrained", "3-new-12w-0"))
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_dataloader, eval_dataloader = preprocessing_kmer(training_data_path, tokenizer, batch_size), preprocessing_kmer(validation_data_path, tokenizer, batch_size)
    loss_weight = create_loss_weight(training_data_path) if use_weighted_loss else None
    
    run_id = wandb.util.generate_id()
    runname = f"{run_name}-{run_id}"
    save_dir = os.path.join("run", runname)
    checkpoint_dir = os.path.join(save_dir, "latest")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    run = wandb.init(id=run_id, project_name=project_name, name=runname, reinit=True, resume="allow")
    start_epoch = 0
    if run.resumed:
        checkpoint = torch.load(checkpoint_path)
        model = torch.load(checkpoint.get("model"))
        optimizer = torch.load(checkpoint.get("optimizer"))
        scheduler = torch.load(checkpoint.get("scheduler"))
        epoch = int(checkpoint.get("epoch"))
        start_epoch = epoch + 1

    train(model, 
        optimizer, 
        scheduler,
        train_dataloader,
        eval_dataloader,
        batch_size,
        num_epochs,
        device,
        save_dir,
        wandb,
        start_epoch,
        device_list,
        loss_weight)
