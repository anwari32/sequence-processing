import torch

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, batch_size, num_epochs, save_dir, wandb):
    