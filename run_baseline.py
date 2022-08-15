import torch
import sys
from utils.cli import parse_args

def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, batch_size, num_epochs, save_dir, wandb):
    raise NotImplementedError("Not yet implemented.")    

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    