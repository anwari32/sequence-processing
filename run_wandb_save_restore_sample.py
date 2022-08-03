import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import os
from getopt import getopt
import sys
from tqdm import tqdm

if __name__ == "__main__":

    opts, outputs = getopt(sys.argv[1:], "-r:", "--run-id=")
    args = {}
    for o, a in opts:
        if o in ["-r", "--run-id"]:
            args["run-id"] = a
        else:
            raise ValueError("Syntax not recognized")

    RUNNAME = args.get("run-id", False)
    PROJECT_NAME = 'pytorch-resume-run'
    SAVE_DIR = os.path.join("run", RUNNAME)
    # CHECKPOINT_PATH = './checkpoint.tar'
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest", "checkpoint.pth")
    N_EPOCHS = 100

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Dummy data
    X = torch.randn(64, 8, requires_grad=True)
    Y = torch.empty(64, 1).random_(2)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    metric = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoch = 0

    run_id = args.get("run-id", False)
    if not run_id:
        raise ValueError("Must provide run_id")
    run_id = f"{wandb.util.generate_id()}-{run_id}"
    run = wandb.init(project=PROJECT_NAME, resume=True, id=run_id)
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model.train()
    #while epoch < N_EPOCHS:
    for epoch in tqdm(range(0, N_EPOCHS), total=N_EPOCHS, desc="Training "):
        optimizer.zero_grad()
        output = model(X)
        loss = metric(output, Y)
        wandb.log({'loss': loss.item()}, step=epoch)
        loss.backward()
        optimizer.step()

        torch.save({ # Save our checkpoint loc
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, CHECKPOINT_PATH)
        wandb.save(CHECKPOINT_PATH) # saves checkpoint to wandb
    #    epoch += 1