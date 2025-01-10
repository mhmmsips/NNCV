"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToTensor,
)

from unet import UNet

def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation data split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    transform = Compose([
        ToTensor(), 
        Normalize((0.5,), (0.5,)),
    ])

    # Load the dataset and make a split for training and validation
    dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    train_dataset, valid_dataset = random_split(
        dataset, 
        [int((1-args.val_split)*len(dataset)), int(args.val_split*len(dataset))],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the model
    model = UNet(
        in_channels=3,  # RGB images
        out_channels=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = SGD(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for (images, labels) in valid_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(), 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(), 
        f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
