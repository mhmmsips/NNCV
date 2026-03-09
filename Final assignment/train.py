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
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
#NOTE train_id == 255 is used for the 'void' category, which is ignored during training and evaluation. We assign it a color (black) for visualization purposes.
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


# Submission platform metrics are reported on these Cityscapes super-categories.
# Mapping found on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py and https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
category_names = ("Flat",
                  "Constrution",
                  "Object", 
                  "Nature",
                  "Sky",
                  "Human",
                  "Vehicle")

train_id_to_category_index = torch.full((256,), -1, dtype=torch.long)
train_id_to_category_index[0] = 0 # road
train_id_to_category_index[1] = 0 # sidewalk
train_id_to_category_index[2] = 1 # building
train_id_to_category_index[3] = 1 # wall
train_id_to_category_index[4] = 1 # fence
train_id_to_category_index[5] = 2 # pole
train_id_to_category_index[6] = 2 # traffic light
train_id_to_category_index[7] = 2 # traffic sign
train_id_to_category_index[8] = 3 # vegetation
train_id_to_category_index[9] = 3 # terrain
train_id_to_category_index[10] = 4 # sky
train_id_to_category_index[11] = 5 # person
train_id_to_category_index[12] = 5 # rider
train_id_to_category_index[13] = 6 # car
train_id_to_category_index[14] = 6 # truck
train_id_to_category_index[15] = 6 # bus
train_id_to_category_index[16] = 6 # train
train_id_to_category_index[17] = 6 # motorcycle
train_id_to_category_index[18] = 6 # bicycle


def update_category_confusion_matrix(confusion_matrix: torch.Tensor,
                                     predictions: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
    """
    Accumulate a confusion matrix over the 7 submission categories.

    Both `predictions` and `labels` are expected to contain Cityscapes train IDs.
    They are mapped to the coarse submission categories defined in `train_id_to_category_index`, and only valid labels are counted.

    Args:
        confusion_matrix: Running `(7, 7)` confusion matrix with rows for ground truth categories and columns for predicted categories.
        predictions: Predicted segmentation map of shape `(B, H, W)` with train IDs.
        labels: Ground-truth segmentation map of shape `(B, H, W)` with train IDs.

    Returns:
        The updated confusion matrix.
    """
    
    # Set the device of the category lookup to match the predictions and labels
    category_lookup = train_id_to_category_index.to(predictions.device)
    
    # Extract the category indices for predictions and labels using the lookup table
    predicted_categories = category_lookup[predictions]
    label_categories = category_lookup[labels]

    # Ignore pixels whose ground-truth label does not belong to one of the 7 evaluated submission categories.
    #NOTE: When choosing "out of distribution", see if this assumption still holds
    valid_mask = label_categories >= 0
    predicted_categories = predicted_categories[valid_mask]
    label_categories = label_categories[valid_mask]

    # Flatten each (ground_truth, prediction) pair into a single index, where `bincount` can accumulate the full confusion matrix efficiently.
    confusion_matrix += torch.bincount(label_categories * len(category_names) + predicted_categories,
                                       minlength=len(category_names) ** 2).reshape(len(category_names), len(category_names))
    
    # Return the CM
    return confusion_matrix


def compute_category_metrics(confusion_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute dataset-level Dice and IoU scores from a confusion matrix.

    The confusion matrix is assumed to contain counts over the 7 submission categories. 
    True positives, false positives, and false negatives are derived from it to produce the mean metrics and the per-category metrics logged to W&B.

    Args:
        confusion_matrix: `(7, 7)` confusion matrix with rows as ground truth and columns as predictions.

    Returns:
        A dictionary containing `MeanDice`, `MeanIoU`, and all per-category `Dice*` and `IoU*` metrics.
    """
    # Convert the confusion matrix to float for metric calculations
    confusion_matrix = confusion_matrix.float()

    # Derive TP, FP, and FN from the confusion matrix
    true_positive = confusion_matrix.diag()
    false_positive = confusion_matrix.sum(dim=0) - true_positive
    false_negative = confusion_matrix.sum(dim=1) - true_positive

    # Calculate the DICE and IoU
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice_scores = torch.zeros_like(true_positive)
    valid_dice = dice_denominator > 0 # Ensure we only compute DICE for categories that are present in the ground truth or predictions
    dice_scores[valid_dice] = (2 * true_positive[valid_dice]) / dice_denominator[valid_dice]
    
    iou_denominator = true_positive + false_positive + false_negative
    iou_scores = torch.zeros_like(true_positive)
    valid_iou = iou_denominator > 0
    iou_scores[valid_iou] = true_positive[valid_iou] / iou_denominator[valid_iou]
    
    # Log the mean DICE and IoU over all categories
    metrics = {"MeanDice": dice_scores[valid_dice].mean().item() if valid_dice.any() else 0.0,
               "MeanIoU": iou_scores[valid_iou].mean().item() if valid_iou.any() else 0.0,}

    # Log per-category metrics, using the category names defined in `category_names`
    for index, category_name in enumerate(category_names):
        metrics[f"Dice{category_name}"] = dice_scores[index].item()
        metrics[f"IoU{category_name}"] = iou_scores[index].item()

    # Return the metrics dictionary, which will be logged to W&B in the training loop
    return metrics


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    img_transform = Compose([
    ToImage(),
    # Resize((256, 256)),
    Resize((252, 252)) #ENSURE DIVISIBILITY BY PATCH SIZE
    ToDtype(torch.float32, scale=True),
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #NOTE: (CHECK IF TRUE) DINOv2 was trained with ImageNet normalization, so we use those mean and std values here 
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        # Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        Resize((252, 252), interpolation=InterpolationMode.NEAREST), #ENSURE DIVISIBILITY BY PATCH SIZE
        ToDtype(torch.int64),  # no scaling
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
    args.data_dir,
    split="train",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)
    
    # Freeze the backbone parameters to prevent them from being updated during training, since we are using a pretrained DINOv2 model as the backbone and only want to train the decoder
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    # optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

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
            # Initialize losses list and confusion matrix for the validation set
            losses = []
            category_confusion_matrix = torch.zeros((len(category_names), len(category_names)),
                                                    dtype=torch.int64, device=device)
            
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                # Convert the predicted segmentation maps to category predictions and update the confusion matrix
                predictions = outputs.softmax(1).argmax(1)
                category_confusion_matrix = update_category_confusion_matrix(category_confusion_matrix,
                                                                             predictions,
                                                                             labels)
            
                if i == 0:
                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            
            # Compute the metrics from the confusion matrix and log them to W&B, along with the validation loss
            validation_metrics = compute_category_metrics(category_confusion_matrix)
            validation_metrics["valid_loss"] = valid_loss
            wandb.log(validation_metrics, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
