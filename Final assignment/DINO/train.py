"""
This script implements a training loop for the DINOv2-models. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.
"""
import os
import random
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW, SGD
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (Compose,
                                       Normalize,
                                       ToImage,
                                       ToDtype,
                                       Pad,
                                       RandomHorizontalFlip,
                                       ColorJitter,
                                       GaussianBlur,
                                       RandomApply)
import math


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
id_to_trainid[255] = 255  # Ignore pixels should stay as 255

# Mapping train IDs to color
#NOTE train_id == 255 is used for the 'void' category, which is ignored during training and evaluation. We assign it a color (black) for visualization purposes.
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

# Helper function
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8, device=prediction.device)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


# Metrics are computed per Cityscapes train class first, then averaged into the seven submission super-categories with an unweighted mean over the member classes.
class_names = ("Road",
               "Sidewalk",
               "Building",
               "Wall",
               "Fence",
               "Pole",
               "TrafficLight",
               "TrafficSign",
               "Vegetation",
               "Terrain",
               "Sky",
               "Person",
               "Rider",
               "Car",
               "Truck",
               "Bus",
               "Train",
               "Motorcycle",
               "Bicycle")

category_names = ("Flat",
                  "Construction",
                  "Object",
                  "Nature",
                  "Sky",
                  "Human",
                  "Vehicle")

# Mapping found on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py and https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
category_to_train_ids = ((0, 1),
                         (2, 3, 4),
                         (5, 6, 7),
                         (8, 9),
                         (10,),
                         (11, 12),
                         (13, 14, 15, 16, 17, 18))

# Define the safety-critical classes to receive higher weights in the safety-critical boundary loss
# 6 = traffic light, 7 = traffic sign, 11 = person, 12 = rider, 17 = motorcycle, 18 = bicycle
safety_critical_class_identifiers = (6, 7, 11, 12, 17, 18)


# Functions to calculate metrics
def update_class_confusion_matrix(confusion_matrix: torch.Tensor,
                                  predictions: torch.Tensor,
                                  labels: torch.Tensor) -> torch.Tensor:
    """
    Accumulate a confusion matrix over the 19 valid Cityscapes train classes.

    Args:
        confusion_matrix: Running `(19, 19)` confusion matrix with rows for ground truth classes and columns for predicted classes.
        predictions: Predicted segmentation map of shape `(B, H, W)` with train IDs.
        labels: Ground-truth segmentation map of shape `(B, H, W)` with train IDs.

    Returns:
        The updated confusion matrix.
    """
    valid_mask = labels != 255
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    confusion_matrix += torch.bincount(labels * len(class_names) + predictions,
                                       minlength=len(class_names) ** 2).reshape(len(class_names), len(class_names))
    return confusion_matrix


def compute_class_dice_scores(confusion_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class Dice scores from a `(19, 19)` confusion matrix.
    """
    confusion_matrix = confusion_matrix.float()

    true_positive = confusion_matrix.diag()
    false_positive = confusion_matrix.sum(dim=0) - true_positive
    false_negative = confusion_matrix.sum(dim=1) - true_positive

    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice_scores = torch.zeros_like(true_positive)
    valid_scores = dice_denominator > 0
    dice_scores[valid_scores] = (2 * true_positive[valid_scores]) / dice_denominator[valid_scores]

    return dice_scores, valid_scores


def aggregate_class_scores(scores: torch.Tensor,
                           valid_scores: torch.Tensor,
                           metric_prefix: str) -> dict[str, float]:
    """
    Aggregate per-class scores into the seven super-categories with an unweighted mean.
    """
    metrics = {}
    for category_name, train_ids in zip(category_names, category_to_train_ids):
        class_indices = torch.tensor(train_ids, device=scores.device)
        category_valid = valid_scores[class_indices]

        if category_valid.any():
            metrics[f"{metric_prefix}{category_name}"] = scores[class_indices][category_valid].mean().item()
        else:
            metrics[f"{metric_prefix}{category_name}"] = 0.0

    return metrics


def compute_dice_metrics(confusion_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute MeanDice over the 19 valid classes and Dice* super-category metrics as unweighted means over the member class Dice scores.
    """
    dice_scores, valid_scores = compute_class_dice_scores(confusion_matrix)

    metrics = {"MeanDice": dice_scores[valid_scores].mean().item() if valid_scores.any() else 0.0}
    metrics.update(aggregate_class_scores(dice_scores, valid_scores, metric_prefix="Dice"))
    return metrics


def get_boundary(mask: torch.Tensor, d: int) -> torch.Tensor:
    """
    Extract boundary pixels for a binary mask using morphological erosion.

    Args:
        mask: Binary mask of shape (H, W) or (B, H, W).
        d: Erosion distance controlling boundary thickness.

    Returns:
        Boundary mask with the same leading dimensions as `mask`.
    """
    # Ensure that the erosion distance is at least 1 to avoid invalid kernel sizes in max pooling
    d = max(1, int(d))

    # Ensure dimensions are correct for max pooling
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    # Convert the mask to float and add a channel dimension for max pooling
    mask = mask.to(dtype=torch.float32)
    mask = mask.unsqueeze(1)

    # Perform morphological erosion using max pooling and compute the boundary as the difference between the original mask and the eroded version
    eroded = 1.0 - F.max_pool2d(1.0 - mask,
                                kernel_size=2 * d + 1,
                                stride=1,
                                padding=d)
    
    # Compute boundary as the difference between the original mask and the eroded version
    boundary = (mask - eroded) > 0

    return boundary.squeeze(1)


def compute_boundary_iou_batch(predictions: torch.Tensor,
                               labels: torch.Tensor,
                               n_classes: int = 19) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class Boundary IoU intersections and unions for a batch.

    Boundary thickness is fixed for the whole batch and set to 0.5% of the image diagonal in pixels.

    Args:
        predictions: Predicted class map of shape (B, H, W).
        labels: Ground-truth class map of shape (B, H, W).
        n_classes: Number of valid train classes.

    Returns:
        Tuple(Tensors): (intersections, unions), each of shape (n_classes,).
    """
    # Set to device
    device = predictions.device

    # Initialize intersections and unions
    intersections = torch.zeros(n_classes, dtype=torch.float32, device=device)
    unions = torch.zeros(n_classes, dtype=torch.float32, device=device)
    valid_mask = labels != 255

    # Fixed boundary thickness d = 0.5% of the image diagonal in pixels
    _, height, width = predictions.shape
    image_diagonal = math.sqrt(height * height + width * width)
    d = max(1, int(0.005 * image_diagonal))

    # Loop over all classes
    for class_index in range(n_classes):
        pred_mask = predictions == class_index
        gt_mask = labels == class_index

        # Loop over the batch to compute boundary IoU for each image
        for batch_index in range(predictions.shape[0]):
            # Get prediction and ground-truth boundaries using morphological erosion
            pred_boundary = get_boundary(pred_mask[batch_index], d)
            gt_boundary = get_boundary(gt_mask[batch_index], d)

            # Get the valid pixels for the current batch index as a mask
            valid_pixels = valid_mask[batch_index]
            
            # Apply the valid mask to both boundaries to ensure that we only consider valid pixels in the IoU calculation
            pred_boundary = pred_boundary & valid_pixels
            gt_boundary = gt_boundary & valid_pixels

            intersections[class_index] += (pred_boundary & gt_boundary).sum()
            unions[class_index] += (pred_boundary | gt_boundary).sum()

    return intersections, unions


def compute_boundary_iou_metrics(intersections: torch.Tensor,
                                 unions: torch.Tensor) -> dict[str, float]:
    """
    Compute MeanBoundaryIoU over the 19 classes, plus safety-critical boundary
    
    IoU metrics for the classes emphasized by the safety-critical loss.
    """
    # Compute Boundary IoU per class, handling cases where the union is zero to avoid division by zero errors.
    boundary_iou = torch.zeros_like(intersections)
    valid_scores = unions > 0
    boundary_iou[valid_scores] = intersections[valid_scores] / unions[valid_scores]

    # Get the MeanBoundaryIoU over the 19 valid classes, and then compute the MeanSafetyCriticalBoundaryIoU as the mean over the safety-critical classes, 
    metrics = {"MeanBoundaryIoU": boundary_iou[valid_scores].mean().item() if valid_scores.any() else 0.0}
    safety_indices = torch.tensor(safety_critical_class_identifiers, device=boundary_iou.device)
    valid_safety_scores = valid_scores[safety_indices]
    metrics["MeanSafetyCriticalBoundaryIoU"] = (boundary_iou[safety_indices][valid_safety_scores].mean().item() if valid_safety_scores.any() else 0.0)

    # Also include the individual Boundary IoU scores for each safety-critical class for more detailed analysis of how the model is performing on those classes specifically
    for class_index in safety_critical_class_identifiers:
        metrics[f"BoundaryIoU{class_names[class_index]}"] = boundary_iou[class_index].item()

    return metrics


# Define a class to augment the Cityscapes dataset with random horizontal flips and do the other augmentations
class AugmentedCityscapes(torch.utils.data.Dataset):
    def __init__(self, dataset, augment=True):
        self.dataset = dataset
        self.augment = augment
        self.joint_transform = RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if self.augment:
            img, mask = self.joint_transform(img, mask)
        return img, mask
    
    
# Define classes for the combined CE + Dice loss
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = logits.softmax(dim=1)
        num_classes = logits.shape[1]

        # Mask out ignore index
        valid_mask = (targets != self.ignore_index).unsqueeze(1)

        # One-hot encode targets, replacing ignore_index with 0
        targets_clean = targets.clone()
        targets_clean[targets == self.ignore_index] = 0
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets_clean.unsqueeze(1), 1)

        # Apply valid mask
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = (probs + targets_one_hot).sum(dim=dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - dice_per_class.mean()


class BoundaryLoss(nn.Module):
    """
    Approximate boundary loss built from a signed boundary band around the ground-truth mask.
    
    band_width_ratio controls the width of the band as a percentage of the image diagonal, so it adapts to different image resolutions.
    Set to 0.5% for Cityscapes, according to the paper "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (arXiv:2205.12681)
    """
    def __init__(self, ignore_index=255, band_width_ratio=0.005, class_weights: torch.Tensor | None = None):
        super().__init__()
        
        # Store the ignore index and band width ratio as instance variables for use in the forward pass
        self.ignore_index = ignore_index
        self.band_width_ratio = band_width_ratio
        
        # Set class weights if provided
        if class_weights is not None:
            class_weights = class_weights.to(dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def _get_band_width(self, height: int, width: int) -> int:
        
        # Calculate the image diagonal in pixels and determine the band width based on the configured ratio
        image_diagonal = math.sqrt(height * height + width * width)
        return max(1, int(self.band_width_ratio * image_diagonal))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get the predicted probabilities from the logits and get valid masks for ignore_index
        probs = logits.softmax(dim=1)
        num_classes = logits.shape[1]
        valid_mask = (targets != self.ignore_index).unsqueeze(1)

        # One-hot encode targets, replacing ignore_index with 0 so they don't contribute to the band, and then apply the valid mask to ensure that ignore_index pixels do not contribute to the loss
        targets_clean = targets.clone() 
        targets_clean[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(targets_clean, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        targets_one_hot = targets_one_hot * valid_mask.to(dtype=probs.dtype)

        # Get band width based on the image size and apply morphological dilation and erosion to create an outer and inner band around the ground-truth mask.
        band_width = self._get_band_width(*targets.shape[-2:])
        dilated = F.max_pool2d(targets_one_hot,
                               kernel_size=2 * band_width + 1,
                               stride=1,
                               padding=band_width)
        
        eroded = 1.0 - F.max_pool2d(1.0 - targets_one_hot,
                                    kernel_size=2 * band_width + 1,
                                    stride=1,
                                    padding=band_width)

        # Get outer and inner bands by taking the difference between the dilated and eroded masks and the original mask, respectively.
        outer_band = (dilated - targets_one_hot).clamp_min(0.0) * valid_mask.to(dtype=probs.dtype)
        inner_band = (targets_one_hot - eroded).clamp_min(0.0) * valid_mask.to(dtype=probs.dtype)
        
        # Use a signed band so probabilities are encouraged on the inside boundary and discouraged just outside it.
        signed_band = outer_band - inner_band

        # Compute a weighted average of the signed band values at the predicted probabilities for each class, and normalize by the total weight of the band to get a per-class boundary score.
        # Then average over classes, applying class weights if provided, to get the final boundary loss.
        boundary_weight = signed_band.abs().sum(dim=(2, 3))
        boundary_score = (probs * signed_band).sum(dim=(2, 3))
        valid_classes = boundary_weight > 0

        if not valid_classes.any():
            return logits.new_zeros(())

        normalized_score = torch.zeros_like(boundary_score)
        normalized_score[valid_classes] = boundary_score[valid_classes] / boundary_weight[valid_classes]

        if self.class_weights is None:
            return normalized_score[valid_classes].mean()

        if self.class_weights.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, but got {self.class_weights.numel()}")

        class_weights = self.class_weights.to(device=logits.device, dtype=normalized_score.dtype).unsqueeze(0).expand_as(normalized_score)
        valid_class_weights = class_weights[valid_classes]
        return (normalized_score[valid_classes] * valid_class_weights).sum() / valid_class_weights.sum()


class RebalancedBoundaryLoss(nn.Module):
    """
    Regional CE + Dice loss with a boundary loss term inspired by Kervadec et al., “Boundary loss for highly unbalanced segmentation” (arXiv:1812.07032).
    Alpha is linearly rebalanced from 0.01 to 0.5 across the configured number of epochs so that the regional and boundary terms have equal weight at the final epoch.
    """
    def __init__(self,
                 ce_loss,
                 dice_loss,
                 boundary_loss,
                 ce_weight=1.0,
                 dice_weight=0.5,
                 alpha_start=0.01,
                 alpha_end=0.5,
                 num_epochs=10):
        
        super().__init__()
        self.ce = ce_loss
        self.dice = dice_loss
        self.boundary = boundary_loss
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.num_epochs = num_epochs

    def get_alpha(self, epoch: int) -> float:
        # Linearly rebalance alpha from alpha_start to alpha_end across the configured number of epochs
        alpha_min = min(self.alpha_start, self.alpha_end)
        alpha_max = max(self.alpha_start, self.alpha_end)
        
        if self.num_epochs > 1:
            alpha = self.alpha_start + (epoch / (self.num_epochs - 1)) * (self.alpha_end - self.alpha_start)
        else:
            alpha = self.alpha_end

        return float(min(max(alpha, alpha_min), alpha_max))

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                epoch: int,
                return_components: bool = False):
        
        # Get all individual loss components
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        # Calculate the combined loss with dynamic alpha rebalancing
        regional_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss 
        alpha = logits.new_tensor(self.get_alpha(epoch))
        total_loss = (1.0 - alpha) * regional_loss + alpha * boundary_loss

        if not return_components:
            return total_loss

        return total_loss, {"total_loss": total_loss.detach(),
                            "ce_loss": ce_loss.detach(),
                            "dice_loss": dice_loss.detach(),
                            "boundary_loss": boundary_loss.detach(),
                            "regional_loss": regional_loss.detach(),
                            "alpha": alpha.detach()}




class SafetyCriticalRebalancedBoundaryLoss(RebalancedBoundaryLoss):
    """
    Prior work has shown that semantic segmentation for autonomous driving should account for the unequal importance of object classes,
    with safety-critical objects such as pedestrians and traffic signs requiring higher accuracy than background classes (Chen et al., 2019).
    
    Title: Importance-Aware Semantic Segmentation for Autonomous Vehicles
    Authors: Chen et al. (full: Bike Chen, Chen Gong, and Jian Yang)
    Year: 2019
    Journal: IEEE Transactions on Intelligent Transportation Systems, vol. 20, no. 1, pp. 137–148

    Building on this idea, a class-dependent weighting is applied specifically to the boundary loss, 
    emphasizing accurate delineation of safety-critical objects, whose boundaries are particularly important for collision avoidance.
    
    #NOTE: The weights applied have been set to 2.0 (double) for safety-critical classes, but no sensitivity analysis has been done to find the optimal values. It would be interesting to experiment with different weights and see how they affect the performance on safety-critical classes and overall metrics.
    #NOTE: Also, there should be an ethical disclaimer about this in the report as the quantification of this weighting.
    """
    
    # Set to variable in the scope of the class
    safety_critical_class_ids = safety_critical_class_identifiers

    def __init__(self,
                 ce_loss,
                 dice_loss,
                 num_classes,
                 ignore_index=255,
                 band_width_ratio=0.005,
                 ce_weight=1.0,
                 dice_weight=0.5,
                 alpha_start=0.01,
                 alpha_end=0.5,
                 num_epochs=10):
        
        # Initialize class weights with 1.0 for all classes, and set higher weights for safety-critical classes
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for class_index in self.safety_critical_class_ids:
            if class_index >= num_classes:
                raise ValueError(f"Safety-critical class index {class_index} is out of range for num_classes={num_classes}")
            class_weights[class_index] = 2.0

        # Pass the class weights to the BoundaryLoss, which will use them to weight the boundary loss contribution from each class accordingly
        boundary_loss = BoundaryLoss(ignore_index=ignore_index,
                                     band_width_ratio=band_width_ratio,
                                     class_weights=class_weights)
        
        super().__init__(ce_loss=ce_loss,
                         dice_loss=dice_loss,
                         boundary_loss=boundary_loss,
                         ce_weight=ce_weight,
                         dice_weight=dice_weight,
                         alpha_start=alpha_start,
                         alpha_end=alpha_end,
                         num_epochs=num_epochs)
  

# Helper function to compute loss and return individual components for logging and RebalancedBoundaryLoss
def compute_loss_with_components(criterion, logits: torch.Tensor, targets: torch.Tensor, epoch: int):
    if isinstance(criterion, RebalancedBoundaryLoss):
        return criterion(logits, targets, epoch=epoch, return_components=True)

    loss = criterion(logits, targets)
    return loss, {"total_loss": loss.detach()}

# Seed worker function to ensure reproducibility in data loading with multiple workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)

    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def get_args_parser():

    parser = ArgumentParser("Training script for DINO-based semantic segmentation")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="DINOv2-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument("--decoder", type=str, default="linear", choices=["linear", "upsample", "mlp", "EoMT"], help="Type of decoder to use on top of the frozen DINOv2 backbone")

    return parser


def main(args):
    # Initialize accelerator for multi-GPU training, mixed precision and gradient accumulation
    accelerator = Accelerator(device_placement=False,
                              mixed_precision=args.mixed_precision if torch.cuda.is_available() else "no",
                              gradient_accumulation_steps=args.gradient_accumulation_steps)

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    if accelerator.is_main_process:
        wandb.init(project="5lsm0-cityscapes-segmentation",  # Project name in wandb
                   name=args.experiment_id,  # Experiment name in wandb
                   config=vars(args),  # Save hyperparameters
                   )
        
        os.makedirs(output_dir, exist_ok=True)
    
    # Wait for the main process to initialize W&B and create the output directory before other processes proceed    
    accelerator.wait_for_everyone()

    # Set seed for reproducability
    # This includes PyTorch, NumPy and Python's random module
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the device
    device = accelerator.device
    
    # Cityscapes images are 1024x2048, while DINOv2-L/14 uses a 14x14 patch size.
    # Pad to 1036x2058 so both spatial dimensions become divisible by 14 and no border pixels are dropped by the patch embedding.
    image_padding = (5, 6, 5, 6) # left, top, right, bottom

    # Training transform (with augmentations)
    train_img_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        Pad(padding=image_padding, fill=0),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #NOTE: (CHECK IF TRUE) DINOv2 was trained with ImageNet normalization, so we use those mean and std values here 
    ])

    # Validation transform (NO augmentations)
    val_img_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Pad(padding=image_padding, fill=0),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #NOTE: (CHECK IF TRUE) DINOv2 was trained with ImageNet normalization, so we use those mean and std values here 
    ])

    
    # Target transform for TRAINING
    train_target_transform = Compose([
        ToImage(),
        Pad(padding=image_padding, fill=255),
        ToDtype(torch.int64),
    ])

    # Target transform for VALIDATION
    val_target_transform = Compose([
        ToImage(),
        Pad(padding=image_padding, fill=255),
        ToDtype(torch.int64),
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(args.data_dir,
                               split="train",
                               mode="fine",
                               target_type="semantic",
                               transform=train_img_transform,
                               target_transform=train_target_transform)

    valid_dataset = Cityscapes(args.data_dir,
                               split="val",
                               mode="fine",
                               target_type="semantic",
                               transform=val_img_transform,
                               target_transform=val_target_transform)

    # Wrap datasets with augmentation (training only)
    train_dataset = AugmentedCityscapes(train_dataset, augment=True)
    valid_dataset = AugmentedCityscapes(valid_dataset, augment=False)

    # Seed the data loaders as well, so shuffling and worker-side randomness stay reproducible
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    valid_generator = torch.Generator()
    valid_generator.manual_seed(args.seed)

    # Define the data loaders with the seeded workers for reproducibility
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=valid_generator
    )

    # Define the model
    #NOTE: Not used in the submission server of the course, but this was just implemented to run the experiments for the report.
    from model import DinoEoMT, DinoLinearDecoder, DinoMLPDecoder, DinoUpsamplingDecoder

    if args.decoder == "linear":
        model = DinoLinearDecoder(n_classes=19)
    elif args.decoder == "upsample":
        model = DinoUpsamplingDecoder(n_classes=19)
    elif args.decoder == "mlp":
        model = DinoMLPDecoder(n_classes=19)
    elif args.decoder == "EoMT":
        model = DinoEoMT(n_classes=19)
    else:
        raise ValueError("Unknown decoder")

    model = model.to(device)
    
    # For the plain DINOv2 models, freeze the entire backbone and train only the decoder head.
    # EoMT is used here as a frozen pretrained baseline, so none of its parameters are updated.
    if args.decoder == "EoMT": #NOTE: This uses the pretrained EoMT for cityscapes, but this highly likely has data leakage.
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True

    is_trainable_model = args.decoder != "EoMT"
    
    # Define the loss function
    # Experiment C: safety-critical rebalanced CE + Dice + boundary loss
    criterion = SafetyCriticalRebalancedBoundaryLoss(ce_loss=nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1),
                                                     dice_loss=DiceLoss(ignore_index=255),
                                                     num_classes=19,
                                                     ignore_index=255,
                                                     band_width_ratio=0.005,
                                                     ce_weight=1.0,
                                                     dice_weight=0.5,
                                                     alpha_start=0.01,
                                                     alpha_end=0.5,
                                                     num_epochs=args.epochs)
    
    # Define the optimizer (SGD) with momentum and weight decay and a polynomial learning rate scheduler.
    # L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” 2018. [Online]. Available: https://arxiv.org/abs/1802.02611
    # M. Li, E. Yumer, and D. Ramanan, “Budgeted training: Rethinking deep neural network training under resource constraints,” 2020. [Online]. Available: https://arxiv.org/abs/1905.04753
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-3)
    
    
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer,
                                               total_iters=args.epochs * len(train_dataloader) // args.gradient_accumulation_steps,
                                               power=0.9)

    # Initialize mixed precision, multi-GPU training and gradient accumulation with accelerator
    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(model,
                                                                                          optimizer,
                                                                                          train_dataloader,
                                                                                          valid_dataloader,
                                                                                          scheduler)

    label_lookup = torch.full((256,), 255, dtype=torch.long)
    for k, v in id_to_trainid.items():
        label_lookup[k] = v

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    train_step = 0
    for epoch in range(args.epochs):
        # Print epochs in the slurm-*.out job output files, to track training progress from the server console as well
        epoch_alpha = criterion.get_alpha(epoch) if isinstance(criterion, RebalancedBoundaryLoss) else None
        if epoch_alpha is None:
            accelerator.print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        else:
            accelerator.print(f"Epoch {epoch+1:04}/{args.epochs:04} | alpha={epoch_alpha:.4f}")

        # Training
        if is_trainable_model:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i, (images, labels) in enumerate(train_dataloader):

                images = images.to(device)
                labels = label_lookup[labels.squeeze(1)].to(device)

                # Use accelerator to handle mixed precision and gradient accumulation
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        outputs = model(images)
                        loss, loss_components = compute_loss_with_components(criterion, outputs, labels, epoch)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        optimizer.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                gathered_loss = accelerator.gather(loss.detach().reshape(1)).mean().item() # NOTE: gradient accumulation
                gathered_loss_components = {
                    name: accelerator.gather(value.detach().float().reshape(1)).mean().item()
                    for name, value in loss_components.items()
                }

                # Log the training metrics only on real optimizer update steps
                if accelerator.sync_gradients:
                    train_step += 1

                if accelerator.is_main_process and accelerator.sync_gradients:
                    train_metrics = {
                        "train_loss": gathered_loss,
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                    }
                    for name, value in gathered_loss_components.items():
                        train_metrics[f"train_{name}"] = value
                    wandb.log(train_metrics, step=train_step)
        else:
            model.eval()
            if accelerator.is_main_process:
                wandb.log({"epoch": epoch + 1}, step=epoch + 1)
            
        # Validation
        model.eval()
        with torch.no_grad():
            # Initialize loss sums and confusion matrix for the validation set
            valid_loss_sum = torch.zeros(1, dtype=torch.float32, device=device)
            valid_loss_count = torch.zeros(1, dtype=torch.float32, device=device)
            valid_loss_components_sum = None
            class_confusion_matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64, device=device)
            boundary_intersections = torch.zeros(len(class_names), dtype=torch.float32, device=device)
            boundary_unions = torch.zeros(len(class_names), dtype=torch.float32, device=device)
            
            for i, (images, labels) in enumerate(valid_dataloader):

                images = images.to(device)
                labels = label_lookup[labels.squeeze(1)].to(device)
            
                with accelerator.autocast():
                    outputs = model(images)
                    loss, loss_components = compute_loss_with_components(criterion, outputs, labels, epoch)

                valid_loss_sum += loss.detach()
                valid_loss_count += 1
                if valid_loss_components_sum is None:
                    valid_loss_components_sum = {
                        name: torch.zeros(1, dtype=torch.float32, device=device)
                        for name in loss_components
                    }
                for name, value in loss_components.items():
                    valid_loss_components_sum[name] += value.detach().float().reshape(1)

                predictions = outputs.softmax(1).argmax(1)

                class_confusion_matrix = update_class_confusion_matrix(class_confusion_matrix,
                                                                       predictions,
                                                                       labels)

                batch_boundary_intersections, batch_boundary_unions = compute_boundary_iou_batch(predictions,
                                                                                                 labels,
                                                                                                 n_classes=len(class_names))
                
                boundary_intersections += batch_boundary_intersections
                boundary_unions += batch_boundary_unions

                # Log to W&B some validation images
                if i == 0 and accelerator.is_main_process:
                    preds_vis = predictions[:2].unsqueeze(1)
                    labels_vis = labels[:2].unsqueeze(1)

                    preds_vis = convert_train_id_to_color(preds_vis)
                    labels_vis = convert_train_id_to_color(labels_vis)

                    predictions_img = make_grid(preds_vis.cpu(), nrow=2).permute(1, 2, 0).numpy()
                    labels_img = make_grid(labels_vis.cpu(), nrow=2).permute(1, 2, 0).numpy()

                    validation_step = train_step if is_trainable_model else epoch + 1
                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=validation_step)
            
            # Aggregate the validation loss sums and counts across all processes, and compute the final validation loss for this epoch on the main process
            valid_loss_sum = accelerator.gather(valid_loss_sum).sum()
            valid_loss_count = accelerator.gather(valid_loss_count).sum()
            if valid_loss_components_sum is not None:
                valid_loss_components_sum = {
                    name: accelerator.gather(value).sum()
                    for name, value in valid_loss_components_sum.items()
                }
            class_confusion_matrix = accelerator.gather(class_confusion_matrix.unsqueeze(0)).sum(dim=0)
            boundary_intersections = accelerator.gather(boundary_intersections.unsqueeze(0)).sum(dim=0)
            boundary_unions = accelerator.gather(boundary_unions.unsqueeze(0)).sum(dim=0)

            # Only the main process should log metrics and save models, to avoid conflicts and redundant logging/saving
            if accelerator.is_main_process:
                valid_loss = (valid_loss_sum / valid_loss_count).item()
                
                # Compute the metrics from the confusion matrix and log them to W&B, along with the validation loss
                validation_metrics = compute_dice_metrics(class_confusion_matrix)
                validation_metrics.update(compute_boundary_iou_metrics(boundary_intersections, boundary_unions))

                # Log the validation loss to W&B
                validation_metrics["valid_loss"] = valid_loss
                if isinstance(criterion, RebalancedBoundaryLoss):
                    validation_metrics["alpha"] = criterion.get_alpha(epoch)
                if valid_loss_components_sum is not None:
                    for name, value in valid_loss_components_sum.items():
                        validation_metrics[f"valid_{name}"] = (value / valid_loss_count).item()
                validation_step = train_step if is_trainable_model else epoch + 1
                wandb.log(validation_metrics, step=validation_step)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if current_best_model_path:
                        os.remove(current_best_model_path)
                    current_best_model_path = os.path.join(
                        output_dir, 
                        f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                    )
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(),
                        current_best_model_path,
                    )

    accelerator.print("Training complete!")

    # Save the model
    if accelerator.is_main_process:
        accelerator.save(accelerator.unwrap_model(model).state_dict(),
                         os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={best_valid_loss:04}.pt")) # type: ignore
        wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
