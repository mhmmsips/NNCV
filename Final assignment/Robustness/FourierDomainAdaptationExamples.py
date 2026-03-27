"""
Visualise the effect of Fourier Domain Adaptation (FDA) on Cityscapes images.

Reference: "FDA: Fourier Domain Adaptation for Semantic Segmentation", Yang & Soatto, CVPR 2020.

Four export directories are created inside the Robustness folder:
  FDATrainingExamples; 50 training images, FDA at 100% (beta sampled from (0.0, 0.05))
  FDAValidationExamples; the 2 W&B validation images, each augmented 25 times, FDA at 100%
  FullyAugmentedFDATrainingExamples; 50 training images, full pipeline (FDA + flips/jitter/blur)
  FullyAugmentedFDAValidationExamples; the 2 W&B validation images, each augmented 25 times, full pipeline

Seed 8 is used throughout for reproducibility.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (Compose,
                                        Normalize,
                                        ToImage,
                                        ToDtype,
                                        Pad,
                                        RandomHorizontalFlip,
                                        ColorJitter,
                                        GaussianBlur,
                                        RandomApply)
import albumentations as A


# Seed everything upfront so every run produces identical outputs
seed = 8
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Data directory; same default as train.py
data_dir = "./data/cityscapes"

# SYNTHIA reference images for FDA style transfer; same default as train.py
synthia_dir = "./data/synthia"

# Padding to make both spatial dims divisible by 14 for DINOv2-L/14
image_padding = (5, 6, 5, 6) # left, top, right, bottom

# Output directories; all siblings of this script inside the Robustness folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_fda_train = os.path.join(script_dir, "FDATrainingExamples")
dir_fda_val = os.path.join(script_dir, "FDAValidationExamples")
dir_full_fda_train = os.path.join(script_dir, "FullyAugmentedFDATrainingExamples")
dir_full_fda_val = os.path.join(script_dir, "FullyAugmentedFDAValidationExamples")

for d in [dir_fda_train, dir_fda_val, dir_full_fda_train, dir_full_fda_val]:
    os.makedirs(d, exist_ok=True)


# Collect SYNTHIA reference image paths
# FDA randomly picks one reference image per call, so the style varies across the export set
synthia_image_paths = []
for root, _, files in os.walk(synthia_dir):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            synthia_image_paths.append(os.path.join(root, fname))

if len(synthia_image_paths) == 0:
    raise ValueError(f"No images found in synthia directory: {synthia_dir}. ")

print(f"Found {len(synthia_image_paths)} SYNTHIA reference images")

# FDA transform applied to 100% of images
# beta_limit=(0.0, 0.05): the paper shows beta <= 0.05 produces clean style transfer without visible artefacts.
# Sampling uniformly from the full range gives diversity across subtle to moderate adaptation strengths; the model sees a range of domain shifts during training.
fda_aug = A.FDA(reference_images=synthia_image_paths,
                beta_limit=(0.0, 0.05),
                read_fn=lambda x: x, # paths passed as strings; albumentations reads them internally
                p=1.0)


# Torchvision transforms; same as train.py
imagenet_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Training transform — only converts to tensor and pads; no jitter, blur, or normalization here
# The correct augmentation order is: FDA --> ColorJitter --> GaussianBlur --> flip --> normalize, so jitter and blur must run after albumentations, not before
train_img_transform = Compose([ToImage(),
                                ToDtype(torch.float32, scale=True),
                                Pad(padding=image_padding, fill=0)])

# Validation transform without normalize; albumentations needs uint8 before normalization shifts the range
val_img_transform_no_norm = Compose([ToImage(),
                                     ToDtype(torch.float32, scale=True),
                                     Pad(padding=image_padding, fill=0)])

# Photometric and geometric transforms applied after albumentations augmentations
color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
gaussian_blur = RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3)
joint_flip = RandomHorizontalFlip(p=0.5)

# Shared mask transform for both train and val datasets
mask_transform = Compose([ToImage(),
                           Pad(padding=image_padding, fill=255),
                           ToDtype(torch.int64)])


def tensor_to_uint8_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a float32 CHW tensor in [0, 1] to a uint8 HWC numpy array for albumentations."""
    return (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")


def uint8_numpy_to_tensor(img_numpy: np.ndarray) -> torch.Tensor:
    """Convert a uint8 HWC numpy array back to a float32 CHW tensor in [0, 1]."""
    return torch.from_numpy(img_numpy).permute(2, 0, 1).float() / 255.0


def apply_fda_only(img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply FDA style transfer and return a uint8 CHW tensor ready for PIL saving.

    The image is NOT normalized; output lives in [0, 255] so it can be saved
    directly as a PNG without needing to undo ImageNet normalization.
    """
    img_numpy = tensor_to_uint8_numpy(img_tensor)
    img_numpy = fda_aug(image=img_numpy)["image"]
    # Return uint8 CHW for easy saving with PIL
    return torch.from_numpy(img_numpy).permute(2, 0, 1)


def apply_full_pipeline_fda(img_tensor: torch.Tensor,
                             mask_tensor: torch.Tensor) -> torch.Tensor:
    """Apply the full training pipeline from train.py (with FDA) and return a uint8 CHW tensor.

    Pipeline: FDA --> ColorJitter --> GaussianBlur --> flip --> normalize --> de-normalize for saving

    The normalize/de-normalize round-trip keeps the augmentation order identical to train.py
    while producing a visually meaningful PNG (pixel values in [0, 255] rather than normalized).
    """
    # Apply FDA first, on the clean uint8 image
    img_numpy = tensor_to_uint8_numpy(img_tensor)
    img_numpy = fda_aug(image=img_numpy)["image"]
    img_tensor = uint8_numpy_to_tensor(img_numpy)

    # Apply photometric augmentations after FDA, then the joint flip
    img_tensor = color_jitter(img_tensor)
    img_tensor = gaussian_blur(img_tensor)
    img_tensor, mask_tensor = joint_flip(img_tensor, mask_tensor)

    # Normalize; same as AugmentedCityscapes.__getitem__ in train.py
    img_tensor = imagenet_normalize(img_tensor)

    # De-normalize so the saved PNG has human-readable pixel values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor * std + mean).clamp(0, 1)

    return (img_tensor * 255).byte() # uint8 CHW


def save_tensor_as_png(img_chw: torch.Tensor, path: str):
    """Save a uint8 CHW tensor as a PNG file."""
    pil_img = Image.fromarray(img_chw.permute(1, 2, 0).numpy())
    pil_img.save(path)


def load_train_dataset() -> Cityscapes:
    """Load the Cityscapes training split with the same transform as train.py.

    NOTE: Normalize is excluded from train_img_transform (matches train.py); it runs after FDA.
    """
    return Cityscapes(data_dir,
                      split="train",
                      mode="fine",
                      target_type="semantic",
                      transform=train_img_transform,
                      target_transform=mask_transform)


def load_val_dataset_no_norm() -> Cityscapes:
    """Load the Cityscapes validation split without the final normalize step.

    Normalization is skipped so albumentations receives a uint8 image in the
    original pixel range, before any normalization shift has been applied.
    """
    return Cityscapes(data_dir,
                      split="val",
                      mode="fine",
                      target_type="semantic",
                      transform=val_img_transform_no_norm,
                      target_transform=mask_transform)


# %%
# Export 1: FDATrainingExamples
# 50 random training images, each augmented once with FDA at 100% probability

print("Exporting FDATrainingExamples...")

train_ds = load_train_dataset()

# Pick 50 random indices; seeded so the selection is reproducible
rng = random.Random(seed)
train_indices = rng.sample(range(len(train_ds)), 50)

for export_idx, ds_idx in enumerate(train_indices):
    img, _ = train_ds[ds_idx] # float32 CHW, padded, not yet jittered or normalized

    img_augmented = apply_fda_only(img)

    fname = f"train_{export_idx:03d}_ds{ds_idx}.png"
    save_tensor_as_png(img_augmented, os.path.join(dir_fda_train, fname))

print(f"  Saved {len(train_indices)} images to {dir_fda_train}")


# %%
# Export 2: FDAValidationExamples
# The 2 W&B validation images (first 2 of the val split, no shuffle), each augmented 25 times

print("Exporting FDAValidationExamples...")

# The W&B images are always the first 2 images from the val dataloader (i == 0, [:2])
# The val dataloader does not shuffle, so these are simply val indices 0 and 1
val_ds_no_norm = load_val_dataset_no_norm()

for val_image_idx in range(2):
    img, _ = val_ds_no_norm[val_image_idx] # float32 CHW, padded, not yet normalized

    for rep in range(25):
        img_augmented = apply_fda_only(img)

        fname = f"val{val_image_idx}_{rep:02d}.png"
        save_tensor_as_png(img_augmented, os.path.join(dir_fda_val, fname))

print(f"  Saved 50 images (2 x 25) to {dir_fda_val}")


# %%
# Export 3: FullyAugmentedFDATrainingExamples
# Same 50 training images, full pipeline from train.py (FDA + flip + jitter + blur)

print("Exporting FullyAugmentedFDATrainingExamples...")

for export_idx, ds_idx in enumerate(train_indices):
    img, mask = train_ds[ds_idx]

    img_augmented = apply_full_pipeline_fda(img, mask)

    fname = f"train_{export_idx:03d}_ds{ds_idx}.png"
    save_tensor_as_png(img_augmented, os.path.join(dir_full_fda_train, fname))

print(f"  Saved {len(train_indices)} images to {dir_full_fda_train}")


# %%
# Export 4: FullyAugmentedFDAValidationExamples
# Same 2 W&B validation images, each augmented 25 times with the full FDA pipeline

print("Exporting FullyAugmentedFDAValidationExamples...")

for val_image_idx in range(2):
    img, mask = val_ds_no_norm[val_image_idx]

    for rep in range(25):
        img_augmented = apply_full_pipeline_fda(img, mask)

        fname = f"val{val_image_idx}_{rep:02d}.png"
        save_tensor_as_png(img_augmented, os.path.join(dir_full_fda_val, fname))

print(f"  Saved 50 images (2 x 25) to {dir_full_fda_val}")


print("\nAll FDA exports complete.")
