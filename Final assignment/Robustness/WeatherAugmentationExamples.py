"""
Visualise the effect of weather augmentations on Cityscapes images.

Four export directories are created inside the Robustness folder:
  WeatherAugmentedTrainingExamples; 50 training images, weather aug at 100%
  WeatherAugmentedValidationExamples; the 2 W&B validation images, each augmented 25 times, weather at 100%
  FullyAugmentedTrainingExamples; 50 training images, full pipeline (50% weather + flips/jitter/blur)
  FullyAugmentedValidationExamples; the 2 W&B validation images, each augmented 25 times, full pipeline

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
data_dir = "../data/cityscapes"

# Padding to make both spatial dims divisible by 14 for DINOv2-L/14
image_padding = (5, 6, 5, 6) # left, top, right, bottom

# Output directories; all siblings of this script inside the Robustness folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_weather_train = os.path.join(script_dir, "WeatherAugmentedTrainingExamples")
dir_weather_val = os.path.join(script_dir, "WeatherAugmentedValidationExamples")
dir_full_train = os.path.join(script_dir, "FullyAugmentedTrainingExamples")
dir_full_val = os.path.join(script_dir, "FullyAugmentedValidationExamples")

for d in [dir_weather_train, dir_weather_val, dir_full_train, dir_full_val]:
    os.makedirs(d, exist_ok=True)


# Weather augmentation block; copied verbatim from train.py
# p=1.0 on the outer OneOf means one effect is always applied (100% for the visualisation export)
weather_aug_always = A.OneOf([A.RandomRain(rain_type="heavy",
                                           drop_width=1,
                                           blur_value=5,
                                           brightness_coefficient=0.8,
                                           p=1.0),

                               A.RandomSnow(snow_point_range=(0.2, 0.4),
                                            brightness_coeff=2.5,
                                            p=1.0),

                               A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), # sun is always in the upper half
                                                p=1.0),

                               A.RandomShadow(shadow_roi=(0.0, 0.5, 1.0, 1.0),
                                              num_shadows_range=(1, 3),
                                              shadow_dimension=5,
                                              p=1.0),

                               A.RandomFog(fog_coef_range=(0.2, 0.5), # moderate fog: visible but not scene-destroying
                                           alpha_coef=0.15,
                                           p=1.0)],

                              p=1.0)

# Identical block with p=0.5; mirrors train.py exactly for the "full pipeline" exports
weather_aug_50pct = A.OneOf([A.RandomRain(rain_type="heavy",
                                          drop_width=1,
                                          blur_value=5,
                                          brightness_coefficient=0.8,
                                          p=1.0),

                              A.RandomSnow(snow_point_range=(0.2, 0.4),
                                           brightness_coeff=2.5,
                                           p=1.0),

                              A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5),
                                               p=1.0),

                              A.RandomShadow(shadow_roi=(0.0, 0.5, 1.0, 1.0),
                                             num_shadows_range=(1, 3),
                                             shadow_dimension=5,
                                             p=1.0),

                              A.RandomFog(fog_coef_range=(0.2, 0.5),
                                          alpha_coef=0.15,
                                          p=1.0)],

                             p=0.5) # 50% probability; matches train.py exactly


# Torchvision transforms; same as train.py
imagenet_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Training transform; only converts to tensor and pads; no jitter, blur, or normalization here
# The correct augmentation order is: WA --> ColorJitter --> GaussianBlur --> flip --> normalize, so jitter and blur must run after albumentations, not before
train_img_transform = Compose([ToImage(),
                                ToDtype(torch.float32, scale=True),
                                Pad(padding=image_padding, fill=0)])

# Validation transform: pad and normalize only, no augmentations
val_img_transform = Compose([ToImage(),
                              ToDtype(torch.float32, scale=True),
                              Pad(padding=image_padding, fill=0),
                              Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Validation transform without normalize; used for the images we want to weather-augment,
# so albumentations receives a uint8 image before the pixel range is shifted by normalization
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


def apply_weather_only(img_tensor: torch.Tensor,
                       weather_aug) -> torch.Tensor:
    """Apply one weather augmentation and return a uint8 CHW tensor ready for PIL saving.

    The image is NOT normalized; output lives in [0, 255] so it can be saved
    directly as a PNG without needing to undo ImageNet normalization.
    """
    img_numpy = tensor_to_uint8_numpy(img_tensor)
    img_numpy = weather_aug(image=img_numpy)["image"]
    # Return uint8 CHW for easy saving with PIL
    return torch.from_numpy(img_numpy).permute(2, 0, 1)


def apply_full_pipeline_train(img_tensor: torch.Tensor,
                               mask_tensor: torch.Tensor,
                               weather_aug) -> torch.Tensor:
    """Apply the full training augmentation pipeline from train.py and return a uint8 CHW tensor.

    Pipeline: WA --> ColorJitter --> GaussianBlur --> flip --> normalize --> de-normalize for saving

    The normalize/de-normalize round-trip keeps the augmentation order identical to train.py
    while producing a visually meaningful PNG (pixel values in [0, 255] rather than normalized).
    """
    # Apply weather augmentation first, on the clean uint8 image
    img_numpy = tensor_to_uint8_numpy(img_tensor)
    img_numpy = weather_aug(image=img_numpy)["image"]
    img_tensor = uint8_numpy_to_tensor(img_numpy)

    # Apply photometric augmentations after WA, then the joint flip
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

    NOTE: Normalize is excluded from train_img_transform (matches train.py); it runs after weather aug.
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
# Export 1: WeatherAugmentedTrainingExamples
# 50 random training images, each augmented once with a weather effect at 100% probability

print("Exporting WeatherAugmentedTrainingExamples...")

train_ds = load_train_dataset()

# Pick 50 random indices; seeded so the selection is reproducible
rng = random.Random(seed)
train_indices = rng.sample(range(len(train_ds)), 50)

for export_idx, ds_idx in enumerate(train_indices):
    img, _ = train_ds[ds_idx] # float32 CHW, padded, not yet jittered or normalized

    img_augmented = apply_weather_only(img, weather_aug_always)

    fname = f"train_{export_idx:03d}_ds{ds_idx}.png"
    save_tensor_as_png(img_augmented, os.path.join(dir_weather_train, fname))

print(f"  Saved {len(train_indices)} images to {dir_weather_train}")


# %%
# Export 2: WeatherAugmentedValidationExamples
# The 2 W&B validation images (first 2 of the val split, no shuffle), each augmented 25 times

print("Exporting WeatherAugmentedValidationExamples...")

# The W&B images are always the first 2 images from the val dataloader (i == 0, [:2])
# The val dataloader does not shuffle, so these are simply val indices 0 and 1
val_ds_no_norm = load_val_dataset_no_norm()

for val_image_idx in range(2):
    img, _ = val_ds_no_norm[val_image_idx] # float32 CHW, padded, not yet normalized

    for rep in range(25):
        img_augmented = apply_weather_only(img, weather_aug_always)

        fname = f"val{val_image_idx}_{rep:02d}.png"
        save_tensor_as_png(img_augmented, os.path.join(dir_weather_val, fname))

print(f"  Saved 50 images (2 x 25) to {dir_weather_val}")


# %%
# Export 3: FullyAugmentedTrainingExamples
# Same 50 training images, full pipeline from train.py (50% weather + flip + jitter + blur)

print("Exporting FullyAugmentedTrainingExamples...")

for export_idx, ds_idx in enumerate(train_indices):
    img, mask = train_ds[ds_idx]

    img_augmented = apply_full_pipeline_train(img, mask, weather_aug_50pct)

    fname = f"train_{export_idx:03d}_ds{ds_idx}.png"
    save_tensor_as_png(img_augmented, os.path.join(dir_full_train, fname))

print(f"  Saved {len(train_indices)} images to {dir_full_train}")


# %%
# Export 4: FullyAugmentedValidationExamples
# Same 2 W&B validation images, each augmented 25 times with the full training pipeline

print("Exporting FullyAugmentedValidationExamples...")

for val_image_idx in range(2):
    img, mask = val_ds_no_norm[val_image_idx]

    for rep in range(25):
        img_augmented = apply_full_pipeline_train(img, mask, weather_aug_50pct)

        fname = f"val{val_image_idx}_{rep:02d}.png"
        save_tensor_as_png(img_augmented, os.path.join(dir_full_val, fname))

print(f"  Saved 50 images (2 x 25) to {dir_full_val}")


print("\nAll exports complete.")
