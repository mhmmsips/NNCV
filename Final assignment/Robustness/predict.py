"""
This script provides and example implementation of a prediction pipeline 
for a DINOv2-large robustness model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Normalize,
    Pad,
)
from torchvision.datasets import Cityscapes

from model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

# Define the padding the same as in the training and validation process
image_padding = (5, 6, 5, 6) # left, top, right, bottom


def preprocess(img: Image.Image) -> torch.Tensor:
    pad_left, pad_top, pad_right, pad_bottom = image_padding
    transform = Compose([
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
        Pad(padding=(pad_left, pad_top, pad_right, pad_bottom)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor) -> np.ndarray:
    # Get argmax over classes — result shape: (1, 1, H_padded, W_padded)
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability

    # Crop away the padding; do NOT resize, to avoid distorting predictions
    pad_left, pad_top, pad_right, pad_bottom = image_padding
    h_padded = pred_max.shape[2]
    w_padded = pred_max.shape[3]
    pred_cropped = pred_max[
        :, :,
        pad_top : h_padded - pad_bottom,
        pad_left : w_padded - pad_right,
    ]

    prediction_numpy = pred_cropped.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy

def remap_train_ids_to_class_ids(prediction: np.ndarray) -> np.ndarray:
    trainid_to_id = {cls.train_id: cls.id for cls in Cityscapes.classes}
    trainid_to_id[255] = 255
    result = np.zeros_like(prediction)
    for train_id, class_id in trainid_to_id.items():
        result[prediction == train_id] = class_id
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model(decoder="upsample", n_classes=19)  #NOTE: Use upsample as that is deemed to be the best decoder in the PP experiments on the validation set.
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
