"""Inference + Grad-CAM++ explanation for the HAM10000 ResNet-50 classifier.

Mirrors the preprocessing, class order and CAM target layer used in the
training notebook so that the app reproduces the dissertation results.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

# --- must match the notebook exactly -----------------------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

FULL_NAME = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# -----------------------------------------------------------------------

# Free hosting gives us ~1 GB of RAM and 1-2 cores; extra threads only add
# memory arenas here, they do not make a single 224x224 image faster.
torch.set_num_threads(1)

CKPT_PATH = Path(os.environ.get("CKPT_PATH", "resnet50_ham10000_best.pt"))


def build_model(pretrained: bool = False) -> nn.Module:
    """ResNet-50 with a 7-way head. No checkpoint loaded — safe to call in CI."""
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def download_checkpoint(url: str, dest: Path = CKPT_PATH) -> Path:
    """Fetch the trained weights once, if they are not already on disk."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_model(ckpt_path: Path | str = CKPT_PATH) -> nn.Module:
    """Build the model and load the trained state dict onto CPU."""
    model = build_model()
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Return (rgb01, input_tensor).

    rgb01 is the HxWx3 float image in [0, 1] used for the heatmap overlay;
    input_tensor is the normalised 1x3xHxW batch fed to the model.
    """
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb01 = np.asarray(rgb).astype("float32") / 255.0

    tensor = torch.from_numpy(np.ascontiguousarray(rgb01.transpose(2, 0, 1))).float()
    for c in range(3):
        tensor[c] = (tensor[c] - MEAN[c]) / STD[c]
    return rgb01, tensor.unsqueeze(0)


def predict(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Softmax probabilities over the seven classes, shape (7,)."""
    with torch.no_grad():
        logits = model(input_tensor)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def gradcam_plus_plus(
    model: nn.Module, input_tensor: torch.Tensor, class_idx: int
) -> np.ndarray:
    """Grad-CAM++ saliency map, shape (IMG_SIZE, IMG_SIZE), values in [0, 1].

    Grad-CAM++ is used rather than Score-CAM because Score-CAM needs 30-60 s
    per image on CPU, which is unusable in an interactive app.
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layers = [model.layer4[-1]]
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(int(class_idx))],
        )
    return grayscale[0]


def overlay(rgb01: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend the saliency map over the image. Returns uint8 RGB."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    return show_cam_on_image(rgb01, cam, use_rgb=True)
