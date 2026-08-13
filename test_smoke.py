"""Smoke tests: exercise the full pipeline with random weights.

Deliberately does NOT download the trained checkpoint, so CI stays fast
and does not depend on external hosting.
"""

import numpy as np
from PIL import Image

from inference import (
    CLASSES,
    IMG_SIZE,
    build_model,
    gradcam_plus_plus,
    overlay,
    predict,
    preprocess,
)


def _dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_class_order_is_fixed():
    # The checkpoint's output head is tied to this exact order.
    assert CLASSES == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_preprocess_shapes():
    rgb01, tensor = preprocess(_dummy_image())
    assert rgb01.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert rgb01.min() >= 0.0 and rgb01.max() <= 1.0
    assert tuple(tensor.shape) == (1, 3, IMG_SIZE, IMG_SIZE)


def test_model_head_is_seven_way():
    model = build_model()
    assert model.fc.out_features == len(CLASSES)


def test_predict_returns_probabilities():
    model = build_model().eval()
    _, tensor = preprocess(_dummy_image())
    probs = predict(model, tensor)
    assert probs.shape == (len(CLASSES),)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_gradcam_plus_plus_and_overlay():
    model = build_model().eval()
    rgb01, tensor = preprocess(_dummy_image())
    cam = gradcam_plus_plus(model, tensor, class_idx=0)
    assert cam.shape == (IMG_SIZE, IMG_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0

    blended = overlay(rgb01, cam)
    assert blended.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert blended.dtype == np.uint8
