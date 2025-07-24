# tools/image_tools.py
"""
Reusable plotting helpers for RGB images and binary masks.
===========================================================
These functions are **stateless** and accept a single image
(H, W, 3) array and/or mask (H, W) boolean array. They return a
**matplotlib.figure.Figure** so callers can choose to display,
save, or convert to base64 for reporting.

Key design points
-----------------
• No global state – safe in multi-threaded workflows.
• Matplotlib-only (no seaborn) to keep deps minimal.
• All plots accept NumPy arrays and optional keyword args for titles.

Figures are closed after conversion to avoid memory leaks.

Example
-------
```python
from tools.image_tools import plot_image_mask_overlay
fig = plot_image_mask_overlay(img, mask, title="Overlay Example")
fig.savefig("overlay.png", dpi=150)
```
"""
from __future__ import annotations

import io
import matplotlib
matplotlib.use('Agg')
import base64
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import measure
from skimage.measure import regionprops, label
from PIL import Image
from pathlib import Path
from joblib import load


MODEL_PATH = Path("models/image/best_image_model.joblib")

__all__ = [
    "fig_to_base64",
    "plot_image_mask_overlay",
    "plot_color_histogram",
    "plot_shape_distribution",
    "plot_texture_glcm",
    "extract_image_features",
    "get_class_ids",
    "IMG_DIR",
    "MASK_DIR"
]

IMG_DIR = Path("data/processed/jpg")
MASK_DIR = Path("data/processed/masks")

# ---------------------------------------------------------------------
# Helper – convert Figure -> base64 PNG and close figure
# ---------------------------------------------------------------------
def fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return "data:image/png;base64," + img_b64

# ---------------------------------------------------------------------
# 1. Image & mask overlay
# ---------------------------------------------------------------------
def plot_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    title: str = "Image/Mask Overlay",
    alpha: float = 0.4,
) -> Figure:
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(mask, cmap='jet', alpha=alpha)
    ax.axis('off')
    ax.set_title(title)
    return fig

# ---------------------------------------------------------------------
# 2. Color histogram per channel
# ---------------------------------------------------------------------
def plot_color_histogram(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    bins: int = 256,
    title: str = "Color Histogram",
) -> Figure:
    fig, ax = plt.subplots()
    for i, channel in enumerate(('R', 'G', 'B')):
        data = image[..., i]
        if mask is not None:
            data = data[mask]
        ax.hist(data.ravel(), bins=bins, alpha=0.5, label=channel)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    return fig

# ---------------------------------------------------------------------
# 3. Shape distribution of mask regions
# ---------------------------------------------------------------------
def plot_shape_distribution(
    mask: np.ndarray,
    *,
    bins: int = 20,
    title: str = "Shape Distribution",
) -> Figure:
    props = measure.regionprops(mask.astype(int))
    areas = [r.area for r in props]
    solidities = [r.solidity for r in props]
    eccentricities = [r.eccentricity for r in props]

    fig, ax = plt.subplots()
    ax.hist(areas, bins=bins, alpha=0.5, label='Area')
    ax.hist(solidities, bins=bins, alpha=0.5, label='Solidity')
    ax.hist(eccentricities, bins=bins, alpha=0.5, label='Eccentricity')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    return fig

# ---------------------------------------------------------------------
# 4. GLCM-based texture features
# ---------------------------------------------------------------------
def plot_texture_glcm(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = (0, np.pi/4, np.pi/2, 3*np.pi/4),
    title: str = "GLCM Texture Features",
) -> Figure:
    gray = image.mean(axis=2).astype(np.uint8)
    arr = np.zeros_like(gray)
    arr[mask] = gray[mask]
    glcm = graycomatrix(arr, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    features = {
        prop: graycoprops(glcm, prop).mean()
        for prop in ('contrast', 'homogeneity', 'energy', 'correlation')
    }

    fig, ax = plt.subplots()
    ax.bar(list(features.keys()), list(features.values()))
    ax.set_ylabel('Value')
    ax.set_title(title)
    return fig


def extract_image_features(img_dir: Path, mask_dir: Path, class_ids: list):
    X, y = [], []
    for mask_path in mask_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path))
        img_path = img_dir / mask_path.name.replace(".png", ".jpg")
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        for class_id in class_ids:
            region = (mask == class_id)
            if np.any(region):
                pixels = img[region]
                mean_color = pixels.mean(axis=0)
                std_color = pixels.std(axis=0)
                features = np.concatenate([np.atleast_1d(mean_color), np.atleast_1d(std_color)])
                X.append(features)
                y.append(class_id)

    return np.array(X), np.array(y)


def get_class_ids(mask_dir: Path) -> list:
    ids = set()
    for mask_path in mask_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path))
        ids.update(np.unique(mask).tolist())
    ids.discard(0)  # skip background
    return sorted(list(ids))


def extract_single_image_features(image_path: Path):
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        # Simple features — same as your training:
        r_mean = np.mean(arr[:, :, 0])
        g_mean = np.mean(arr[:, :, 1])
        b_mean = np.mean(arr[:, :, 2])
        return [r_mean, g_mean, b_mean][:2]  # if you trained on only R/G
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
    
    
def extract_features(image_path: str) -> np.ndarray:
    """Compute mean and std from RGB image."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img) / 255.0
    mean = arr.mean()
    std = arr.std()
    return np.array([[mean, std]])

    
def predict_image_spill(image_path: str) -> str:
    if not MODEL_PATH.exists():
        return "⚠️ Spill detection model not found. Please train it first."

    features = extract_features(image_path)
    model = load(MODEL_PATH)
    pred = model.predict(features)[0]

    if pred in [204, 221, 255]:  # assume these are 'spill' classes
        return f"🔍 Predicted class ID: {pred}\n🛢️ Spill detected!"
    else:
        return f"🔍 Predicted class ID: {pred}\n✅ No spill detected."
    
