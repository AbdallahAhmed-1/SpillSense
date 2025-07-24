# tools/hsi_tools.py
"""Reusable plotting helpers for hyperspectral cubes
=====================================================
These functions are **stateless** and accept a single cube (H, W, B)
array or its derivatives. They return a **matplotlib.figure.Figure** so
callers (e.g., MATExplorator) can decide whether to show, save, or
convert to base64 for the LLM / Reporter agent.

Key design points
-----------------
• No global state – safe in multi‑threaded workflows.
• Matplotlib‑only (no seaborn) to keep deps thin.
• All spectral plots expect reflectance‐scaled data (0‑1 float32).
• Optional keyword args allow customising titles or band selections.

Example
-------
```python
from tools import hsi_tools
fig = hsi_tools.plot_mean_spectrum(cube, title="Mean Spectrum – GM18")
fig.savefig("mean_spectrum.png", dpi=150)
```
"""
from __future__ import annotations

import io
import base64
from typing import Tuple, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = [
    "plot_mean_spectrum",
    "plot_band_variance",
    "plot_pca_scatter",
    "plot_rgb_composite",
    "plot_variance_explained",
    "plot_ndvi_map",
]

# ---------------------------------------------------------------------
# Helper util – convert fig → base64 PNG (for LLM embedding)
# ---------------------------------------------------------------------


def fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    return "data:image/png;base64," + img_b64

# ---------------------------------------------------------------------
# 1. Mean spectrum
# ---------------------------------------------------------------------

def plot_mean_spectrum(cube: np.ndarray, *, title: str = "Mean Spectrum") -> Figure:
    """Plot the average reflectance per band across the entire scene."""
    h, w, b = cube.shape
    spectrum = cube.reshape(-1, b).mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(b), spectrum, "-o", linewidth=1)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Mean reflectance")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.3)
    return fig

# ---------------------------------------------------------------------
# 2. Per‑band variance
# ---------------------------------------------------------------------

def plot_band_variance(cube: np.ndarray, *, title: str = "Band Variance") -> Figure:
    h, w, b = cube.shape
    var = cube.reshape(-1, b).var(axis=0)

    fig, ax = plt.subplots()
    ax.bar(range(b), var)
    ax.set_xlabel("Band index")
    ax.set_ylabel("Variance")
    ax.set_title(title)
    return fig

# ---------------------------------------------------------------------
# 3. PCA 2‑D scatter (pixels)
# ---------------------------------------------------------------------

def plot_pca_scatter(
    cube: np.ndarray,
    *,
    sample: int = 5000,
    title: str = "PCA Scatter – first 2 comps",
) -> Figure:
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)

    if sample < flat.shape[0]:
        idx = np.random.choice(flat.shape[0], sample, replace=False)
        flat = flat[idx]

    scaler = StandardScaler()
    flat_z = scaler.fit_transform(flat)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(flat_z)

    fig, ax = plt.subplots()
    ax.scatter(pcs[:, 0], pcs[:, 1], s=3, alpha=0.3)
    ax.set_xlabel("PC‑1")
    ax.set_ylabel("PC‑2")
    ax.set_title(f"{title} (var exp {pca.explained_variance_ratio_.sum():.2%})")
    return fig

# ---------------------------------------------------------------------
# 4. Simple RGB composite (choose 3 bands)
# ---------------------------------------------------------------------

def plot_rgb_composite(
    cube: np.ndarray,
    *,
    rgb_indices: Tuple[int, int, int] = (30, 20, 10),
    title: str = "False‑colour composite",
    stretch: float = 0.01,
) -> Figure:
    r, g, b = [cube[:, :, i] for i in rgb_indices]
    rgb = np.stack([r, g, b], axis=-1)

    # Contrast stretch
    lo, hi = np.quantile(rgb, stretch), np.quantile(rgb, 1 - stretch)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title)
    return fig

# ---------------------------------------------------------------------
# 5. PCA variance explained curve
# ---------------------------------------------------------------------

def plot_variance_explained(cube: np.ndarray, *, title: str = "PCA Variance Curve") -> Figure:
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    pca = PCA().fit(flat)
    cum = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.plot(range(1, b + 1), cum, "-o")
    ax.set_xlabel("# Components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title(title)
    ax.axhline(0.99, color="red", linestyle="--", linewidth=1)
    return fig

# ---------------------------------------------------------------------
# 6. NDVI map (requires NIR & Red band indices)
# ---------------------------------------------------------------------

def plot_ndvi_map(
    cube: np.ndarray,
    *,
    nir_band: int,
    red_band: int,
    title: str = "NDVI Map",
) -> Figure:
    nir = cube[:, :, nir_band].astype(np.float32)
    red = cube[:, :, red_band].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)

    cmap = LinearSegmentedColormap.from_list("ndvi", ["brown", "yellow", "green"])

    fig, ax = plt.subplots()
    im = ax.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    return fig


