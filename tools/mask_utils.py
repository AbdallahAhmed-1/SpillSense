# tools/mask_utils.py
"""
Utilities for parsing multi-class segmentation masks using a color-to-label mapping.

This module reads a label definition file (e.g., `label.txt`) that maps RGB colors
in your mask image to semantic class names, and provides functions to:
  - load the mapping
  - convert an RGB mask image into per-class boolean masks
  - compute per-class metrics (area, bounding box, shape descriptors)

Example:
    mapping = load_label_map("label.txt")
    class_masks = parse_label_mask("mask.png", mapping)
    metrics = compute_class_metrics(class_masks)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from PIL import Image
from skimage import measure


def load_label_map(label_file: str) -> Dict[Tuple[int,int,int], str]:
    """
    Load a label mapping file with lines:
        R G B classname
    Returns a dict mapping (R,G,B) triplets to class names.
    """
    mapping: Dict[Tuple[int,int,int], str] = {}
    p = Path(label_file)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    with open(p, 'r') as f:
        for line in f:
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            r, g, b, name = parts
            mapping[(int(r), int(g), int(b))] = name
    return mapping


def parse_label_mask(mask_path: str,
                     mapping: Dict[Tuple[int,int,int], str]
                    ) -> Dict[str, np.ndarray]:
    """
    Read an RGB mask image and create a boolean mask for each class name.

    Returns:
        { class_name: mask_bool_array }
    """
    img = Image.open(mask_path).convert("RGB")
    arr = np.array(img)
    class_masks: Dict[str, np.ndarray] = {}
    for color, name in mapping.items():
        # boolean mask where all channels match the color
        mask_bool = np.all(arr == color, axis=-1)
        class_masks[name] = mask_bool
    return class_masks


def compute_class_metrics(class_masks: Dict[str, np.ndarray]
                         ) -> Dict[str, dict]:
    """
    Compute simple metrics for each class mask:
      - area (pixel count)
      - bounding box (min_row, min_col, max_row, max_col)
      - shape descriptors: area, perimeter, solidity, eccentricity

    Returns a dict: { class_name: {metrics...} }
    """
    results: Dict[str, dict] = {}
    for name, mask in class_masks.items():
        # total pixel count
        area = int(mask.sum())
        bbox = None
        solidity = None
        eccentricity = None
        perimeter = None
        # get regionprops on the largest connected component
        props = measure.regionprops(mask.astype(int))
        if props:
            # choose the component with largest area
            largest = max(props, key=lambda r: r.area)
            minr, minc, maxr, maxc = largest.bbox
            bbox = (int(minr), int(minc), int(maxr), int(maxc))
            solidity = float(largest.solidity)
            eccentricity = float(largest.eccentricity)
            perimeter = float(largest.perimeter)
        results[name] = {
            "area_pixels": area,
            "bounding_box": bbox,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "perimeter": perimeter,
        }
    return results
