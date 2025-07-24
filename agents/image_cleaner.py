# === agents/image_cleaner.py ===
"""
Pre-processing pipeline for RGB images and multi-class masks.

Image Processing Strategy
-------------------------
1. Resize images and masks to a target size (if specified).
2. Normalize image pixel values to [0, 1].
3. Denoise images (Gaussian or Median).
4. Clean each class mask: remove small objects and fill holes.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Any, Dict
from state.state_utils import OilGasRCAState


MaskDict = Dict[str, np.ndarray]


class ImageCleaner:
    """LangGraph node – expects `state['img_data']` and `state['mask_data']` as dicts."""

    def __init__(
        self,
        *,
        target_size: tuple[int, int] | None = None,
        normalize: bool = True,
        denoise_method: str = 'gaussian',
        denoise_params: dict[str, Any] | None = None,
        min_mask_size: int = 100
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params or {}
        self.min_mask_size = min_mask_size

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        imgs: dict[str, np.ndarray] = state.get("img_data", {})
        masks: dict[str, MaskDict] = state.get("mask_data", {})
        print("🧹 Starting image cleaning...")
        print(f"images: {imgs}, masks: {masks}")
        print("===============================================")
        if not imgs or not masks:
            print("🛑 No image or mask data found; skipping cleaning.")
            return state

        cleaned_imgs: Dict[str, np.ndarray] = {}
        cleaned_masks: Dict[str, MaskDict] = {}

        # ─── Clean Images ─────────────────────────────────────────────
        for name, img in imgs.items():
            print(f"🔍 Cleaning image {name}")
            clean_img = img.copy()
            if self.target_size:
                clean_img = self._resize(clean_img, self.target_size)
                print(f"   ▶︎ resized to {self.target_size}")
            if self.normalize:
                clean_img = self._normalize(clean_img)
                print("   ▶︎ normalized to [0, 1]")
            clean_img = self._denoise(clean_img)
            print(f"   ▶︎ denoised using {self.denoise_method}")
            cleaned_imgs[name] = clean_img

        # ─── Clean Masks ──────────────────────────────────────────────
        for name, mask_entry in masks.items():
            print(f"🔍 Cleaning masks for {name}")
            clean_entry: MaskDict = {}
            for class_name, mask in mask_entry.items():
                clean_mask = self._clean_mask(mask)
                print(f"   ▶︎ cleaned '{class_name}' mask (<{self.min_mask_size} px removed)")
                clean_entry[class_name] = clean_mask
            cleaned_masks[name] = clean_entry

        state["img_data_clean"] = cleaned_imgs
        state["mask_data_clean"] = cleaned_masks
        print(f"📦 Cleaned images: , {state['img_data_clean']}")
        print("📦 Cleaned class masks:", {k: list(v.keys()) for k, v in cleaned_masks.items()})
        
        print(f"ImageCleaner result - img_data_clean: {type(state.get('img_data_clean'))}")
        print(f"ImageCleaner result - img_data_clean length: {len(state.get('img_data_clean', {}))}")
        return state

    # ------------------------------------------------------------------
    def _resize(self, image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        if self.denoise_method == 'gaussian':
            k = self.denoise_params.get('kernel_size', (5, 5))
            return cv2.GaussianBlur(image, k, 0)
        elif self.denoise_method == 'median':
            k = self.denoise_params.get('kernel_size', 5)
            return cv2.medianBlur(image, k)
        else:
            return image

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_mask_size:
                cleaned[labels == i] = 1
        # Fill holes via morphological closing
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed.astype(bool)
