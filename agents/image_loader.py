# === agents/image_loader.py ===
"""
Load RGB `.jpg` images and their multi-class masks, then move from
`data/raw/images/` & `data/raw/masks/` → `data/processed/images/` & `data/processed/masks/`
so files aren’t re-processed.

Images are stored under `state["img_data"][stem]` as H×W×3 arrays.
Masks are stored under `state["mask_data"][stem]` as a dict of
{class_name: mask_bool_array} or {'_binary': mask_bool_array}.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import numpy as np
from state.state_utils import OilGasRCAState
from tools.mask_utils import load_label_map, parse_label_mask

RAW_IMG_DIR = Path("data/raw/jpg")
PROCESSED_IMG_DIR = Path("data/processed/jpg")
RAW_MASK_DIR = Path("data/raw/masks")
PROCESSED_MASK_DIR = Path("data/processed/masks")


class ImageLoader:
    """LangGraph node – expects `state['image_paths']`, `state['mask_paths']`,
and optional `label_map_file` for multi-class masks."""

    def __init__(
        self,
        *,
        load_data: bool = True,
        label_map_file: str | None = None
    ):
        self.load_data = load_data
        self.mapping: Dict[tuple[int,int,int], str] | None = None
        if label_map_file:
            self.mapping = load_label_map(label_map_file)

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        state.setdefault("img_data", {})
        state.setdefault("mask_data", {})
        processed_imgs: list[str] = []
        processed_masks: list[str] = []

        # Process Images
        for p_str in state.get("image_paths", []):
            p = Path(p_str).resolve()
            stem = p.stem
            if not p.exists():
                print(f"[ImageLoader] 🚫 image not found: {p}")
                continue
            img_arr: Any | None = None
            if self.load_data:
                try:
                    img = Image.open(p).convert("RGB")
                    img_arr = np.array(img)
                    print(f"[ImageLoader] ✅ loaded {p.name} size={img_arr.shape}")
                except Exception as e:
                    print(f"[ImageLoader] ⚠️ could not load {p.name}: {e}")
            if img_arr is not None:
                state["img_data"][stem] = img_arr
                
            print("=========================================================")
            print(f"[ImageLoader] atate['img_data']={state['img_data']}")
            print("=========================================================")

        # Process Masks
        for m_str in state.get("mask_paths", []):
            m = Path(m_str).resolve()
            stem = m.stem
            if not m.exists():
                print(f"[ImageLoader] 🚫 mask not found: {m}")
                continue
            mask_result: Any | None = None
            if self.load_data:
                try:
                    if self.mapping:
                        class_masks = parse_label_mask(str(m), self.mapping)
                        mask_result = class_masks
                        print(f"[ImageLoader] ✅ parsed mask {m.name} classes={list(class_masks)}")
                    else:
                        mask_img = Image.open(m).convert('L')
                        bin_mask = np.array(mask_img) > 0
                        mask_result = {'_binary': bin_mask}
                        print(f"[ImageLoader] ✅ loaded binary mask {m.name}")
                except Exception as e:
                    print(f"[ImageLoader] ⚠️ could not load mask {m.name}: {e}")
            if mask_result is not None:
                state["mask_data"][stem] = mask_result

        # Update paths
        state["image_paths"] = processed_imgs
        state["mask_paths"] = processed_masks
        return state
