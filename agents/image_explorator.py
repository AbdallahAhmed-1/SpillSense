# === agents/image_explorator.py ===
"""
Visualization pipeline for cleaned RGB images and multi-class masks.

Generates class-specific overlays, color histograms, shape distributions,
texture analysis, and computes mask metrics. Stores figures and
metrics in state for reporting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import shutil
import re
from state.state_utils import OilGasRCAState, save_figure_to_state
from tools.image_tools import (
    plot_image_mask_overlay,
    plot_color_histogram,
    plot_shape_distribution,
    plot_texture_glcm,
    fig_to_base64,
)
from tools.mask_utils import compute_class_metrics

# Directory for storing exploration JSON summaries
EXP_DIR = Path("data/processed/images/explore").absolute()
EXP_DIR.mkdir(parents=True, exist_ok=True)

RAW_IMG_DIR = Path("data/raw/jpg")
PROCESSED_IMG_DIR = Path("data/processed/jpg")
RAW_MASK_DIR = Path("data/raw/masks")
PROCESSED_MASK_DIR = Path("data/processed/masks")


class ImageExplorator:
    """LangGraph node – expects `state['img_data_clean']` and `state['mask_data_clean']`."""

    def __init__(self, llm: Any = None):
        self.llm = llm

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        print("🔍 DEBUG - ImageExplorator Input Analysis:")
        print(f"   - State keys: {list(state.keys())}")
        
        def _normalize_key(k: str) -> str:
            return re.sub(r"[^a-z0-9]", "", k.lower())
        
        # Check all possible data keys
        for key in state.keys():
            if 'img' in key.lower() or 'mask' in key.lower() or 'data' in key.lower():
                value = state[key]
                if isinstance(value, dict):
                    print(f"   - {key}: {type(value)} with {len(value)} items")
                    if len(value) > 0:
                        first_key = list(value.keys())[0]
                        print(f"     - First item: {first_key}")
                else:
                    print(f"   - {key}: {type(value)}")
        
        imgs = state.get("img_data_clean")
        masks = state.get("mask_data_clean")
        # ---- robust path lists ----
        raw_paths = state.get("paths", [])
        if isinstance(raw_paths, dict):                # new nested layout
            img_paths = state.get("image_paths") or raw_paths.get("image", [])
            mask_paths = state.get("mask_paths") or raw_paths.get("mask",  [])
        else:                                          
            img_paths = state.get("image_paths") or list(raw_paths)
            mask_paths = state.get("mask_paths") or [p for p in raw_paths if p.lower().endswith(".png")]
            
            
        # ---------- FALLBACK: derive mask paths from mask_data -------------------
        if not mask_paths:                             # ← **add this block**
            for mask_name in masks.keys():             # e.g.  "Oil (210)"
                guess = RAW_MASK_DIR / f"{mask_name}.png"
                if guess.exists():
                    mask_paths.append(str(guess))
# -------------------------------------------------------------------------
   
# -------------------------------------------------

        if not imgs or not masks:
            print("🛑 No cleaned images or masks to explore.")
            return state
        
        state.setdefault("image_summaries", {})

        for name, img in imgs.items():
            mask_entry = masks.get(name)
            if not mask_entry:
                print(f"⚠️ No masks for image '{name}', skipping.")
                continue
            stem = Path(name).stem
            print(f"🔍 Exploring image: {stem}")

            # Compute per-class metrics
            metrics = compute_class_metrics(mask_entry)
            state["image_summaries"][stem] = metrics
            print("***************************************************")
            print(state["image_summaries"][stem])
            print("***************************************************")
            # Save JSON summary
            with open(EXP_DIR / f"{stem}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Generate visualizations per class
            for class_name, mask in mask_entry.items():
                title_base = f"{stem} – {class_name}"

                # 1) Overlay
                fig = plot_image_mask_overlay(img, mask, title=f"Overlay – {title_base}")
                save_figure_to_state(
                    state,
                    base64_image=fig_to_base64(fig),
                    plot_type="overlay",
                    title=f"{title_base} Overlay",
                    description=f"Mask overlay for class '{class_name}'",
                )
                print("***************************************************")
                print(f"Overlay saved for {title_base}")
                print("***************************************************")
                # 2) Color histogram (object pixels only)
                fig = plot_color_histogram(img, mask=mask, title=f"Color Histogram – {title_base}")
                save_figure_to_state(
                    state,
                    base64_image=fig_to_base64(fig),
                    plot_type="color_histogram",
                    title=f"{title_base} Color Histogram",
                    description=f"RGB distribution for class '{class_name}'",
                )
                print("***************************************************")
                print(f"Color histogram saved for {title_base}")
                print("***************************************************")

                # 3) Shape distribution (single class mask)
                fig = plot_shape_distribution(mask, title=f"Shape – {title_base}")
                save_figure_to_state(
                    state,
                    base64_image=fig_to_base64(fig),
                    plot_type="shape_distribution",
                    title=f"{title_base} Shape Distribution",
                    description=f"Shape metrics for class '{class_name}'",
                )
                print("***************************************************")
                print(f"Shape distribution saved for {title_base}")
                print("***************************************************")

                # 4) Texture (GLCM) analysis
                fig = plot_texture_glcm(img, mask, title=f"Texture – {title_base}")
                save_figure_to_state(
                    state,
                    base64_image=fig_to_base64(fig),
                    plot_type="texture_glcm",
                    title=f"{title_base} Texture Analysis",
                    description=f"Texture metrics for class '{class_name}'",
                )
                print("***************************************************")
                print(f"Texture analysis saved for {title_base}")
                print("***************************************************")

                # ---------- MOVE THIS IMAGE ----------
                norm_name = _normalize_key(name)
                # 🌟 Patch: try both raw/processed and append fallback with .jpg
                matched_image_path = None
                for p in img_paths:
                    path_obj = Path(p)
                    if _normalize_key(path_obj.stem) == norm_name:
                        matched_image_path = path_obj.resolve()
                        break

                # ⛑ Fallback: if not found, try adding .jpg explicitly
                if matched_image_path is None:
                    fallback = RAW_IMG_DIR / f"{name}.jpg"
                    if fallback.exists():
                        matched_image_path = fallback.resolve()

                if matched_image_path and matched_image_path.exists():
                    if matched_image_path.parent != PROCESSED_IMG_DIR.resolve():
                        dest = (PROCESSED_IMG_DIR / matched_image_path.name).resolve()
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(str(matched_image_path), str(dest))
                            print(f"[ImageLoader] 📁 moved image → {dest.relative_to(Path.cwd())}")
                        except shutil.Error as e:
                            print(f"[ImageLoader] ℹ️  move skipped: {e}")
                    else:
                        print(f"[ImageLoader] ℹ️  {matched_image_path.name} already in processed/")
                else:
                    print(f"[ImageLoader] ⚠️  No matching image file found for '{name}'")

                # ---------- MOVE THIS IMAGE'S MASK ----------
                matched_mask_path = next(
                    (Path(m).resolve() for m in mask_paths if _normalize_key(Path(m).stem) == norm_name),
                    None
                )

                if matched_mask_path and matched_mask_path.exists():
                    if matched_mask_path.parent != PROCESSED_MASK_DIR.resolve():
                        dest = (PROCESSED_MASK_DIR / matched_mask_path.name).resolve()
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(str(matched_mask_path), str(dest))
                            print(f"[ImageLoader] 📁 moved mask → {dest.relative_to(Path.cwd())}")
                        except shutil.Error as e:
                            print(f"[ImageLoader] ℹ️  move skipped: {e}")
                    else:
                        print(f"[ImageLoader] ℹ️  {matched_mask_path.name} already in processed/")
                else:
                    print(f"[ImageLoader] ⚠️  No matching mask file found for '{name}'")

            print(f"✅ Exploration completed for: {stem}")

        state["image_exploration_completed"] = True
        
        return state