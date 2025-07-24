# === agents/mat_explorator.py ================================================
"""
Generate visual insights for each cleaned hyperspectral cube and stash
figures as base64 strings so the Reporter/LLM can embed them.
After exploration, moves files from raw to processed directories.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np

from state.state_utils import OilGasRCAState, save_figure_to_state
from tools.hsi_tools import (
    plot_mean_spectrum,
    plot_band_variance,
    plot_pca_scatter,
    plot_rgb_composite,
    plot_ndvi_map,
    fig_to_base64,
)

# Directory constants
RAW_MAT_DIR = Path("data/raw/mat")
PROCESSED_MAT_DIR = Path("data/processed/mat")
EXP_DIR = Path("data/processed/mat/explore").absolute()
EXP_DIR.mkdir(parents=True, exist_ok=True)


class MATExplorator:
    def __init__(self, llm=None):
        self.llm = llm

    def _move_files_to_processed(self, state: OilGasRCAState) -> OilGasRCAState:
        """Move .mat files from raw/ to processed/ directory after exploration."""
        processed: list[str] = []
        
        # Get paths from state, default to empty list if not present
        paths = state.get("paths", [])
        if not paths:
            print("[MATExplorator] No paths found in state for moving files")
            return state
        
        for p_str in paths:
            p = Path(p_str).resolve()
            if not p.exists():
                print(f"[MATExplorator] 🚫 file not found: {p}")
                continue

            # Move to processed/
            dest = PROCESSED_MAT_DIR / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(dest))
                try:
                    rel = dest.relative_to(Path.cwd())
                except ValueError:
                    rel = dest
                print(f"[MATExplorator] 📁 moved → {rel}")
            except shutil.Error:
                print(f"[MATExplorator] ℹ️ {p.name} already in processed/")
            
            processed.append(str(dest))

        # Update paths for downstream nodes
        if processed:
            state["paths"] = processed
        
        return state

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        print(f"[MATExplorator] DEBUG: Available state keys: {list(state.keys())}")
        
        # First, do the exploration if cleaned data exists
        if not state.get("hsi_data_clean"):
            print("🛑 No cleaned HSI cubes to explore.")
            
            state["mat_analysis_completed"] = True
            # Still try to move files if they exist
            print("[MATExplorator] Moving files to processed directory...")
            state = self._move_files_to_processed(state)
            return state

        state.setdefault("hsi_summaries", {})

        # default NDVI bands from analysis_params or fallback
        ndvi_red   = state.get("analysis_params", {}).get("ndvi_red_band", 29)
        ndvi_nir   = state.get("analysis_params", {}).get("ndvi_nir_band", 50)

        for fname, arte in state["hsi_data_clean"].items():
            cube     = arte["clean_cube"]
            pca_cube = arte["pca_cube"]
            stem     = Path(fname).stem
            print(f"🔍 Exploring cube: {stem}")

            # 1) Mean spectrum
            fig = plot_mean_spectrum(cube, title=f"Mean Spectrum – {stem}")
            save_figure_to_state(
                state,
                base64_image=fig_to_base64(fig),
                plot_type="mean_spectrum",
                title=f"{stem} – Mean Spectrum",
                description="Mean ± 1σ across bands",
            )

            # 2) Band variance
            fig = plot_band_variance(cube, title=f"Band Variance – {stem}")
            save_figure_to_state(
                state,
                base64_image=fig_to_base64(fig),
                plot_type="band_variance",
                title=f"{stem} – Band Variance",
                description="Variance per spectral band",
            )

            # 3) PCA scatter
            fig = plot_pca_scatter(cube, title=f"PCA Scatter – {stem}")
            save_figure_to_state(
                state,
                base64_image=fig_to_base64(fig),
                plot_type="pca_scatter",
                title=f"{stem} – PCA Scatter",
                description="First two PCA components of pixels",
            )

            # 4) RGB composite
            fig = plot_rgb_composite(
                pca_cube,
                rgb_indices=(0, 1, 2),               # use first three PCA components
                title=f"RGB (PCA1-3) – {stem}"
            )
            save_figure_to_state(
                state,
                base64_image=fig_to_base64(fig),
                plot_type="rgb_composite",
                title=f"{stem} – RGB Composite",
                description="False-color image using PCA components 1-3",
            )

            # 5) NDVI map
            try:
                fig = plot_ndvi_map(
                    cube,
                    nir_band=ndvi_nir,
                    red_band=ndvi_red,
                    title=f"NDVI – {stem}" ,
                )
                save_figure_to_state(
                    state,
                    base64_image=fig_to_base64(fig),
                    plot_type="ndvi_map",
                    title=f"{stem} – NDVI",
                    description=f"NDVI using NIR={ndvi_nir}, Red={ndvi_red}",
                )
            except Exception as e:
                print(f"⚠️  {stem} – NDVI failed: {e}")

            # record summary
            state["hsi_summaries"][stem] = {
                "bands_kept": arte["bands_kept"],
                "ndvi_red_band": ndvi_red,
                "ndvi_nir_band": ndvi_nir,
            }

            # save JSON summary
            with open(EXP_DIR / f"{stem}_summary.json", "w") as f:
                json.dump(state["hsi_summaries"][stem], f, indent=2)

            print(f"✅ Explorator finished: {stem}")

        state["exploration_completed"] = True

        # After exploration is complete, move files to processed directory
        print("[MATExplorator] Moving files to processed directory...")
        state = self._move_files_to_processed(state)

        return state