# === mat_loader.py ===
"""Load hyperspectral `.mat` cubes once from `data/raw/mat/`.
The raw MATLAB dict is stored under ``state["hsi_data"][filename]``.
Files remain in raw directory until processed by explorator.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import scipy.io as sio
from state.state_utils import OilGasRCAState

RAW_MAT_DIR = Path("data/raw/mat")

class MATLoader:
    """LangGraph node – expects ``state["paths"]`` list of .mat files."""
    def __init__(self, *, load_data: bool = True):
        self.load_data = load_data

    # ------------------------------------------------------------------
    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        state.setdefault("hsi_data", {})  # ensure key exists

        for p_str in state["paths"]:
            p = Path(p_str).resolve()
            if not p.exists():
                print(f"[MATLoader] 🚫 file not found: {p}")
                continue

            # 1️⃣ Load cube (optional)
            cube: Any | None = None
            if self.load_data:
                try:
                    cube = sio.loadmat(p)
                    if "img" in cube and "cube" not in cube:
                        cube["cube"] = cube.pop("img")
                    print(f"[MATLoader] ✅ loaded {p.name} keys={list(cube)[:4]}")
                except Exception as e:
                    print(f"[MATLoader] ⚠️ could not load {p.name}: {e}")

            # 2️⃣ Stash in state
            if cube is not None:
                state["hsi_data"][p.name] = cube

        # Paths remain unchanged - files stay in raw directory
        return state