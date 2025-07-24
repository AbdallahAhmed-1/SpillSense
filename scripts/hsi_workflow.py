"""scripts/hsi_workflow.py
===========================
Phase‑1 entry point for the **hyperspectral (.mat)** pipeline.

Mirrors `scripts/run_phase1.py` (CSV) but calls the MAT‑specific
LangGraph builder and seeds the `OilGasRCAState` with `modality='mat'`
plus the list of cube paths.
"""
from __future__ import annotations

import os
import glob
import traceback
import datetime
from typing import List, cast
from pathlib import Path

from tools.hsi_graph_builder import build_hsi_graph
from state.state_utils import create_initial_state, save_state, OilGasRCAState
from state.session_data import session
from joblib import dump

# ---------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------


def run_hsi_pipeline(mat_paths: List[str], report_path: str = "Reports") -> None:
    """Execute Phase‑1 (load → clean → explore → report) for HSI cubes."""

    if not mat_paths:
        raise ValueError("No .mat paths provided to run_hsi_pipeline().")

    # Ensure report directory exists
    os.makedirs(report_path, exist_ok=True)

    print("🧊📊 Starting HSI Phase‑1 pipeline on", len(mat_paths), "cube(s)…")

    # Compile graph once per call; caching happens inside builder
    graph = build_hsi_graph()

    # Create an initial cross‑modality state
    initial_state = create_initial_state(
        modality="mat",
        paths=[str(Path(p).resolve()) for p in mat_paths],
        pdf_report_path=report_path,
    )

    try:
        final_state = graph.invoke(initial_state)
        print("\n✅ HSI Phase‑1 completed: cubes loaded, cleaned, explored.")

        # Persist intermediate state next to the report dir
        inter_path = Path(report_path) / "hsi_state_after_eda.joblib"
        dump(final_state, inter_path)
        print(f"🧠 Intermediate state saved to: {inter_path}")

        # Save main state for Phase‑2 / Phase‑3 reuse
        state_typed = cast(OilGasRCAState, final_state)
        save_state(state_typed, path="state/rca_state_hsi.joblib")
        session.state = state_typed

    except Exception as e:
        print(f"\n🚨 Error running HSI Phase‑1 pipeline: {e}")
        traceback.print_exc()
        
        
def analyze_hsi_dataset(mat_dir: str = "data/processed/mat",
                        output_root: str = "static/reports") -> str:
    """
    Finds all .mat files in `mat_dir`, runs the Phase-1 HSI pipeline,
    and returns the basename of the generated PDF report.
    """
    mats = glob.glob(os.path.join(mat_dir, "*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files in {mat_dir}")

    # Each run gets its own timestamped sub-folder
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_root, f"hsi_{stamp}")
    os.makedirs(report_dir, exist_ok=True)

    # ➊ Run the full pipeline (loads → cleans → explores → writes PDF)
    run_hsi_pipeline(mats, report_dir)

    # ➋ Find the PDF that the graph produced
    pdfs = list(Path(report_dir).glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("HSI pipeline finished but no PDF was produced.")
    return pdfs[0].name           # only the basename; backend will move it        


# ---------------------------------------------------------------------
# CLI wrapper for ad‑hoc runs
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Run Phase‑1 HSI pipeline on .mat file(s)")
    p.add_argument("paths", nargs="+", help="one or more .mat file paths")
    p.add_argument("--report-dir", default="Reports", help="output dir for PDF & state")
    args = p.parse_args()

    run_hsi_pipeline(args.paths, args.report_dir)
    
    
    
