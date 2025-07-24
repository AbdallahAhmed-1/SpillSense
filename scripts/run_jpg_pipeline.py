# scripts/run_jpg_pipeline.py

import os
import traceback
from pathlib import Path
from typing import cast, List, Tuple

from joblib import dump
from tools.img_graph_builder import build_image_graph
from state.state_utils import create_initial_state, save_state, OilGasRCAState
from state.session_data import session


def get_image_and_mask_paths(input_path: str) -> Tuple[List[str], List[str]]:
    """Find JPG image paths and corresponding mask paths in both raw/processed folders."""
    image_paths, mask_paths = [], []

    # Directories
    raw_dir = Path("data/raw/jpg")
    processed_dir = Path("data/processed/jpg")
    mask_raw_dir = Path("data/raw/masks")
    mask_processed_dir = Path("data/processed/masks")

    valid_img_exts = ['.jpg', '.jpeg']
    valid_mask_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

    for dir_path in [raw_dir, processed_dir]:
        for ext in valid_img_exts:
            image_paths.extend(str(p) for p in dir_path.glob(f"*{ext}"))

    for img_path in image_paths:
        img_stem = Path(img_path).stem
        for mask_dir in [mask_raw_dir, mask_processed_dir]:
            for ext in valid_mask_exts:
                candidate = mask_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    mask_paths.append(str(candidate))
                    break

    return image_paths, mask_paths


def run_jpg_pipeline(input_path: str, report_path: str = "Reports"):
    """
    Executes Phase 1 of the RCA pipeline for JPG processing:
    loading, cleaning, exploring, and generating a PDF report.
    Saves state to disk for use in modeling and inference.
    """
    os.makedirs(report_path, exist_ok=True)
    os.makedirs("state", exist_ok=True)
    print("🖼️ Starting Phase 1 JPG/HSI pipeline...")

    image_paths, mask_paths = get_image_and_mask_paths(input_path)
    initial_state = create_initial_state(
        modality="jpg",
        data_path=input_path,
        pdf_report_path=report_path,
    )
    initial_state["image_paths"] = image_paths
    initial_state["mask_paths"] = mask_paths

    try:
        graph = build_image_graph()
        final_state = graph.invoke(initial_state)

        print("\n✅ Phase 1 completed: JPG data loaded, cleaned, and explored.")

        # Save intermediate debug state
        intermediate_path = os.path.join(report_path, "state_after_jpg_eda.joblib")
        dump(final_state, intermediate_path)
        print(f"🧠 Debug state saved to: {intermediate_path}")

        # Save main state
        state_typed = cast(OilGasRCAState, final_state)
        save_state(state_typed)
        session.state = state_typed

        # Save modeling state
        dump(final_state, "state/image_state.joblib")
        print("💾 Modeling state saved to: state/image_state.joblib")

    except Exception as e:
        print(f"\n🚨 Error running Phase 1 JPG pipeline: {str(e)}")
        traceback.print_exc()


def main():
    run_jpg_pipeline("data/raw/", "Reports")  # Or use a specific file path


if __name__ == "__main__":
    main()