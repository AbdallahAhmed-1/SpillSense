# scripts/process_scene.py
"""
Orchestrator: Dispatch artefacts to the appropriate Phase-1 pipelines
(csv, mat, and future modalities). Runs pipelines, collects status, 
and prints summary.
"""

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List

from db.db_utils import load_artefacts 

_LOG = logging.getLogger("orchestrate_phase1")
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    datefmt="%H:%M:%S"
)

# Register your modality pipelines here.
PIPELINE_REGISTRY: Dict[str, str] = {
    "csv": "scripts.run_phase1:run_csv_pipeline",
    "mat": "scripts.hsi_workflow:run_hsi_pipeline",
    "jpg": "scripts.run_jpg_pipeline:run_jpg_pipeline",
}

def _load_runner(dotted: str) -> Callable[..., None]:
    """Import and return the function referenced by 'module:function'."""
    module_name, func_name = dotted.split(":")
    func = getattr(import_module(module_name), func_name)
    return func

def _ensure_reports_dir() -> str:
    reports_dir = Path("Reports")
    reports_dir.mkdir(exist_ok=True)
    return str(reports_dir)

def dispatch(buckets: Dict[str, List[str]]) -> None:
    """Run each modality's Phase-1 pipeline sequentially."""
    reports_dir = _ensure_reports_dir()

    summary = []
    for mod, paths in buckets.items():
        if not paths:
            continue
        if mod not in PIPELINE_REGISTRY:
            _LOG.warning("No pipeline registered for modality %s; skipping", mod)
            summary.append((mod, "no_runner", len(paths)))
            continue

        runner = _load_runner(PIPELINE_REGISTRY[mod])
        _LOG.info("▶ Running %s pipeline on %d file(s)…", mod.upper(), len(paths))

        try:
            if mod == "csv":
                # One file at a time
                for p in paths:
                    runner(p, reports_dir)
            elif mod == "jpg":
                # One file at a time for JPG
                for p in paths:
                    runner(p, reports_dir)
            else:  # mat or other
                runner(paths, reports_dir)
            _LOG.info("✓ %s pipeline finished", mod.upper())
            summary.append((mod, "success", len(paths)))
        except Exception as exc:
            _LOG.exception("✗ %s pipeline failed: %s", mod.upper(), exc)
            summary.append((mod, "error", len(paths)))

    # Print summary after all pipelines run
    print("\n==== PHASE 1 PIPELINE SUMMARY ====")
    for mod, status, count in summary:
        print(f"  - {mod.upper()}: {count} files → {status}")
    print("==================================")

def cli():
    ap = argparse.ArgumentParser("Process artefacts via Phase‑1 pipelines.")
    ap.add_argument("--uuid")
    ap.add_argument("--spill")
    ap.add_argument("--site")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--min-score", type=float, default=0.8)
    args = ap.parse_args()

    artefacts = load_artefacts(
        uuid=args.uuid,
        spill_number=args.spill,
        site_code=args.site,
        date_from=args.date_from,
        date_to=args.date_to,
        min_link_score=args.min_score,
    )
    buckets = {k: v for k, v in artefacts.items() if v}

    if not buckets:
        _LOG.warning("No artefacts matched the filters.")
        return

    _LOG.info("Buckets: %s", {k: len(v) for k, v in buckets.items()})
    dispatch(buckets)

if __name__ == "__main__":
    cli()
