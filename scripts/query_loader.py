# query_loader.py
"""
Fetch all artefacts (CSV, MAT, JPG) that belong to a spill / site / uuid
by querying `file_catalog` and `link_table`.

Usage
-----
from query_loader import load_artefacts

# A)  Drill-down into one incident
files = load_artefacts(spill_number="100045", site_code="GM18")

# B)  Pull everything linked to a single raw file UUID
files = load_artefacts(uuid="4f9ad…")

# C)  Fleet-wide sample in a date window
files = load_artefacts(date_from="2024-01-01", date_to="2024-06-30")
"""

from __future__ import annotations
from typing import Dict, List
from db.db_utils import get_linked_files
import argparse
import json


def _group_by_modality(rows: List[dict]) -> Dict[str, List[str]]:
    """
    rows -> {'csv': [paths], 'mat': [...], 'jpg': [...]}
    """
    buckets = {"csv": [], "mat": [], "jpg": []}
    for r in rows:
        m = r["file_type"]
        if m in buckets:
            buckets[m].append(r["file_path"])
    return buckets


def load_artefacts(
    *,
    uuid: str | None = None,
    spill_number: str | None = None,
    site_code: str | None = None,
    date_from: str | None = None,  # 'YYYY-MM-DD' or datetime
    date_to: str | None = None,
    min_link_score: float = 0.8,
) -> Dict[str, List[str]]:
    """
    Returns
    -------
    dict
        Keys: 'csv', 'mat', 'jpg' → list of absolute file paths
    """
    filters = {
        k: v
        for k, v in {
            "uuid": uuid,
            "spill_number": spill_number,
            "site_code": site_code,
            "date_from": date_from,
            "date_to": date_to,
        }.items()
        if v is not None
    }

    rows = get_linked_files(filters, min_score=min_link_score)
    return _group_by_modality(rows)


# ------------------------------------------------------------------------
# Convenience wrappers (optional)
# ------------------------------------------------------------------------

def load_by_incident(spill_number: str, site_code: str | None = None):
    return load_artefacts(spill_number=spill_number, site_code=site_code)


def load_by_uuid(uuid: str):
    return load_artefacts(uuid=uuid)


# CLI test:  python -m query_loader --spill 100045 --site GM18
if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--uuid")
    p.add_argument("--spill")
    p.add_argument("--site")
    p.add_argument("--from-date")
    p.add_argument("--to-date")
    p.add_argument("--min-score", type=float, default=0.8)
    args = p.parse_args()

    files = load_artefacts(
        uuid=args.uuid,
        spill_number=args.spill,
        site_code=args.site,
        date_from=args.from_date,
        date_to=args.to_date,
        min_link_score=args.min_score,
    )
    print(json.dumps(files, indent=2))
