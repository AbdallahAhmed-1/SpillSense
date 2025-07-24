# agents/linker_agent.py
"""
LinkerAgent
===========

After every new file is registered in `file_catalog`, call::

    linker = LinkerAgent()
    linker.link_new_file(uuid, meta_dict)

`meta_dict` is the same dict you just passed to `insert_file_catalog()`
(contains timestamp, lat, lon, site_code, spill_number …).

The agent:

1. Pulls *candidate* files from `file_catalog` that are plausibly related.
2. Computes a simple heuristic score:
       +0.4  same spill_number          (Tier-2)
       +0.3  GPS within 100 m           (Tier-1)
       +0.2  same site_code             (Tier-1)
       +0.1  timestamps ≤ 1 h apart     (Tier-1)
3. For every candidate with `score ≥ 0.8`, writes a symmetric row into
   `link_table` via `insert_link(uuid_a, uuid_b, score)`.
4. Propagates `spill_number` / `well_id` between linked files if one side
   has the value and the other is NULL.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, Any, List

from db.db_utils import (
    get_connection,
    insert_link,
    upsert_incident_key,
)

# ---------------------------------------------------------------------------


class LinkerAgent:
    def __init__(
        self,
        time_window_hours: float = 1.0,
        spatial_window_m: float = 100.0,
        score_threshold: float = 0.8,
    ):
        self.time_window = timedelta(hours=time_window_hours)
        self.spatial_window_deg = spatial_window_m / 111_139.0  # rough deg↔︎m
        self.score_threshold = score_threshold

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def link_new_file(self, uuid: str, meta: Dict[str, Any]) -> None:
        """
        Link *uuid* against other files already in `file_catalog`.

        Parameters
        ----------
        uuid : str
            SHA-256 digest of the newly ingested file.
        meta : dict
            The metadata dict you inserted into `file_catalog`.
        """
        candidates = self._fetch_candidates(meta)
        for cand in candidates:
            cand_uuid = cand["uuid"]
            if cand_uuid == uuid:
                continue

            score = self._score_link(meta, cand)
            if score >= self.score_threshold:
                insert_link(uuid, cand_uuid, score)
                self._propagate_incident_keys(uuid, cand_uuid, meta, cand)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    # ---------- 1. Candidate retrieval ----------------------------------

    def _fetch_candidates(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pull rows that share time, site or spatial proximity."""
        clauses = ["uuid <> %s"]
        params = [meta["uuid"]]

        # ±1 h time window if timestamp present
        if meta.get("timestamp"):
            t0 = meta["timestamp"] - self.time_window
            t1 = meta["timestamp"] + self.time_window
            clauses.append("timestamp BETWEEN %s AND %s")
            params.extend([t0, t1])

        # Same site_code if available
        if meta.get("site_code"):
            clauses.append("site_code = %s")
            params.append(meta["site_code"])

        # Rough lat/lon bounding box if available
        if meta.get("lat") is not None and meta.get("lon") is not None:
            lat = meta["lat"]
            lon = meta["lon"]
            d = self.spatial_window_deg
            clauses.append("lat BETWEEN %s AND %s")
            clauses.append("lon BETWEEN %s AND %s")
            params.extend([lat - d, lat + d, lon - d, lon + d])

        where_sql = " AND ".join(clauses) if clauses else "1"
        sql = f"SELECT * FROM file_catalog WHERE {where_sql}"
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    # ---------- 2. Scoring heuristic ------------------------------------

    def _score_link(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        score = 0.0

        # Tier-2: same spill number
        if a.get("spill_number") and a["spill_number"] == b.get("spill_number"):
            score += 0.4

        # Tier-1: spatial proximity
        if (
            a.get("lat") is not None
            and b.get("lat") is not None
            and self._haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= 0.1
        ):
            score += 0.3

        # Tier-1: same site code
        if a.get("site_code") and a["site_code"] == b.get("site_code"):
            score += 0.2

        # Tier-1: timestamp proximity
        if a.get("timestamp") and b.get("timestamp"):
            dt = abs((a["timestamp"] - b["timestamp"]).total_seconds())
            if dt <= self.time_window.total_seconds():
                score += 0.1

        return round(score, 2)

    # ---------- 3. Incident-key propagation -----------------------------

    def _propagate_incident_keys(
        self,
        uuid_a: str,
        uuid_b: str,
        meta_a: Dict[str, Any],
        meta_b: Dict[str, Any],
    ):
        """
        If one side has spill_number or well_id and the other does not,
        push the value into the empty row.
        """
        if meta_a.get("spill_number") and not meta_b.get("spill_number"):
            upsert_incident_key(uuid_b, spill_number=meta_a["spill_number"])
        elif meta_b.get("spill_number") and not meta_a.get("spill_number"):
            upsert_incident_key(uuid_a, spill_number=meta_b["spill_number"])

        if meta_a.get("well_id") and not meta_b.get("well_id"):
            upsert_incident_key(uuid_b, well_id=meta_a["well_id"])
        elif meta_b.get("well_id") and not meta_a.get("well_id"):
            upsert_incident_key(uuid_a, well_id=meta_b["well_id"])

    # ---------- 4. Utilities -------------------------------------------

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres."""
        r = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dl = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        )
        return r * 2 * math.asin(math.sqrt(a))
