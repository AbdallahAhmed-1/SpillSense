# tools/metadata_extractors.py

from pathlib import Path
from datetime import datetime
import scipy.io
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def extract_csv_meta(path: Path) -> dict:
    # Heuristic: extract spill number or date from filename
    parts = path.stem.split("_")
    meta = {}

    if len(parts) >= 2:
        meta["site_code"] = parts[0]
        meta["spill_number"] = parts[1]

    # Optional: look inside CSV to pull spill_date or more
    meta["timestamp"] = None
    meta["lat"] = None
    meta["lon"] = None
    meta["sensor"] = None
    meta["well_id"] = None
    return meta


def extract_mat_meta(path: Path) -> dict:
    meta = {
        "timestamp": None, "lat": None, "lon": None,
        "sensor": "HSI", "site_code": None,
        "spill_number": None, "well_id": None
    }

    try:
        mat = scipy.io.loadmat(path)
        if "metadata" in mat:
            md = mat["metadata"]
            # Try to parse timestamp or lat/lon here
            # Example: meta["timestamp"] = datetime.strptime(str(md["acquisition_time"]), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"⚠️ Failed to parse .mat metadata for {path.name}: {e}")
    return meta


def extract_jpg_meta(path: Path) -> dict:
    meta = {
        "timestamp": None, "lat": None, "lon": None,
        "sensor": "Camera", "site_code": None,
        "spill_number": None, "well_id": None
    }

    try:
        image = Image.open(path)
        exif = image._getexif()

        if exif:
            for tag, value in exif.items():
                tag_name = TAGS.get(tag)
                if tag_name == "DateTimeOriginal":
                    meta["timestamp"] = datetime.strptime(value, "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")
                elif tag_name == "GPSInfo":
                    gps_info = {}
                    for t in value:
                        subtag = GPSTAGS.get(t)
                        gps_info[subtag] = value[t]
                    # You can decode lat/lon here if available
    except Exception as e:
        print(f"⚠️ No EXIF metadata for {path.name}: {e}")
    return meta
