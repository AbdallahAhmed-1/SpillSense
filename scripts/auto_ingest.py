# scripts/auto_ingest.py

import hashlib
from pathlib import Path
from db.db_utils import insert_file_catalog
from agents.linker_agent import LinkerAgent
from tools.metadata_extractors import extract_csv_meta, extract_mat_meta, extract_jpg_meta  # <- you'll create this module
from scripts.file_scanner import load_all_raw_files, print_summary
import uuid
from datetime import datetime

RAW_CSV_DIR = "data/raw/csv/"
RAW_HSI_DIR = "data/raw/mat/"
RAW_IMG_DIR = "data/raw/jpg/"


def register_file_in_db(file_info, file_type):
    meta = {
        "uuid": str(uuid.uuid4()),
        "file_name": file_info["name"],
        "file_path": file_info["path"],
        "file_type": file_type,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "lat": None,
        "lon": None,
        "sensor": None,
        "site_code": None,
        "spill_number": None,
        "well_id": None,
    }
    try:
        insert_file_catalog(meta)
        print(f"🗃️  Registered in DB: {meta['file_name']}")
    except Exception as e:
        print(f"❌ Could not insert {meta['file_name']} in DB: {e}")


def compute_uuid(file_path: str) -> str:
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash


def register_file(file_path: str, file_type: str, extractor_func) -> str:
    path = Path(file_path)
    meta = extractor_func(path)

    meta["uuid"] = compute_uuid(file_path)
    meta["file_name"] = path.name
    meta["file_path"] = str(path)
    meta["file_type"] = file_type

    insert_file_catalog(meta)
    print(f"✅ Registered {file_type.upper()} file: {path.name} → {meta['uuid']}")
    return meta["uuid"], meta


def scan_and_link():
    files = load_all_raw_files()
    print_summary(files)
    linker = LinkerAgent()
    # Process all types found by scanner
    for file_type, extractor_func in [
        ("csv", extract_csv_meta),
        ("mat", extract_mat_meta),
        ("jpg", extract_jpg_meta)
    ]:
        for file_info in files[file_type]:
            uuid, meta = register_file(file_info["path"], file_type, extractor_func)
            linker.link_new_file(uuid, meta)


if __name__ == "__main__":
    scan_and_link()
