import mysql.connector
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os

# Load DB credentials from .env
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

print("Connecting to DB:", DB_CONFIG['database'])  # For debugging only


def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print("❌ Database connection failed:", err)
        raise


def scene_exists(scene_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scene_metadata WHERE scene_id = %s", (scene_id,))
    exists = cursor.fetchone()[0] > 0
    cursor.close()
    conn.close()
    return exists


def insert_scene_record(scene_info):
    """
    Insert a new scene into `scene_metadata`. Assumes the scene_id is unique.
    
    If scene_id already exists:
      - It will update any non-null fields from the new record,
        such as csv_file_id, mat_file_id, jpg_file_id, etc.
        
    Fields:
        - scene_id, site_code, spill_number, spill_date
        - csv_file_id, hsi_file_id, img_file_id
    """
    
    conn = get_connection()
    cursor = conn.cursor()

    scene_uuid = str(uuid.uuid4())

    sql = """
        INSERT INTO scene_metadata (
            scene_id, uuid, site_code, spill_number, spill_date,
            csv_file_id, hsi_file_id, img_file_id, scene_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            csv_file_id = IFNULL(VALUES(csv_file_id), csv_file_id),
            hsi_file_id = IFNULL(VALUES(hsi_file_id), hsi_file_id),
            img_file_id = IFNULL(VALUES(img_file_id), img_file_id),
            scene_status = VALUES(scene_status)
    """
    values = (
        scene_info["scene_id"],
        scene_uuid,
        scene_info["site_code"],
        scene_info["spill_number"],
        scene_info["spill_date"],
        scene_info.get("csv_file_id"),
        scene_info.get("hsi_file_id"),
        scene_info.get("img_file_id"),
        scene_info.get("scene_status", "new")
    )

    cursor.execute(sql, values)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Scene {scene_info['scene_id']} inserted or updated.")
    return True

# === NEW / REWRITTEN HELPERS ===============================================

def insert_file_catalog(meta: dict) -> str:
    """
    Insert (or update) a row in `file_catalog`.

    Expected keys in `meta`:
        uuid, file_name, file_path, file_type,
        timestamp, lat, lon, sensor,
        site_code, spill_number, well_id
    Any missing key will be inserted as NULL and later updatable.
    """
    required = {"uuid", "file_name", "file_path", "file_type"}
    if not required.issubset(meta):
        raise ValueError(f"insert_file_catalog: missing keys {required - meta.keys()}")

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO file_catalog (
            uuid, file_name, file_path, file_type,
            timestamp, lat, lon, sensor,
            site_code, spill_number, well_id
        )
        VALUES (
            %(uuid)s, %(file_name)s, %(file_path)s, %(file_type)s,
            %(timestamp)s, %(lat)s, %(lon)s, %(sensor)s,
            %(site_code)s, %(spill_number)s, %(well_id)s
        )
        ON DUPLICATE KEY UPDATE
            -- only overwrite if NEW value is NOT NULL
            file_name     = IFNULL(VALUES(file_name),  file_name),
            file_path     = IFNULL(VALUES(file_path),  file_path),
            file_type     = IFNULL(VALUES(file_type),  file_type),
            timestamp     = IFNULL(VALUES(timestamp),  timestamp),
            lat           = IFNULL(VALUES(lat),        lat),
            lon           = IFNULL(VALUES(lon),        lon),
            sensor        = IFNULL(VALUES(sensor),     sensor),
            site_code     = IFNULL(VALUES(site_code),  site_code),
            spill_number  = IFNULL(VALUES(spill_number), spill_number),
            well_id       = IFNULL(VALUES(well_id),    well_id)
    """
    cursor.execute(sql, meta)
    conn.commit()
    cursor.close()
    conn.close()
    return meta["uuid"]


def insert_link(uuid_a: str, uuid_b: str, score: float) -> None:
    """
    Insert a symmetric link between two file UUIDs.
    Keeps the *highest* confidence if the pair already exists.
    """
    if uuid_a == uuid_b:
        return  # don't link a file to itself
    
    if not uuid_a or not uuid_b:
        return

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO link_table (uuid_a, uuid_b, link_confidence)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            link_confidence = GREATEST(link_confidence, VALUES(link_confidence))
    """

    cursor.execute(sql, (uuid_a, uuid_b, score))
    cursor.execute(sql, (uuid_b, uuid_a, score))          # make it symmetric
    conn.commit()
    cursor.close()
    conn.close()


def get_linked_files(filters: dict, min_score: float = 0.8) -> list[dict]:
    """
    Retrieve file_catalog rows that match the filters **or**
    are linked (above `min_score`) to a UUID that matches.

    Supported filters keys:
        uuid, spill_number, site_code, date_from, date_to
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # ---------- 1. Build WHERE clause on file_catalog -----------------------
    where_clauses = []
    params = []

    if "spill_number" in filters:
        where_clauses.append("fc.spill_number = %s")
        params.append(filters["spill_number"])

    if "site_code" in filters:
        where_clauses.append("fc.site_code = %s")
        params.append(filters["site_code"])

    if "date_from" in filters:
        where_clauses.append("fc.timestamp >= %s")
        params.append(filters["date_from"])

    if "date_to" in filters:
        where_clauses.append("fc.timestamp <= %s")
        params.append(filters["date_to"])

    # ---------- 2. If a UUID filter is given, join via link_table ----------
    if "uuid" in filters:
        uuid_val = filters["uuid"]
        sql = """
            SELECT fc.*
            FROM link_table lt
            JOIN file_catalog fc
              ON fc.uuid = lt.uuid_b
            WHERE lt.uuid_a = %s
              AND lt.link_confidence >= %s
            UNION
            SELECT * FROM file_catalog WHERE uuid = %s
        """
        params = [uuid_val, min_score, uuid_val]
        cursor.execute(sql, params)

    else:
        # Pure catalog query with optional WHERE conditions
        base_sql = "SELECT * FROM file_catalog fc"
        if where_clauses:
            base_sql += " WHERE " + " AND ".join(where_clauses)
        cursor.execute(base_sql, params)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def upsert_incident_key(uuid: str, spill_number: str | None = None, well_id: str | None = None) -> None:
    """
    Populate / update Tier-2 incident keys for a given file UUID
    but never overwrite an existing non-NULL value.
    """
    if spill_number is None and well_id is None:
        return  # nothing to do

    conn = get_connection()
    cursor = conn.cursor()

    set_clauses = []
    params = []

    if spill_number is not None:
        set_clauses.append("spill_number = IFNULL(spill_number, %s)")
        params.append(spill_number)

    if well_id is not None:
        set_clauses.append("well_id = IFNULL(well_id, %s)")
        params.append(well_id)

    sql = f"UPDATE file_catalog SET {', '.join(set_clauses)} WHERE uuid = %s"
    params.append(uuid)
    cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    conn.close()


def get_file_by_id(file_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM files_metadata WHERE file_id = %s", (file_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result


def get_latest_files(limit=10):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM files_metadata ORDER BY upload_time DESC LIMIT %s", (limit,)
    )
    results = cursor.fetchall()

    cursor.close()
    conn.close()
    return results


def get_scene_by_id(scene_id):
    """
    Fetch a single scene record from scene_metadata by scene_id.
    Returns a dictionary or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM scene_metadata WHERE scene_id = %s"
    cursor.execute(query, (scene_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result

def load_artefacts(uuid=None, spill_number=None, site_code=None, date_from=None, date_to=None, min_link_score=0.8):
    """
    Mimics the output of the previous query_loader.load_artefacts, 
    grouping file paths by modality for the orchestrator.
    Returns: { "csv": [<csv paths>], "mat": [<mat paths>], ... }
    """
    filters = {}
    if uuid:         filters["uuid"] = uuid
    if spill_number: filters["spill_number"] = spill_number
    if site_code:    filters["site_code"] = site_code
    if date_from:    filters["date_from"] = date_from
    if date_to:      filters["date_to"] = date_to

    # Pull matching file_catalog rows
    rows = get_linked_files(filters, min_score=min_link_score)

    artefact_buckets = {"csv": [], "mat": [], "jpg": []}

    for row in rows:
        fpath = row["file_path"]
        ftype = row["file_type"].lower()
        # Map file_type (assuming values like "csv", "mat", "jpg")
        if ftype in artefact_buckets:
            artefact_buckets[ftype].append(fpath)
        # Optionally add any new modalities here (e.g., tif)
    
    # Remove empty buckets
    artefact_buckets = {k: v for k, v in artefact_buckets.items() if v}
    return artefact_buckets