import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv
import os

# Load DB credentials from .env
load_dotenv()

config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}


# SQL Table definitions
TABLES = {}

TABLES['files_metadata'] = """
CREATE TABLE IF NOT EXISTS files_metadata (
    file_id          VARCHAR(64) PRIMARY KEY,
    file_name        VARCHAR(255),
    file_path        TEXT,
    modality         ENUM('csv', 'mat', 'jpg'),
    upload_time      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    linked_scan_id   VARCHAR(64),
    linked_scene_id  VARCHAR(64),
    additional_meta  JSON
)
"""

TABLES['scan_results'] = """
CREATE TABLE IF NOT EXISTS scan_results (
    scan_id          VARCHAR(64),
    file_id          VARCHAR(64),
    summary_json     JSON,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(scan_id, file_id),
    FOREIGN KEY(file_id) REFERENCES files_metadata(file_id)
)
"""

TABLES['scene_metadata'] = """
CREATE TABLE IF NOT EXISTS scene_metadata (
    scene_id     VARCHAR(100) PRIMARY KEY,
    uuid         VARCHAR(100) UNIQUE,
    site_code    VARCHAR(50),
    spill_number VARCHAR(50),
    spill_date   DATE,
    csv_path     TEXT,
    hsi_path     TEXT,
    img_path     TEXT,
    csv_file_id  VARCHAR(36),
    hsi_file_id  VARCHAR(36),
    img_file_id  VARCHAR(36),
    scene_status VARCHAR(20) DEFAULT 'new',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
"""

TABLES['file_catalog'] = """
CREATE TABLE IF NOT EXISTS file_catalog (
    uuid          CHAR(64) PRIMARY KEY,
    file_name     VARCHAR(255),
    file_path     TEXT,
    file_type     ENUM('csv','mat','jpg'),
    timestamp     DATETIME NULL,
    lat           DOUBLE NULL,
    lon           DOUBLE NULL,
    sensor        VARCHAR(32) NULL,
    site_code     VARCHAR(16) NULL,
    spill_number  VARCHAR(16) NULL,
    well_id       VARCHAR(16) NULL,
    INDEX idx_time_site (timestamp, site_code),
    INDEX idx_spill (spill_number),
    INDEX idx_site_time (site_code, timestamp)
)
"""

TABLES['link_table'] = """
CREATE TABLE IF NOT EXISTS link_table (
    uuid_a         CHAR(64),
    uuid_b         CHAR(64),
    link_confidence DECIMAL(3,2),
    PRIMARY KEY (uuid_a, uuid_b),
    FOREIGN KEY (uuid_a) REFERENCES file_catalog(uuid),
    FOREIGN KEY (uuid_b) REFERENCES file_catalog(uuid)
)
"""

# Connect to MySQL and create tables
try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    for table_name, ddl in TABLES.items():
        print(f"Creating table `{table_name}`... ", end='')
        cursor.execute(ddl)
        print("OK ✅")

    cursor.close()
    conn.close()

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("❌ Invalid credentials.")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("❌ Database does not exist.")
    else:
        print(err)
