#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download raw data from Zenodo (DOI: 10.5281/zenodo.20025878).
Files are saved to data/raw/monthly/ and data/raw/financial/.
"""

import requests
from pathlib import Path
from tqdm import tqdm
import time

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ZENODO_API = "https://zenodo.org/api/records/20032245"   # DOI record ID
OUTPUT_DIRS = {
    "monthly": PROJECT_ROOT / "data" / "raw" / "monthly",
    "financial": PROJECT_ROOT / "data" / "raw" / "financial",
}

for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
def classify_file(filename: str) -> str:
    """
    Classify file into 'monthly' or 'financial' based on filename.
    - Contains '三因子' -> monthly
    - Contains '资产' or '利润' -> financial
    - Everything else -> monthly (default)
    """
    if "三因子" in filename:
        return "monthly"
    if "资产" in filename or "利润" in filename:
        return "financial"
    # 默认归入 monthly
    return "monthly"

def download_file(url: str, dest_path: Path) -> bool:
    """Download a single file with progress bar."""
    if dest_path.exists():
        print(f"Skipping (already exists): {dest_path.name}")
        return True
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded: {dest_path.name}")
        return True
    except Exception as e:
        print(f"Failed to download {dest_path.name}: {e}")
        return False

# ----------------------------------------------------------------------
def main():
    print("Fetching Zenodo record...")
    resp = requests.get(ZENODO_API)
    resp.raise_for_status()
    record = resp.json()

    files = record.get("files", [])
    print(f"Found {len(files)} files in the record.\n")

    success = 0
    for file_info in files:
        key = file_info["key"]
        url = file_info["links"]["self"]
        dest_dir = OUTPUT_DIRS[classify_file(key)]
        dest_path = dest_dir / key
        if download_file(url, dest_path):
            success += 1
        time.sleep(0.2)   # brief pause to be polite to the server

    print(f"\n✅ Successfully downloaded {success}/{len(files)} files.")
    print("Raw data is now ready in `data/raw/`.\n")

if __name__ == "__main__":
    main()