"""Download and load the UCI Default of Credit Card Clients dataset.

Source: Yeh, I.-C. & Lien, C.-H. (2009). UCI Machine Learning Repository,
dataset id 350. https://doi.org/10.24432/C55S3H
"""

from pathlib import Path
import io
import urllib.request
import zipfile

import pandas as pd

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
RAW_FILENAME = "default of credit card clients.xls"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_raw(force=False):
    """Download the raw .xls file into data/raw/ if it isn't there already.

    Returns the path to the raw file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / RAW_FILENAME
    if raw_path.exists() and not force:
        return raw_path

    print(f"Downloading dataset from {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    archive.extract(RAW_FILENAME, RAW_DIR)
    print(f"Saved to {raw_path}")
    return raw_path


def load_raw():
    """Load the raw dataset as a DataFrame.

    The .xls has a two-row header (X1..X23 labels, then real column names),
    so we read with header=1 to get the descriptive names.
    """
    raw_path = download_raw()
    return pd.read_excel(raw_path, header=1)
