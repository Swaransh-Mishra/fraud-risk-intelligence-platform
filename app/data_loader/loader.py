from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def load_transactions(dataset_path: Path) -> pd.DataFrame:
    """Load and combine transaction records from the dataset archive."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {dataset_path}"
        )

    with ZipFile(dataset_path) as archive:
        pickle_files = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".pkl")
        )

        if not pickle_files:
            raise ValueError("No pickle files found in the dataset archive.")

        transactions = [
            pd.read_pickle(archive.open(file_name))
            for file_name in pickle_files
        ]

    return pd.concat(transactions, ignore_index=True)