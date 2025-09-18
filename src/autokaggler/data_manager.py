"""Dataset management utilities for the AutoKaggler agent."""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"


@dataclass
class DataMeta:
    """Metadata describing the dataset that was loaded."""

    source: str
    location: str
    additional: Dict[str, str]


class DataManager:
    """Handle retrieval and caching of Titanic datasets."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / "titanic"
        self.submission_dir = self.cache_dir / "submissions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)

    def prepare_datasets(
        self, prefer_source: str = "auto", force_download: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        """Return Titanic train and test dataframes plus metadata.

        Parameters
        ----------
        prefer_source:
            ``"auto"`` will attempt Kaggle download first and fallback to the
            bundled sample dataset. ``"kaggle"`` forces Kaggle (raising on
            failure) and ``"sample"`` forces the bundled dataset.
        force_download:
            When ``True`` any cached Kaggle download is ignored and files are
            re-fetched.
        """

        prefer_source = prefer_source or "auto"
        if prefer_source not in {"auto", "kaggle", "sample"}:
            logging.warning("Unknown data source '%s'; defaulting to 'auto'", prefer_source)
            prefer_source = "auto"

        if prefer_source == "sample":
            return self._load_sample()

        if prefer_source in {"auto", "kaggle"}:
            try:
                return self._load_kaggle(force_download=force_download)
            except Exception as exc:
                if prefer_source == "kaggle":
                    logging.error("Kaggle download requested but failed: %s", exc)
                    raise
                logging.warning(
                    "Falling back to bundled sample dataset due to Kaggle error: %s", exc
                )
        return self._load_sample()

    def _load_kaggle(
        self, force_download: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        """Download from Kaggle (if needed) and return the datasets."""

        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"

        if force_download or not train_path.exists() or not test_path.exists():
            self._download_from_kaggle()

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        meta = DataMeta(
            source="kaggle",
            location=str(self.data_dir),
            additional={},
        )
        logging.info("Loaded Kaggle dataset from %s", self.data_dir)
        return train_df, test_df, meta

    def _download_from_kaggle(self) -> None:
        """Use the Kaggle API to download the Titanic dataset."""

        logging.info("Attempting to download Titanic dataset from Kaggle")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:  # pragma: no cover - handled by dependency management
            raise RuntimeError("Kaggle package is required to download datasets") from exc

        api = KaggleApi()
        api.authenticate()

        archive_path = self.data_dir / "titanic.zip"
        if archive_path.exists():
            archive_path.unlink()
        api.competition_download_files("titanic", path=str(self.data_dir), quiet=True)

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(self.data_dir)
        archive_path.unlink(missing_ok=True)
        logging.info("Titanic dataset downloaded to %s", self.data_dir)

    def _load_sample(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
        """Load the bundled sample dataset for offline usage."""

        train_path = SAMPLE_DATA_DIR / "train.csv"
        test_path = SAMPLE_DATA_DIR / "test.csv"
        schema_path = SAMPLE_DATA_DIR / "schema.json"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Sample dataset is missing from the repository")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        additional: Dict[str, str] = {}
        if schema_path.exists():
            additional["schema"] = json.dumps(json.loads(schema_path.read_text(encoding="utf-8")))
        meta = DataMeta(
            source="sample",
            location=str(SAMPLE_DATA_DIR),
            additional=additional,
        )
        logging.info("Loaded bundled sample dataset from %s", SAMPLE_DATA_DIR)
        return train_df, test_df, meta


__all__ = ["DataManager", "DataMeta"]
