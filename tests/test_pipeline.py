"""Tests for the AutoKaggler Titanic pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autokaggler.agent import build_success_result
from autokaggler.data_manager import DataManager, SAMPLE_DATA_DIR
from autokaggler.pipeline import TitanicPipeline, TitanicPipelineResult


def test_pipeline_runs_on_sample_data(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="sample")
    pipeline = TitanicPipeline(profile="fast", random_seed=7)
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name="test_submission.csv",
        output_dir=manager.submission_dir,
        notes="pytest",
        data_meta=meta,
    )

    assert 0.0 <= result.cv_mean <= 1.0
    assert Path(result.submission_path).exists()
    assert result.data_source == "sample"
    assert isinstance(result.feature_importances, list)

    submission_df = pd.read_csv(result.submission_path)
    assert list(submission_df.columns) == ["PassengerId", "Survived"]
    assert len(submission_df) == len(test_df)
    assert set(submission_df["Survived"].unique()).issubset({0, 1})


def test_boosting_profile_runs_with_ensemble(tmp_path):
    manager = DataManager(cache_dir=tmp_path)
    train_df, test_df, meta = manager.prepare_datasets(prefer_source="sample")
    pipeline = TitanicPipeline(profile="boosting", random_seed=11, enable_ensemble=True)
    result = pipeline.run(
        train_df=train_df,
        test_df=test_df,
        submission_name="boosting_submission.csv",
        output_dir=manager.submission_dir,
        notes="pytest",
        data_meta=meta,
    )

    assert 0.0 <= result.cv_mean <= 1.0
    assert Path(result.submission_path).exists()
    assert result.feature_importances
    submission_df = pd.read_csv(result.submission_path)
    assert len(submission_df) == len(test_df)


def test_prepare_datasets_uses_cached_when_kaggle_unavailable(tmp_path, monkeypatch):
    manager = DataManager(cache_dir=tmp_path)
    cached_train = pd.read_csv(SAMPLE_DATA_DIR / "train.csv")
    cached_test = pd.read_csv(SAMPLE_DATA_DIR / "test.csv")
    cached_train.to_csv(manager.data_dir / "train.csv", index=False)
    cached_test.to_csv(manager.data_dir / "test.csv", index=False)

    def fail_download(self):
        raise RuntimeError("kaggle down")

    monkeypatch.setattr(DataManager, "_download_from_kaggle", fail_download)

    train_df, test_df, meta = manager.prepare_datasets(prefer_source="auto", force_download=True)

    assert meta.source == "kaggle_cached"
    assert len(train_df) == len(cached_train)
    assert len(test_df) == len(cached_test)


def test_prepare_datasets_falls_back_to_sample_when_no_cache(tmp_path, monkeypatch):
    manager = DataManager(cache_dir=tmp_path)

    def fail_download(self):
        raise RuntimeError("kaggle down")

    monkeypatch.setattr(DataManager, "_download_from_kaggle", fail_download)

    train_df, test_df, meta = manager.prepare_datasets(prefer_source="auto", force_download=True)

    assert meta.source == "sample"
    assert not train_df.empty
    assert not test_df.empty


def test_submission_validation_rejects_invalid_format():
    df = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 2]})
    with pytest.raises(ValueError):
        TitanicPipeline._validate_submission(df, expected_rows=2)


def test_success_result_contains_required_metadata(tmp_path):
    dummy_result = TitanicPipelineResult(
        cv_mean=0.5,
        cv_std=0.1,
        model_name="LogisticRegression",
        submission_path=str(tmp_path / "submission.csv"),
        data_source="sample",
        notes=None,
        feature_importances=[{"model": "lr", "feature": "Age", "importance": 0.1}],
    )
    agent_result = build_success_result(
        run_id="test-run",
        log_path=tmp_path / "log.txt",
        result=dummy_result,
        profile="fast",
    )
    assert agent_result.ok is True
    assert "#KGNINJA" in agent_result.meta["tags"]
