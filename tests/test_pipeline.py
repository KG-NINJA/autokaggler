"""Tests for the AutoKaggler Titanic pipeline."""

from __future__ import annotations
from pathlib import Path

import pytest
import pandas as pd

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


def test_success_result_contains_required_metadata(tmp_path):
    dummy_result = TitanicPipelineResult(
        cv_mean=0.5,
        cv_std=0.1,
        model_name="LogisticRegression",
        submission_path=str(tmp_path / "submission.csv"),
        data_source="sample",
        notes=None,
    )
    agent_result = build_success_result(
        run_id="test-run",
        log_path=tmp_path / "log.txt",
        result=dummy_result,
        profile="fast",
    )
    assert agent_result.ok is True
    assert "#KGNINJA" in agent_result.meta["tags"]
