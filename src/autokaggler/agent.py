"""Core agent implementation orchestrating the AutoKaggler pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .data_manager import DataManager
from .pipeline import TitanicPipeline, TitanicPipelineResult

RUNTIME_DIRS = [Path(".agent_tmp"), Path(".agent_logs")]
DEFAULT_PROFILE = "fast"
TAG = "#KGNINJA"


@dataclass
class TaskInput:
    """Declarative configuration for a pipeline execution."""

    profile: Optional[str] = None
    force_download: bool = False
    data_source: str = "auto"
    random_seed: int = 42
    submission_name: Optional[str] = None
    notes: Optional[str] = None

=======

    if profile not in ("fast", "boosting", "tree", "linear"):
        logging.warning(
            "Unknown profile '%s'; falling back to default '%s'",
            profile,
            DEFAULT_PROFILE,
        )
        profile = DEFAULT_PROFILE


    os.environ["PROFILE"] = profile
    return profile


def run_agent(task_input: TaskInput, run_id: str) -> TitanicPipelineResult:
    """Execute the Titanic pipeline."""

    data_manager = DataManager(cache_dir=RUNTIME_DIRS[0])
    train_df, test_df, data_meta = data_manager.prepare_datasets(
        prefer_source=task_input.data_source,
        force_download=task_input.force_download,
    )

    profile = resolve_profile(task_input)

if __name__ == "__main__":  # pragma: no cover

    main()
