"""Machine learning pipeline implementations for the Kaggle Titanic challenge."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 目的変数
TARGET_COLUMN = "Survived"


@dataclass
class TitanicPipelineResult:
    """Summary of a Titanic pipeline execution."""

    cv_mean: float
    cv_std: float
    model_name: str
    submission_path: str
    data_source: str
    notes: Optional[str]


class TitanicPipeline:
    """Configurable Titanic modelling pipeline."""

    def __init__(self, profile: str = "fast", random_seed: int = 42) -> None:
        self.profile = profile
        self.random_seed = random_seed
        self._model_name = "logreg"

        # 特徴量定義
        self.numeric_features = ["Age", "Fare"]
        self.categorical_features = ["Sex", "Embarked"]

        # 前処理パイプライン
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        # モデル
        self.model = LogisticRegression(max_iter=1000, random_state=random_seed)

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        submission_name: str,
        output_dir: Path,
        notes: Optional[str],
        data_meta,
    ) -> TitanicPipelineResult:
        """Train, evaluate and export predictions for the Titanic task."""

        logging.info("Starting Titanic pipeline with profile '%s'", self.profile)
        np.random.seed(self.random_seed)

        # 入力検証
        self._validate_dataframe(train_df, is_train=True)
        self._validate_dataframe(test_df, is_train=False)

        X = train_df[self.numeric_features + self.categorical_features]
        y = train_df[TARGET_COLUMN]

        # 学習パイプライン
        pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", self.model)]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        logging.info("CV scores: %s", scores)

        # fitting
        pipeline.fit(X, y)
        logging.info("Model '%s' fitted on %d samples", self._model_name, len(train_df))

        # submission
        submission_df = self._build_submission(pipeline, test_df)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / submission_name
        submission_df.to_csv(submission_path, index=False)
        logging.info("Submission saved to %s", submission_path)

        note_lines = []
        if notes:
            note_lines.append(notes)
        if data_meta.source == "sample":
            note_lines.append("Using bundled sample dataset")
        compiled_notes = " | ".join(note_lines) if note_lines else None

        return TitanicPipelineResult(
            cv_mean=float(scores.mean()),
            cv_std=float(scores.std()),
            model_name=self._model_name,
            submission_path=str(submission_path),
            data_source=data_meta.source,
            notes=compiled_notes,
        )

    def _build_submission(self, pipeline: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate submission dataframe with PassengerId and predictions."""
        X_test = test_df[self.numeric_features + self.categorical_features]
        preds = pipeline.predict(X_test)
        return pd.DataFrame(
            {"PassengerId": test_df["PassengerId"], "Survived": preds}
        )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, *, is_train: bool) -> None:
        """Ensure required columns exist."""
        required_columns = ["PassengerId", "Age", "Fare", "Sex", "Embarked"]
        if is_train:
            required_columns.append(TARGET_COLUMN)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")


__all__ = ["TitanicPipeline", "TitanicPipelineResult"]
