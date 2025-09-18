"""Machine learning pipeline implementations for the Kaggle Titanic challenge."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_manager import DataMeta

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
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
        self._model_name = self._select_model_name(profile)

    @staticmethod
    def _select_model_name(profile: str) -> str:
        if profile == "power":
            return "RandomForestClassifier"
        return "LogisticRegression"

    def _build_model(self) -> Pipeline:
        numeric_features = ["Age", "SibSp", "Parch", "Fare"]
        categorical_features = ["Pclass", "Sex", "Embarked"]

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        if self.profile == "power":
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                random_state=self.random_seed,
                n_jobs=-1,
                min_samples_split=2,
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=self.random_seed)

        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        return pipeline

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        submission_name: str,
        output_dir: Path,
        notes: Optional[str],
        data_meta: DataMeta,
    ) -> TitanicPipelineResult:
        """Train, evaluate and export predictions for the Titanic task."""

        logging.info("Starting Titanic pipeline with profile '%s'", self.profile)
        np.random.seed(self.random_seed)

        self._validate_dataframe(train_df, is_train=True)
        self._validate_dataframe(test_df, is_train=False)

        X = train_df[FEATURE_COLUMNS]
        y = train_df[TARGET_COLUMN]

        estimator = self._build_model()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        logging.info("Cross-validation accuracy: mean=%.4f std=%.4f", scores.mean(), scores.std())

        estimator.fit(X, y)
        logging.info("Model '%s' fitted on %d samples", self._model_name, len(train_df))

        submission_df = self._build_submission(estimator, test_df)

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

    @staticmethod
    def _build_submission(model: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
        predictions = model.predict(test_df[FEATURE_COLUMNS])
        submission_df = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions.astype(int),
        })
        return submission_df

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, *, is_train: bool) -> None:
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if is_train and TARGET_COLUMN not in df.columns:
            missing_columns.append(TARGET_COLUMN)
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")


__all__ = ["TitanicPipeline", "TitanicPipelineResult"]
