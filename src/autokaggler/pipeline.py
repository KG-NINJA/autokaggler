"""Machine learning pipeline implementations for the Kaggle Titanic challenge."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data_manager import DataMeta

TARGET_COLUMN = "Survived"

RAW_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "PassengerId",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Name",
    "Cabin",
)

NUMERIC_FEATURES: Tuple[str, ...] = (
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "CabinKnown",
    "Age*Pclass",
    "FarePerPerson",
)

CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
)

TITLE_NORMALISATION: Dict[str, str] = {
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}

FeatureImportanceEntry = Dict[str, Union[str, float]]


class FeatureEngineer:
    """Composable feature engineering pipeline for Titanic data."""

    __slots__ = ()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        original_columns = set(df.columns)
        data = self._add_title(data)
        data = self._add_family_size(data)
        data = self._add_cabin_flag(data)
        data = self._add_interactions(data)
        new_columns = sorted(set(data.columns) - original_columns)
        if new_columns:
            logging.debug("FeatureEngineer created columns: %s", ", ".join(new_columns))
        return data

    def _add_title(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Title" in data.columns:
            return data
        names = data.get("Name", pd.Series(index=data.index, dtype="object"))
        titles = names.fillna("Unknown").apply(self._extract_title).apply(self._normalise_title)
        data["Title"] = titles
        return data

    @staticmethod
    def _extract_title(name: str) -> str:
        if not isinstance(name, str):
            return "Unknown"
        parts = name.split(",", maxsplit=1)
        if len(parts) != 2:
            return "Unknown"
        title_section = parts[1]
        title = title_section.split(".", maxsplit=1)[0].strip()
        return title or "Unknown"

    @staticmethod
    def _normalise_title(title: str) -> str:
        mapped = TITLE_NORMALISATION.get(title, title)
        if mapped not in COMMON_TITLES:
            return "Rare"
        return mapped

    @staticmethod
    def _add_family_size(data: pd.DataFrame) -> pd.DataFrame:
        sibsp = pd.to_numeric(data.get("SibSp", 0), errors="coerce").fillna(0)
        parch = pd.to_numeric(data.get("Parch", 0), errors="coerce").fillna(0)
        data["FamilySize"] = sibsp + parch
        return data

    @staticmethod
    def _add_cabin_flag(data: pd.DataFrame) -> pd.DataFrame:
        data["CabinKnown"] = data.get("Cabin").notna().astype(int)
        return data

    @staticmethod
    def _add_interactions(data: pd.DataFrame) -> pd.DataFrame:
        age = pd.to_numeric(data.get("Age"), errors="coerce")
        pclass = pd.to_numeric(data.get("Pclass"), errors="coerce")
        data["Age*Pclass"] = (age * pclass).replace([np.inf, -np.inf], np.nan)

        family_size = pd.to_numeric(data.get("FamilySize"), errors="coerce").fillna(0)
        fare = pd.to_numeric(data.get("Fare", 0.0), errors="coerce").fillna(0.0)
        denominator = (family_size + 1).replace(0, 1)
        data["FarePerPerson"] = (fare / denominator).replace([np.inf, -np.inf], np.nan)
        return data


@dataclass
class TitanicPipelineResult:
    """Summary of a Titanic pipeline execution."""

    cv_mean: float
    cv_std: float
    model_name: str
    submission_path: str
    data_source: str
    notes: Optional[str]
    feature_importances: List[FeatureImportanceEntry]


class TitanicPipeline:
    """Configurable Titanic modelling pipeline."""

    def __init__(
        self,
        profile: str = "fast",
        random_seed: int = 42,
        enable_ensemble: Optional[bool] = None,
        profile_registry: Optional[Dict[str, Callable[[], Tuple[Pipeline, str]]]] = None,
    ) -> None:
        self.profile = profile
        self.random_seed = random_seed
        self.enable_ensemble = enable_ensemble
        self._model_name = ""
        self.feature_engineer = FeatureEngineer()
        self.profile_builders: Dict[str, Callable[[], Tuple[Pipeline, str]]] = {}
        self.profile_builders.update(self._default_profiles())
        if profile_registry:
            self.profile_builders.update(profile_registry)

    def _default_profiles(self) -> Dict[str, Callable[[], Tuple[Pipeline, str]]]:
        return {
            "fast": self._build_logistic_pipeline,
            "power": self._build_random_forest_pipeline,
            "boosting": self._build_boosting_pipeline,
        }

    def _create_one_hot_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._create_one_hot_encoder()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, list(NUMERIC_FEATURES)),
                ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
            ],
            sparse_threshold=0.0,
        )

    def _feature_steps(self) -> List[Tuple[str, object]]:
        return [
            ("features", FunctionTransformer(self.feature_engineer.transform, validate=False)),
            ("preprocess", self._build_preprocessor()),
        ]

    def _build_logistic_pipeline(self) -> Tuple[Pipeline, str]:
        model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        steps = self._feature_steps() + [("model", model)]
        return Pipeline(steps=steps), "LogisticRegression"

    def _build_random_forest_pipeline(self) -> Tuple[Pipeline, str]:
        model = RandomForestClassifier(
            n_estimators=600,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        steps = self._feature_steps() + [("model", model)]
        return Pipeline(steps=steps), "RandomForestClassifier"

    def _build_boosting_estimator(self) -> Tuple[object, str]:
        try:
            from lightgbm import LGBMClassifier
            booster = LGBMClassifier(
                objective="binary",
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1,
            )
            return booster, "LightGBM"
        except ImportError:
            try:
                from xgboost import XGBClassifier
                booster = XGBClassifier(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    eval_metric="logloss",
                    random_state=self.random_seed,
                    n_jobs=-1,
                    use_label_encoder=False,
                )
                return booster, "XGBoost"
            except ImportError:
                logging.warning("Neither LightGBM nor XGBoost found; fallback to GradientBoosting")
                return GradientBoostingClassifier(random_state=self.random_seed), "GradientBoosting"

    def _build_boosting_pipeline(self) -> Tuple[Pipeline, str]:
        booster, booster_name = self._build_boosting_estimator()
        use_ensemble = self.enable_ensemble if self.enable_ensemble is not None else True

        feature_step = ("features", FunctionTransformer(self.feature_engineer.transform, validate=False))
        preprocess_step = ("preprocess", self._build_preprocessor())

        if use_ensemble:
            ensemble = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, random_state=self.random_seed)),
                    ("rf", RandomForestClassifier(n_estimators=400, random_state=self.random_seed, n_jobs=-1)),
                    ("boost", booster),
                ],
                voting="soft",
                weights=[1.0, 1.0, 2.0],
            )
            pipeline = Pipeline(steps=[feature_step, preprocess_step, ("model", ensemble)])
            model_name = f"Voting(LogReg+RF+{booster_name})"
        else:
            pipeline = Pipeline(steps=[feature_step, preprocess_step, ("model", booster)])
            model_name = booster_name

        return pipeline, model_name

    def _build_model(self) -> Pipeline:
        builder = self.profile_builders.get(self.profile, self._build_logistic_pipeline)
        pipeline, model_name = builder()
        self._model_name = model_name
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
        random.seed(self.random_seed)

        self._validate_dataframe(train_df, is_train=True)
        self._validate_dataframe(test_df, is_train=False)

        X = train_df.drop(columns=[TARGET_COLUMN])
        y = train_df[TARGET_COLUMN]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        estimator = self._build_model()
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        logging.info("CV accuracy: mean=%.4f std=%.4f", scores.mean(), scores.std())

        estimator.fit(X, y)
        feature_summary = self._log_feature_importance(estimator)

        submission_df = self._build_submission(estimator, test_df)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / submission_name
        submission_df.to_csv(submission_path, index=False)

        compiled_notes = " | ".join(
            filter(None, [notes, "Using bundled sample dataset" if data_meta.source == "sample" else None])
        )

        return TitanicPipelineResult(
            cv_mean=float(scores.mean()),
            cv_std=float(scores.std()),
            model_name=self._model_name,
            submission_path=str(submission_path),
            data_source=data_meta.source,
            notes=compiled_notes,
            feature_importances=feature_summary,
        )

    def _log_feature_importance(self, estimator: Pipeline) -> List[FeatureImportanceEntry]:
        preprocess = estimator.named_steps.get("preprocess")
        if preprocess is None:
            return []
        try:
            feature_names = preprocess.get_feature_names_out()
        except Exception:
            return []

        model = estimator.named_steps.get("model")
        if model is None:
            return []

        summary: List[FeatureImportanceEntry] = []

        def log_single(name: str, values: Iterable[float]) -> None:
            importance = np.asarray(values, dtype=float)
            if importance.ndim > 1:
                importance = np.mean(np.abs(importance), axis=0)
            top_idx = np.argsort(importance)[::-1][:10]
            for idx in top_idx:
                if importance[idx] > 0:
                    summary.append({"model": name, "feature": str(feature_names[idx]), "importance": float(importance[idx])})

        if isinstance(model, VotingClassifier):
            for (name, _), fitted in zip(model.estimators, model.estimators_):
                if hasattr(fitted, "feature_importances_"):
                    log_single(name, fitted.feature_importances_)
                elif hasattr(fitted, "coef_"):
                    log_single(name, np.abs(fitted.coef_))
        else:
            if hasattr(model, "feature_importances_"):
                log_single(self._model_name, model.feature_importances_)
            elif hasattr(model, "coef_"):
                log_single(self._model_name, np.abs(model.coef_))

        return summary

    def _build_submission(self, model: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
        probs = model.predict_proba(test_df)[:, 1]
        survived = (probs >= 0.5).astype(int)
        submission = pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(int), "Survived": survived})
        self._validate_submission(submission, expected_rows=len(test_df))
        return submission

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, *, is_train: bool) -> None:
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
        if is_train and TARGET_COLUMN not in df.columns:
            missing.append(TARGET_COLUMN)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

    @staticmethod
    def _validate_submission(submission: pd.DataFrame, *, expected_rows: int) -> None:
        if list(submission.columns) != ["PassengerId", "Survived"]:
            raise ValueError("Submission must have exactly ['PassengerId','Survived']")
        if len(submission) != expected_rows:
            raise ValueError(f"Submission row count {len(submission)} != expected {expected_rows}")
        if not set(submission["Survived"]).issubset({0, 1}):
            raise ValueError("Survived column must only contain 0/1")


__all__ = ["TitanicPipeline", "TitanicPipelineResult"]
