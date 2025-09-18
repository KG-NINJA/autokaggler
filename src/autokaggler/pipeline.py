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

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def _default_profiles(self) -> Dict[str, Callable[[], Tuple[Pipeline, str]]]:
        return {
            "fast": self._build_logistic_pipeline,
            "power": self._build_random_forest_pipeline,
            "boosting": self._build_boosting_pipeline,
        }

    def register_profile(
        self, name: str, builder: Callable[[], Tuple[Pipeline, str]]
    ) -> None:
        """Register an additional modelling profile."""

        self.profile_builders[name] = builder

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------
    def _create_one_hot_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # pragma: no cover - backwards compatibility
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
        pipeline = Pipeline(steps=steps)
        return pipeline, "LogisticRegression"

    def _build_random_forest_pipeline(self) -> Tuple[Pipeline, str]:
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            random_state=self.random_seed,
            n_jobs=-1,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        steps = self._feature_steps() + [("model", model)]
        pipeline = Pipeline(steps=steps)
        return pipeline, "RandomForestClassifier"

    def _build_boosting_estimator(self) -> Tuple[object, str]:
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            booster = LGBMClassifier(
                objective="binary",
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1,
                reg_lambda=1.0,
                min_child_samples=20,
                bagging_seed=self.random_seed,
                feature_fraction_seed=self.random_seed,
            )
            return booster, "LightGBM"
        except ImportError:
            try:
                from xgboost import XGBClassifier  # type: ignore

                booster = XGBClassifier(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    eval_metric="logloss",
                    random_state=self.random_seed,
                    tree_method="hist",
                    reg_lambda=1.0,
                    gamma=0.0,
                    min_child_weight=1.0,
                    n_jobs=-1,
                    seed=self.random_seed,
                    use_label_encoder=False,
                )
                return booster, "XGBoost"
            except ImportError:
                logging.warning(
                    "Neither LightGBM nor XGBoost is installed; falling back to GradientBoostingClassifier"
                )
                booster = GradientBoostingClassifier(random_state=self.random_seed)
                return booster, "GradientBoostingClassifier"

    def _build_boosting_pipeline(self) -> Tuple[Pipeline, str]:
        booster, booster_name = self._build_boosting_estimator()

        use_ensemble = self.enable_ensemble if self.enable_ensemble is not None else True
        feature_step = ("features", FunctionTransformer(self.feature_engineer.transform, validate=False))
        preprocess_step = ("preprocess", self._build_preprocessor())

        if use_ensemble:
            base_estimators: List[Tuple[str, object]] = [
                (
                    "lr",
                    LogisticRegression(max_iter=1000, random_state=self.random_seed),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        random_state=self.random_seed,
                        n_jobs=-1,
                    ),
                ),
                ("boost", booster),
            ]
            ensemble = VotingClassifier(
                estimators=base_estimators,
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
        builder = self.profile_builders.get(self.profile)
        if builder is None:
            logging.warning("Unknown profile '%s'; defaulting to logistic regression", self.profile)
            builder = self._build_logistic_pipeline
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

        logging.info(
            "Dataset summary: train=%d rows, test=%d rows, target positive rate=%.3f",
            len(train_df),
            len(test_df),
            float(train_df[TARGET_COLUMN].mean()),
        )

        estimator = self._build_model()

        X = train_df.drop(columns=[TARGET_COLUMN])
        y = train_df[TARGET_COLUMN]

        if y.empty:
            raise ValueError("Training data must contain target labels")
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            raise ValueError("Stratified K-Fold requires at least two samples per class")
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        self._log_cv_scores(scores)

        estimator.fit(X, y)
        logging.info("Model '%s' fitted on %d samples", self._model_name, len(train_df))

        feature_summary = self._log_feature_importance(estimator)

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
            feature_importances=feature_summary,
        )

    def _log_cv_scores(self, scores: np.ndarray) -> None:
        for index, fold_score in enumerate(scores, start=1):
            logging.info("Fold %d accuracy: %.4f", index, fold_score)
        logging.info(
            "Cross-validation accuracy summary: mean=%.4f std=%.4f",
            float(scores.mean()),
            float(scores.std()),
        )

    def _log_feature_importance(self, estimator: Pipeline) -> List[FeatureImportanceEntry]:
        preprocess = estimator.named_steps.get("preprocess")
        if preprocess is None:
            return []
        try:
            feature_names = preprocess.get_feature_names_out()
        except AttributeError:  # pragma: no cover - sklearn < 1.0
            logging.info("Skipping feature importance logging; transformer lacks feature names")
            return []

        model = estimator.named_steps.get("model")
        if model is None:
            return []

        summary: List[FeatureImportanceEntry] = []

        def log_single(model_name: str, values: Iterable[float]) -> None:
            importance = np.asarray(list(values), dtype=float)
            if importance.ndim > 1:
                importance = np.mean(np.abs(importance), axis=0)
            top_indices = np.argsort(importance)[::-1][:10]
            pairs = [f"{feature_names[i]}={importance[i]:.4f}" for i in top_indices if importance[i] > 0]
            if pairs:
                logging.info("Top features for %s: %s", model_name, ", ".join(pairs))
            for idx in top_indices:
                if importance[idx] <= 0:
                    continue
                summary.append(
                    {
                        "model": model_name,
                        "feature": str(feature_names[idx]),
                        "importance": float(importance[idx]),
                    }
                )

        if isinstance(model, VotingClassifier):
            for (name, _), fitted in zip(model.estimators, model.estimators_):
                if hasattr(fitted, "feature_importances_"):
                    log_single(f"{name}", fitted.feature_importances_)
                elif hasattr(fitted, "coef_"):
                    log_single(f"{name}", np.abs(fitted.coef_))
        else:
            if hasattr(model, "feature_importances_"):
                log_single(self._model_name, model.feature_importances_)
            elif hasattr(model, "coef_"):
                log_single(self._model_name, np.abs(model.coef_))

        return summary

    def _build_submission(self, model: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
        predictions = model.predict_proba(test_df)[:, 1]
        survived = (predictions >= 0.5).astype(int)
        submission_df = pd.DataFrame({
            "PassengerId": test_df["PassengerId"].astype(int),
            "Survived": survived,
        })
        self._validate_submission(submission_df, expected_rows=len(test_df))
        return submission_df

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, *, is_train: bool) -> None:
        missing_columns = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
        if is_train and TARGET_COLUMN not in df.columns:
            missing_columns.append(TARGET_COLUMN)
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    @staticmethod
    def _validate_submission(submission_df: pd.DataFrame, *, expected_rows: int) -> None:
        expected_columns = ["PassengerId", "Survived"]
        if list(submission_df.columns) != expected_columns:
            raise ValueError(
                f"Submission must have exactly columns {expected_columns}; got {list(submission_df.columns)}"
            )
        if len(submission_df) != expected_rows:
            raise ValueError(
                f"Submission row count {len(submission_df)} does not match expected {expected_rows}"
            )
        unique_values = set(submission_df["Survived"].unique())
        if not unique_values.issubset({0, 1}):
            raise ValueError("Submission contains values other than 0/1 in 'Survived'")


__all__ = ["TitanicPipeline", "TitanicPipelineResult"]
