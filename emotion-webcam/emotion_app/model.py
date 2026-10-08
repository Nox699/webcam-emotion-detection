"""Train scikit-learn logistic regression; deploy validated JSON parameters."""
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BASIC_FEATURES, EXTRACTOR_VERSION, LABELS, PERSONAL_FEATURES
from .data import split_dataset
from .storage import atomic_json


@dataclass
class EmotionModel:
    payload: dict

    def __post_init__(self):
        try:
            p = self.payload
            if p["format_version"] != 1 or p["extractor_version"] != EXTRACTOR_VERSION:
                raise ValueError("Unsupported model or feature-extractor version.")
            if p["source"] not in ("simulated", "personal"):
                raise ValueError("Unknown model source.")
            self.names = BASIC_FEATURES if p["source"] == "simulated" else PERSONAL_FEATURES
            if tuple(p["feature_names"]) != self.names:
                raise ValueError("Incompatible model features.")
            self.classes = tuple(p["classes"])
            if self.classes != LABELS:
                raise ValueError("Incompatible class order.")
            self.mean = np.asarray(p["mean"], dtype=float)
            self.scale = np.asarray(p["scale"], dtype=float)
            self.coef = np.asarray(p["coef"], dtype=float)
            self.intercept = np.asarray(p["intercept"], dtype=float)
            size = len(self.names)
            if (self.mean.shape != (size,) or self.scale.shape != (size,)
                    or self.coef.shape != (3, size) or self.intercept.shape != (3,)):
                raise ValueError("Invalid model parameter dimensions.")
            if not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.coef, self.intercept)):
                raise ValueError("Non-finite model parameters.")
            if (self.scale <= 0).any():
                raise ValueError("Model feature scales must be positive.")
        except (KeyError, TypeError) as error:
            raise ValueError("Corrupt model. Retrain it from the launcher.") from error

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Model is missing: {path.name}. Train it first.")
        if path.stat().st_size > 8_000_000:
            raise ValueError("Model file exceeds the 8 MB limit.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Model is not valid JSON. Retrain it.") from error
        if not isinstance(payload, dict):
            raise ValueError("Model must be a JSON object.")
        return cls(payload)

    def probabilities(self, features) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.names) or not np.isfinite(values).all():
            raise ValueError("Invalid prediction features.")
        scores = ((values - self.mean) / self.scale) @ self.coef.T + self.intercept
        scores -= scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)


def train(frame: pd.DataFrame, personal: bool = False, seed: int = 42):
    names = PERSONAL_FEATURES if personal else BASIC_FEATURES
    train_indices, test_indices = split_dataset(frame, personal, seed)
    x = frame.loc[:, list(names)].to_numpy(dtype=float)
    y = frame.label.to_numpy()
    # All preprocessing is learned from training data alone.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.2 if personal else 1.0, max_iter=2000, random_state=seed,
        class_weight="balanced", solver="lbfgs"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x[train_indices], y[train_indices])
    predictions = pipeline.predict(x[test_indices])
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    metrics = {
        "source": "personal" if personal else "simulated",
        "seed": seed,
        "samples": len(frame),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "accuracy": float(accuracy_score(y[test_indices], predictions)),
        "classification_report": classification_report(y[test_indices], predictions,
            labels=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y[test_indices], predictions, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
        "split": "recording-session holdout (20% of sessions)" if personal else "stratified 80/20",
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_sessions": sorted(frame.iloc[train_indices].session.unique().tolist()) if personal else [],
        "test_sessions": sorted(frame.iloc[test_indices].session.unique().tolist()) if personal else [],
    }
    payload = {
        "format_version": 1, "extractor_version": EXTRACTOR_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": metrics["source"], "algorithm": "StandardScaler + LogisticRegression",
        "feature_names": list(names), "classes": classifier.classes_.tolist(),
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
        "evaluation": {k: v for k, v in metrics.items() if not k.endswith("indices")},
    }
    # Deploy the same model that was evaluated, not a different full-data refit.
    return EmotionModel(payload), metrics, pipeline


def save_model(model: EmotionModel, path: Path) -> None:
    atomic_json(path, model.payload)
