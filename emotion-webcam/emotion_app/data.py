"""Pandas datasets, reproducible simulation, and recording-session holdouts."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import BASIC_FEATURES, LABELS, PERSONAL_FEATURES
from .storage import atomic_text


def simulated_dataset(samples_per_class: int = 800, seed: int = 42) -> pd.DataFrame:
    if samples_per_class < 20:
        raise ValueError("Use at least 20 samples per class.")
    rng = np.random.default_rng(seed)
    means = {"Happy": (0.72, 0.82), "Sad": (0.34, 0.15), "Neutral": (0.62, 0.36)}
    frames = []
    for label in LABELS:
        # Overlap is intentional: accuracy should not be trivially perfect.
        features = np.clip(rng.normal(means[label], (0.14, 0.16),
                                      (samples_per_class, 2)), 0, 1)
        frame = pd.DataFrame(features, columns=BASIC_FEATURES)
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def validate_dataset(frame: pd.DataFrame, feature_names: tuple[str, ...],
                     personal: bool = False) -> None:
    needed = set(feature_names) | {"label"} | ({"session"} if personal else set())
    if not needed.issubset(frame.columns):
        raise ValueError("Dataset has missing columns or an incompatible feature schema.")
    if frame.empty or set(frame.label.unique()) != set(LABELS):
        raise ValueError("Dataset must contain Happy, Neutral, and Sad, with no other labels.")
    try:
        features = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Dataset features must be numeric.") from error
    if not np.isfinite(features).all() or (features < 0).any() or (features > 1).any():
        raise ValueError("Dataset features must be finite numbers between 0 and 1.")
    if frame.groupby("label").size().min() < 10:
        raise ValueError("Collect at least 10 samples for each expression.")
    if personal:
        if frame.session.isna().any() or frame.session.astype(str).str.strip().eq("").any():
            raise ValueError("Each personal sample must have a nonempty recording-session ID.")
        if frame.session.nunique() < 5:
            raise ValueError("Collect at least 5 separate complete sessions before training. "
                             "One session contains all 3 expressions.")
        for _, session in frame.groupby("session"):
            if set(session.label) != set(LABELS) or session.groupby("label").size().min() < 10:
                raise ValueError("Each session must contain at least 10 samples of all 3 expressions.")


def split_dataset(frame: pd.DataFrame, personal: bool = False, seed: int = 42):
    features = PERSONAL_FEATURES if personal else BASIC_FEATURES
    validate_dataset(frame, features, personal)
    indices = np.arange(len(frame))
    if personal:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train, test = next(splitter.split(indices, frame.label, groups=frame.session))
    else:
        train, test = train_test_split(indices, test_size=0.2, stratify=frame.label, random_state=seed)
    return train, test


def read_personal(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise ValueError("No personal dataset yet. Record a complete session first.")
    # Reject huge accidental inputs before parsing; the normal five-session set is <8 MB.
    if Path(path).stat().st_size > 250_000_000:
        raise ValueError("Dataset is too large for this laptop application (250 MB limit).")
    return pd.read_csv(path, dtype={"session": str})


def save_session(path: Path, rows: list[dict]) -> None:
    """Commit only a complete recording session; an interrupted one is discarded."""
    incoming = pd.DataFrame(rows)
    if incoming.empty or incoming.session.nunique() != 1:
        raise ValueError("Expected one complete recording session.")
    if set(incoming.label) != set(LABELS) or incoming.groupby("label").size().min() < 10:
        raise ValueError("Incomplete session: at least 10 samples per expression are required.")
    expected = list(PERSONAL_FEATURES) + ["label", "session", "sample", "captured_at"]
    if list(incoming.columns) != expected:
        raise ValueError("Recording schema is incompatible.")
    values = incoming.loc[:, list(PERSONAL_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Recording contains invalid features.")
    if Path(path).exists():
        previous = read_personal(path)
        if list(previous.columns) != expected:
            raise ValueError("Existing dataset uses a different feature schema. Back it up first.")
        if incoming.session.iloc[0] in set(previous.session):
            raise ValueError("This session was already saved.")
        incoming = pd.concat([previous, incoming], ignore_index=True)
    atomic_text(path, incoming.to_csv(index=False, float_format="%.8g"))
