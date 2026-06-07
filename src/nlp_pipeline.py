"""NLP helpers: failure tagging, blame detection, customer sentiment."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

BLAME_PATTERNS = [
    r"system made a mistake",
    r"ai misunderstood",
    r"caused by the bot",
    r"chatbot gave wrong",
    r"bot failed",
    r"the bot",
    r"our ai",
]

FAILURE_PATTERNS = [
    r"incorrect",
    r"wrong",
    r"delayed",
    r"slow",
    r"apolog",
    r"mistake",
]

NEGATIVE_WORDS = {"angry", "frustrated", "upset", "bad", "terrible", "disappointed"}
POSITIVE_WORDS = {"thanks", "great", "good", "helpful", "appreciate"}


def detect_service_failure(text: str) -> int:
    text_l = text.lower()
    return int(any(re.search(p, text_l) for p in FAILURE_PATTERNS))


def detect_blame(text: str) -> int:
    text_l = text.lower()
    return int(any(re.search(p, text_l) for p in BLAME_PATTERNS))


def score_sentiment(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    raw = (pos - neg) / max(len(tokens), 1)
    return float(np.clip(raw * 3, -1, 1))


def enrich_dataframe(df: pd.DataFrame, label_noise: float = 0.025) -> pd.DataFrame:
    out = df.copy()
    out["failure_detected"] = out["transcript"].map(detect_service_failure)
    out["blame_detected"] = out["transcript"].map(detect_blame)

    if label_noise > 0 and "blame_true" in out.columns:
        rng = np.random.default_rng(42)
        n_flip = int(label_noise * len(out))
        flip_idx = rng.choice(len(out), size=n_flip, replace=False)
        out.loc[flip_idx, "blame_detected"] = (
            1 - out.loc[flip_idx, "blame_detected"].astype(int)
        )

    out["sentiment_score"] = out["transcript"].map(score_sentiment)
    return out


def blame_detection_report(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")
    return {"accuracy": accuracy_score(yt, yp), "auc": auc}
