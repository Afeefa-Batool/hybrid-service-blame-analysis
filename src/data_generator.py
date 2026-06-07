"""Synthetic hybrid-service chat session data aligned with Chen et al. (2025)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BLAME_PHRASES = [
    "the system made a mistake",
    "our ai misunderstood",
    "this issue was caused by the bot",
    "the chatbot gave wrong info",
    "the bot failed to update",
]

FAILURE_PHRASES = [
    "sorry, that price was incorrect",
    "apologies for the delayed response",
    "the information i gave earlier was wrong",
    "please wait, system is slow today",
]


def _make_transcript(blame: int, failure: int, rng: np.random.Generator) -> str:
    customer = rng.choice(
        [
            "why is the price different from the page?",
            "i need warranty details before buying",
            "can you confirm delivery date?",
            "the link you sent does not work",
        ]
    )
    agent = rng.choice(FAILURE_PHRASES if failure else ["happy to help with your order."])
    tail = ""
    if blame:
        tail = " " + rng.choice(BLAME_PHRASES) + "."
    return f"customer: {customer} | agent: {agent}{tail}"


def generate_sessions(n_sessions: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    agents = rng.integers(1, 121, size=n_sessions)
    days = rng.integers(0, 92, size=n_sessions)
    day = pd.Timestamp("2024-05-01") + pd.to_timedelta(days, unit="D")

    peer_blame_rate = np.clip(rng.normal(0.04, 0.012, size=92), 0.01, 0.12)
    peer_delta = np.diff(peer_blame_rate, prepend=peer_blame_rate[0]) * 100

    blame_prob = 1 / (
        1
        + np.exp(
            -(
                1.8 * peer_delta[days]
                + 0.35 * peer_blame_rate[days]
                + rng.normal(0, 0.35, n_sessions)
                - 2.0
            )
        )
    )
    blame = rng.binomial(1, np.clip(blame_prob, 0.03, 0.22))

    failure = np.ones(n_sessions, dtype=int)
    vip = rng.integers(0, 5, size=n_sessions)
    prev_purchase = rng.poisson(1.5, size=n_sessions)
    is_fan = rng.binomial(1, 0.3, size=n_sessions)
    experience = rng.integers(0, 97, size=n_sessions)

    emotion = (
        -0.15
        + 0.18 * blame
        + 0.02 * vip
        + rng.normal(0, 0.45, n_sessions)
    )
    engagement = (
        1.55
        - 0.08 * blame
        + 0.03 * np.log1p(prev_purchase)
        + rng.normal(0, 0.35, n_sessions)
    )
    purchase_logit = (
        -2.6
        - 0.35 * blame
        + 0.12 * np.log1p(prev_purchase)
        + 0.08 * is_fan
        + rng.normal(0, 0.4, n_sessions)
    )
    purchase = (purchase_logit + rng.normal(0, 0.2, n_sessions) > 0).astype(int)

    df = pd.DataFrame(
        {
            "session_id": np.arange(n_sessions),
            "agent_id": agents,
            "day": day,
            "day_idx": days,
            "transcript": [
                _make_transcript(int(b), 1, rng) for b in blame
            ],
            "service_failure": failure,
            "blame_true": blame,
            "peer_blame_delta": peer_delta[days],
            "customer_emotion": emotion,
            "log_engagement": engagement,
            "purchase": purchase,
            "vip_level": vip,
            "is_fan": is_fan,
            "previous_purchase": prev_purchase,
            "agent_experience": experience,
            "human_sentiment": rng.uniform(0.4, 0.9, n_sessions),
            "ai_sentiment": rng.uniform(0.3, 0.8, n_sessions),
            "ai_detect_ratio": rng.uniform(0, 1, n_sessions),
        }
    )
    return df
