"""Control-function style two-stage models (paper-inspired)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


CONTROL_COLS = [
    "human_sentiment",
    "ai_sentiment",
    "ai_detect_ratio",
    "vip_level",
    "is_fan",
    "previous_purchase",
    "agent_experience",
]


def first_stage_blame(df: pd.DataFrame, blame_col: str = "blame_detected") -> tuple:
    formula = (
        f"{blame_col} ~ peer_blame_delta + human_sentiment + ai_sentiment + "
        "ai_detect_ratio + vip_level + is_fan + previous_purchase + agent_experience"
    )
    model = smf.logit(formula, data=df).fit(disp=0)
    residual = model.resid_pearson
    return model, residual


def second_stage_outcome(
    df: pd.DataFrame,
    outcome: str,
    residual: pd.Series,
    blame_col: str = "blame_detected",
    model_type: str = "ols",
):
    data = df.copy()
    data["control_residual"] = residual
    formula = (
        f"{outcome} ~ {blame_col} + control_residual + human_sentiment + "
        "ai_sentiment + ai_detect_ratio + vip_level + is_fan + "
        "previous_purchase + agent_experience"
    )
    if model_type == "logit":
        try:
            return smf.logit(formula, data=data).fit(disp=0, maxiter=200)
        except Exception:
            return smf.ols(formula, data=data).fit(cov_type="HC1")
    return smf.ols(formula, data=data).fit(cov_type="HC1")


def summarize_effects(models: dict, blame_col: str = "blame_detected") -> pd.DataFrame:
    rows = []
    for name, fit in models.items():
        coef = fit.params[blame_col]
        se = fit.bse[blame_col]
        pval = fit.pvalues[blame_col]
        pct = np.exp(coef) - 1 if name in {"engagement", "purchase"} else coef
        rows.append(
            {
                "outcome": name,
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "effect_note": (
                    f"{pct:.2%} change" if name in {"engagement", "purchase"} else f"+{coef:.3f} sentiment"
                ),
            }
        )
    return pd.DataFrame(rows)
