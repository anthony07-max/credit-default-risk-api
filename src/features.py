"""Shared feature definitions, used by training and by the API."""
import numpy as np
import pandas as pd

PAY_STATUS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["EDUCATION", "MARRIAGE"]
# SEX is deliberately NOT a model input: it is kept only for the fairness audit.
RAW_FEATURES = ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS + BILL + PAY_AMT
ENGINEERED = ["UTILISATION", "PAY_RATIO_1", "MONTHS_DELINQUENT", "MAX_DELAY", "AVG_BILL", "AVG_PAYMENT"]
NUMERIC = [c for c in RAW_FEATURES if c not in CATEGORICAL] + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    limit = out["LIMIT_BAL"].clip(lower=1)
    out["UTILISATION"] = (out["BILL_AMT1"] / limit).clip(-1, 5)
    out["PAY_RATIO_1"] = np.where(out["BILL_AMT1"] > 0,
                                  (out["PAY_AMT1"] / out["BILL_AMT1"].clip(lower=1)).clip(0, 5), 1.0)
    out["MONTHS_DELINQUENT"] = (out[PAY_STATUS] > 0).sum(axis=1)
    out["MAX_DELAY"] = out[PAY_STATUS].max(axis=1).clip(lower=0)
    out["AVG_BILL"] = out[BILL].mean(axis=1)
    out["AVG_PAYMENT"] = out[PAY_AMT].mean(axis=1)
    return out

FEATURES = NUMERIC + CATEGORICAL


def to_score(p, base=600, pdo=20, base_odds=50):
    """Classic scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    return offset + factor * np.log((1 - p) / p)
