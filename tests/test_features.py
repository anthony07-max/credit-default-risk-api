import json
from pathlib import Path

import pandas as pd

from src.clean import clean
from src.features import ENGINEERED, add_features

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "models" / "example_request.json").read_text())


def test_engineered_columns_present_and_finite():
    out = add_features(pd.DataFrame([EXAMPLE]))
    for c in ENGINEERED:
        assert c in out and pd.notna(out[c].iloc[0])
    assert out["MONTHS_DELINQUENT"].iloc[0] == 0


def test_delinquency_count():
    row = {**EXAMPLE, "PAY_1": 2, "PAY_2": 1}
    out = add_features(pd.DataFrame([row]))
    assert out["MONTHS_DELINQUENT"].iloc[0] == 2
    assert out["MAX_DELAY"].iloc[0] == 2


def test_clean_recodes_undocumented_categories():
    raw = pd.DataFrame([{**EXAMPLE, "ID": 1, "SEX": 2, "EDUCATION": 0, "MARRIAGE": 0,
                         "default payment next month": 1}]).rename(columns={"PAY_1": "PAY_0"})
    out = clean(raw)
    assert out.loc[0, "EDUCATION"] == 4 and out.loc[0, "MARRIAGE"] == 3
    assert "DEFAULT" in out and "PAY_1" in out and "ID" not in out
