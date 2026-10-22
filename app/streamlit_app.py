"""Streamlit front end: model performance, fairness audit and an interactive scorer."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.main import Customer, META, EXAMPLE, score  # noqa: E402

st.set_page_config(page_title="Credit default risk", layout="wide")
st.title("Credit card default risk")
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())

c1, c2, c3, c4 = st.columns(4)
m = metrics["models"][META["model"]]
c1.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
c2.metric("Test PR-AUC", f"{m['pr_auc']:.3f}")
c3.metric("Cost-optimal threshold", f"{META['threshold']:.2f}")
c4.metric("Defaulters caught", f"{metrics['threshold']['recall']:.0%}")

tab1, tab2, tab3 = st.tabs(["Score a customer", "Model performance", "Fairness"])
with tab1:
    cols = st.columns(3)
    vals = {}
    for i, (k, v) in enumerate(EXAMPLE.items()):
        vals[k] = cols[i % 3].number_input(k, value=int(v), step=1)
    try:
        pred = score([Customer(**vals)])[0]
        st.subheader(f"PD {pred.probability_of_default:.1%} | score {pred.score:.0f} | decision: {pred.decision.upper()}")
    except Exception as e:  # pydantic validation errors
        st.error(str(e))
with tab2:
    st.dataframe(pd.DataFrame(metrics["models"]).T.round(4))
    for c in ["02_roc_pr_curves", "03_calibration", "04_threshold_cost", "05_permutation_importance"]:
        st.image(str(ROOT / "charts" / f"{c}.png"))
with tab3:
    st.dataframe(pd.read_csv(ROOT / "reports" / "fairness.csv").round(3))
    st.image(str(ROOT / "charts" / "06_fairness.png"))
