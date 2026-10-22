"""FastAPI service that scores credit card customers for next-month default risk."""
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.features import FEATURES, RAW_FEATURES, add_features, to_score

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "model.joblib")
META = json.loads((ROOT / "models" / "metadata.json").read_text())
EXAMPLE = json.loads((ROOT / "models" / "example_request.json").read_text())

Status = Field(ge=-2, le=9, description="Repayment status: -2/-1/0 = no delay, 1..9 = months late")
Amount = Field(description="Amount in NT dollars")


class Customer(BaseModel):
    """One customer's credit line, demographics (no sex) and 6 months of statement history."""
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE}, extra="forbid")

    LIMIT_BAL: float = Field(gt=0, description="Credit limit (NT dollars)")
    EDUCATION: int = Field(ge=1, le=4, description="1 graduate school, 2 university, 3 high school, 4 other")
    MARRIAGE: int = Field(ge=1, le=3, description="1 married, 2 single, 3 other")
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Status
    PAY_2: int = Status
    PAY_3: int = Status
    PAY_4: int = Status
    PAY_5: int = Status
    PAY_6: int = Status
    BILL_AMT1: float = Amount
    BILL_AMT2: float = Amount
    BILL_AMT3: float = Amount
    BILL_AMT4: float = Amount
    BILL_AMT5: float = Amount
    BILL_AMT6: float = Amount
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class Prediction(BaseModel):
    probability_of_default: float
    score: float = Field(description="Scorecard points (600 at 50:1 odds, 20 points to double the odds)")
    decision: str = Field(description="'refer' if PD >= cost-optimal threshold, else 'approve'")
    threshold: float


app = FastAPI(title="Credit default risk API", version="1.0.0",
              description="Calibrated gradient-boosting model trained on the UCI Default of Credit Card Clients dataset.")


def score(customers: List[Customer]) -> List[Prediction]:
    df = add_features(pd.DataFrame([c.model_dump() for c in customers])[RAW_FEATURES])
    pd_ = MODEL.predict_proba(df[FEATURES])[:, 1]
    t = META["threshold"]
    return [Prediction(probability_of_default=round(float(p), 4), score=round(float(to_score(p)), 1),
                       decision="refer" if p >= t else "approve", threshold=t) for p in pd_]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": META["model"], "test_roc_auc": round(META["test_roc_auc"], 4)}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return score([customer])[0]


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer]) -> List[Prediction]:
    return score(customers) if customers else []
