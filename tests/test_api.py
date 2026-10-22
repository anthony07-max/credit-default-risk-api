import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import META, app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "models" / "example_request.json").read_text())
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_example():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "refer"}
    assert body["threshold"] == META["threshold"]


def test_late_payments_raise_risk():
    late = {**EXAMPLE, **{f"PAY_{i}": 3 for i in range(1, 7)}}
    p_ok = client.post("/predict", json=EXAMPLE).json()["probability_of_default"]
    p_late = client.post("/predict", json=late).json()["probability_of_default"]
    assert p_late > p_ok


def test_score_is_monotonic_in_pd():
    late = {**EXAMPLE, **{f"PAY_{i}": 3 for i in range(1, 7)}}
    preds = client.post("/predict/batch", json=[EXAMPLE, late]).json()
    assert len(preds) == 2
    assert preds[1]["score"] < preds[0]["score"]


@pytest.mark.parametrize("field,value", [("AGE", 10), ("EDUCATION", 7), ("LIMIT_BAL", -5), ("PAY_1", 12)])
def test_validation_rejects_bad_input(field, value):
    r = client.post("/predict", json={**EXAMPLE, field: value})
    assert r.status_code == 422


def test_rejects_unknown_fields_such_as_sex():
    r = client.post("/predict", json={**EXAMPLE, "SEX": 1})
    assert r.status_code == 422


def test_missing_field():
    bad = dict(EXAMPLE); bad.pop("LIMIT_BAL")
    assert client.post("/predict", json=bad).status_code == 422
