"""Train and evaluate a logistic-regression scorecard and a gradient-boosting model.

Outputs: models/model.joblib, models/metadata.json, reports/metrics.json, charts/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sklearn
from src.features import CATEGORICAL, FEATURES, NUMERIC, RAW_FEATURES, add_features, to_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "credit_clean.csv"
CHARTS, MODELS, REPORTS = ROOT / "charts", ROOT / "models", ROOT / "reports"
SEED = 42
# Cost assumption: missing a defaulter (false negative) costs 5x a wrongly declined good customer.
COST_FN, COST_FP = 5.0, 1.0


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def logistic() -> Pipeline:
    return Pipeline([("prep", preprocessor(True)),
                     ("clf", LogisticRegression(max_iter=2000, C=0.5))])


def boosting() -> Pipeline:
    gbm = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15,
                                         random_state=SEED)
    return Pipeline([("prep", preprocessor(False)), ("clf", gbm)])


def expected_cost(y, p, t):
    pred = p >= t
    fn = ((~pred) & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    return (COST_FN * fn + COST_FP * fp) / len(y)


def main() -> None:
    for d in (CHARTS, MODELS, REPORTS):
        d.mkdir(exist_ok=True)
    df = add_features(pd.read_csv(DATA))
    X, y = df[FEATURES], df["DEFAULT"].values
    X_tr, X_te, y_tr, y_te, sex_tr, sex_te, age_tr, age_te = train_test_split(
        X, y, df["SEX"], df["AGE"], test_size=0.2, stratify=y, random_state=SEED)
    print(f"train {len(X_tr):,} / test {len(X_te):,}; default rate {y.mean():.2%}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {
        "Logistic scorecard": logistic(),
        "Gradient boosting (calibrated)": CalibratedClassifierCV(boosting(), method="isotonic", cv=cv),
    }
    results, test_probs, oof_probs = {}, {}, {}
    for name, m in models.items():
        oof_probs[name] = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        test_probs[name] = p
        results[name] = {
            "roc_auc": roc_auc_score(y_te, p),
            "pr_auc": average_precision_score(y_te, p),
            "brier": brier_score_loss(y_te, p),
            "cv_roc_auc": roc_auc_score(y_tr, oof_probs[name]),
        }
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    # Uncalibrated boosting for the calibration comparison
    raw_gbm = boosting().fit(X_tr, y_tr)
    p_raw = raw_gbm.predict_proba(X_te)[:, 1]
    results["Gradient boosting (uncalibrated)"] = {
        "roc_auc": roc_auc_score(y_te, p_raw), "pr_auc": average_precision_score(y_te, p_raw),
        "brier": brier_score_loss(y_te, p_raw)}

    best = "Gradient boosting (calibrated)"
    final = models[best]
    p_best = test_probs[best]

    # Threshold chosen on out-of-fold TRAIN predictions (no test leakage), then evaluated on test
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    oof_cost = [expected_cost(y_tr, oof_probs[best], t) for t in grid]
    t_star = float(grid[int(np.argmin(oof_cost))])
    test_cost = {t: expected_cost(y_te, p_best, t) for t in grid}
    pred = p_best >= t_star
    tp = int((pred & (y_te == 1)).sum()); fp = int((pred & (y_te == 0)).sum())
    fn = int((~pred & (y_te == 1)).sum()); tn = int((~pred & (y_te == 0)).sum())
    baseline_costs = {"approve_everyone": expected_cost(y_te, p_best, 1.01),
                      "threshold_0.5": expected_cost(y_te, p_best, 0.5),
                      "cost_optimal": expected_cost(y_te, p_best, t_star)}
    threshold = {"threshold": t_star, "cost_fn": COST_FN, "cost_fp": COST_FP,
                 "recall": tp / (tp + fn), "precision": tp / (tp + fp),
                 "flag_rate": float(pred.mean()), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                 "cost_per_customer": baseline_costs}
    print("threshold", threshold)

    # Permutation importance on the test set
    pi = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

    # Fairness audit (SEX is not a model input, but outcomes may still differ by group)
    audit = pd.DataFrame({"y": y_te, "p": p_best, "flag": pred,
                          "sex": sex_te.map({1: "Male", 2: "Female"}).values,
                          "age_band": pd.cut(age_te, [0, 29, 39, 49, 120],
                                             labels=["<30", "30-39", "40-49", "50+"]).astype(str).values})
    fair = []
    for col in ["sex", "age_band"]:
        for g, sub in audit.groupby(col):
            pos, neg = sub[sub.y == 1], sub[sub.y == 0]
            fair.append({"attribute": col, "group": g, "n": len(sub),
                         "actual_default_rate": sub.y.mean(), "mean_predicted_pd": sub.p.mean(),
                         "flag_rate": sub.flag.mean(),
                         "true_positive_rate": pos.flag.mean(), "false_positive_rate": neg.flag.mean(),
                         "roc_auc": roc_auc_score(sub.y, sub.p)})
    fair = pd.DataFrame(fair)
    fair.to_csv(REPORTS / "fairness.csv", index=False)
    print(fair.round(3).to_string())

    # ---------- charts ----------
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    colours = {"Logistic scorecard": "#1f77b4", "Gradient boosting (calibrated)": "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 5))
    status = df.assign(s=df["PAY_1"].clip(-1, 4)).groupby("s")["DEFAULT"].agg(["mean", "size"])
    labels = {-1: "Paid duly / no use", 0: "Revolving", 1: "1 month late", 2: "2 months",
              3: "3 months", 4: "4+ months"}
    ax.bar([labels[i] for i in status.index], status["mean"] * 100, color="#1f77b4")
    for i, (m, n) in enumerate(zip(status["mean"], status["size"])):
        ax.text(i, m * 100 + 1, f"{m:.0%}\nn={n:,}", ha="center", fontsize=8)
    ax.set_ylabel("Default rate next month (%)"); ax.set_title("Default rate by most recent repayment status")
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(CHARTS / "01_default_by_repayment_status.png"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})", color=colours[name])
        pr, rc, _ = precision_recall_curve(y_te, p)
        axes[1].plot(rc, pr, label=f"{name} (AP {results[name]['pr_auc']:.3f})", color=colours[name])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    axes[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8, label=f"Base rate {y_te.mean():.1%}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    for a in axes: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "02_roc_pr_curves.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p, c in [("Logistic scorecard", test_probs["Logistic scorecard"], "#1f77b4"),
                       ("Boosting uncalibrated", p_raw, "#ff7f0e"),
                       ("Boosting + isotonic", p_best, "#d62728")]:
        fy, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fy, "o-", label=f"{name} (Brier {brier_score_loss(y_te, p):.4f})", color=c)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title="Calibration (test set, deciles)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(CHARTS / "03_calibration.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, oof_cost, label="Train (out-of-fold)", color="#7f7f7f")
    ax.plot(grid, list(test_cost.values()), label="Test", color="#d62728")
    ax.axvline(t_star, ls="--", color="k"); ax.text(t_star + 0.01, max(oof_cost) * 0.95, f"chosen t = {t_star:.2f}")
    ax.axvline(0.5, ls=":", color="#1f77b4"); ax.text(0.51, max(oof_cost) * 0.85, "default 0.5", color="#1f77b4")
    ax.set(xlabel="Decision threshold (predicted PD)", ylabel="Expected cost per customer",
           title=f"Cost-based threshold (FN cost {COST_FN:.0f}, FP cost {COST_FP:.0f})")
    ax.legend(); fig.tight_layout(); fig.savefig(CHARTS / "04_threshold_cost.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp.head(12)[::-1]
    ax.barh(top.index, top.values, xerr=pd.Series(pi.importances_std, index=FEATURES)[top.index], color="#d62728")
    ax.set(xlabel="Drop in test ROC-AUC when shuffled", title="Permutation importance (gradient boosting)")
    fig.tight_layout(); fig.savefig(CHARTS / "05_permutation_importance.png"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, col in zip(axes, ["sex", "age_band"]):
        sub = fair[fair.attribute == col].set_index("group")
        if col == "age_band": sub = sub.loc[["<30", "30-39", "40-49", "50+"]]
        x = np.arange(len(sub)); w = 0.2
        for i, (m, lab) in enumerate([("actual_default_rate", "Actual default rate"), ("mean_predicted_pd", "Mean predicted PD"),
                                      ("true_positive_rate", "True positive rate"), ("false_positive_rate", "False positive rate")]):
            a.bar(x + (i - 1.5) * w, sub[m] * 100, w, label=lab)
        a.set_xticks(x, sub.index); a.set_title(f"By {col.replace('_', ' ')}")
    axes[0].set_ylabel("%"); axes[0].legend(fontsize=8)
    fig.suptitle(f"Fairness audit at threshold {t_star:.2f} (test set)"); fig.tight_layout()
    fig.savefig(CHARTS / "06_fairness.png"); plt.close()

    # ---------- artefacts ----------
    joblib.dump(final, MODELS / "model.joblib", compress=3)
    meta = {"model": best, "features": FEATURES, "threshold": t_star, "cost_fn": COST_FN, "cost_fp": COST_FP,
            "test_roc_auc": results[best]["roc_auc"], "test_pr_auc": results[best]["pr_auc"],
            "sklearn_version": sklearn.__version__, "train_rows": len(X_tr), "seed": SEED}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    # One real test-set row as an example request for docs and tests
    raw_cols = RAW_FEATURES
    example = df.loc[X_te.index[0], raw_cols].astype(int).to_dict()
    (MODELS / "example_request.json").write_text(json.dumps(example, indent=2))
    lr = models["Logistic scorecard"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
    scores = to_score(test_probs["Logistic scorecard"])
    metrics = {"models": results, "threshold": threshold,
               "permutation_importance_top10": imp.head(10).round(4).to_dict(),
               "logistic_top_coefficients": coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(3).to_dict(),
               "scorecard_points": {"median_defaulters": float(np.median(scores[y_te == 1])),
                                    "median_non_defaulters": float(np.median(scores[y_te == 0]))},
               "test_rows": len(X_te), "default_rate": float(y.mean()),
               "default_rate_by_pay1": status["mean"].round(4).to_dict()}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("saved model, metadata, metrics and charts")


if __name__ == "__main__":
    main()
