"""Threshold sweep for logistic regression without payment lateness.

The threshold results in the paper (Section 4.5 and the flagged counts in
Section 4.7) come from Model B with payment_lateness_months included. The
sensitivity analysis in main_experiment.py (Section 10) removes that variable
and confirms the AUC gain, but reports metrics at the default threshold only.
This script repeats the threshold sweep of Section 3 of main_experiment.py with
payment_lateness_months removed from both feature sets, at the primary seed, and
compares the review list at the operating threshold with the list produced by
the full Model B.

The data and preprocessing helpers are copied from main_experiment.py unchanged.
Section 0 tests the copy against two stored result files and stops on any
difference.

Run:      python scripts/threshold_sweep_no_lateness.py
Writes:   results/threshold_sweep_no_lateness.csv
          results/review_list_overlap_no_lateness.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"

DATA_FILE = DATA / "dataset.csv"
PRIMARY_SEED = 42
SEEDS = [42, 1, 7, 13, 99]
OPERATING_THRESHOLD = 0.30

raw = pd.read_csv(DATA_FILE)
df = raw.drop(columns=["member_id"])
y = df.pop("default").values

FEATURES_A = [
    "age",
    "gender",
    "marital_status",
    "education",
    "occupation",
    "city",
    "number_of_dependents",
    "monthly_income_idr",
    "loan_amount_idr",
    "loan_term_months",
    "interest_rate_pct_per_year",
    "debt_ratio",
    "prior_loans",
    "payment_lateness_months",
]
FEATURES_MEMBERSHIP = [
    "membership_tenure_months",
    "principal_savings_idr",
    "mandatory_savings_idr",
    "voluntary_savings_idr",
    "membership_category",
    "annual_meeting_participation",
    "payroll_deduction",
    "activity_status",
    "transaction_activity_12m_idr",
]
FEATURES_B = FEATURES_A + FEATURES_MEMBERSHIP
A_NO_LATENESS = [c for c in FEATURES_A if c != "payment_lateness_months"]
B_NO_LATENESS = A_NO_LATENESS + FEATURES_MEMBERSHIP

CATEGORY_ORDER = {
    "gender": ["male", "female"],
    "marital_status": ["single", "divorced", "married"],
    "education": ["diploma", "bachelor's degree", "master's degree", "senior high school"],
    "occupation": [
        "laborer",
        "teacher",
        "private-sector employee",
        "fisher",
        "civil servant",
        "trader",
        "nurse",
        "farmer",
        "micro or small business owner",
        "self-employed",
    ],
    "city": ["Bandung", "Jakarta", "Makassar", "Medan", "Palembang", "Semarang", "Surabaya", "Yogyakarta"],
    "membership_category": ["full member", "prospective member"],
    "annual_meeting_participation": ["no", "yes"],
    "payroll_deduction": ["no", "yes"],
    "activity_status": ["active", "inactive"],
}


def make_logreg(seed):
    return LogisticRegression(max_iter=5000, random_state=seed)


def preprocess(cols):
    cat = [c for c in cols if df[c].dtype.name in ("object", "str")]
    num = [c for c in cols if c not in cat]
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORY_ORDER[c] for c in cat],
                    handle_unknown="ignore",
                    drop="first",
                ),
                cat,
            ),
            ("num", StandardScaler(), num),
        ]
    )


def oof(cols, seed):
    X = df[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline([("pre", preprocess(cols)), ("clf", make_logreg(seed))])
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def metrics(y_true, p, t=0.5):
    pred = (p >= t).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, pred),
        Precision=precision_score(y_true, pred, zero_division=0),
        Recall=recall_score(y_true, pred),
        F1=f1_score(y_true, pred),
        AUC=roc_auc_score(y_true, p),
        Brier=brier_score_loss(y_true, p),
    )


print("=" * 60)
print("0. PARITY GUARD against two stored result files")
print("=" * 60)
stored_sens = pd.read_csv(RESULTS / "sensitivity_payment_lateness.csv")
stored_sweep = pd.read_csv(RESULTS / "threshold_sweep.csv")
ok = True
for tag, cols in [("A", A_NO_LATENESS), ("B", B_NO_LATENESS)]:
    auc = round(np.mean([roc_auc_score(y, oof(cols, s)) for s in SEEDS]), 4)
    ref = stored_sens.query("Model == 'Logistic Regression' and Features == @tag").AUC.iloc[0]
    print(f"  mean AUC over five seeds, {tag} without lateness: recomputed {auc:.4f}, stored {ref:.4f}")
    ok &= auc == ref
pB_full = oof(FEATURES_B, PRIMARY_SEED)
tn, fp, fn, tp = confusion_matrix(y, (pB_full >= OPERATING_THRESHOLD).astype(int)).ravel()
ref = stored_sweep.query("Features == 'B' and Threshold == @OPERATING_THRESHOLD").iloc[0]
print(f"  full Model B at {OPERATING_THRESHOLD:.2f}: recomputed TP={tp} FP={fp}, stored TP={ref.TP} FP={ref.FP}")
ok &= tp == ref.TP and fp == ref.FP
if not ok:
    sys.exit("Parity guard failed: the copied helpers do not reproduce the stored results.")
print("  parity guard passed")

print("\n" + "=" * 60)
print("1. THRESHOLD SWEEP WITHOUT payment_lateness_months (seed 42)")
print("=" * 60)
pA_no = oof(A_NO_LATENESS, PRIMARY_SEED)
pB_no = oof(B_NO_LATENESS, PRIMARY_SEED)
rows = []
for t in np.arange(0.20, 0.65, 0.05):
    for tag, p in [("A", pA_no), ("B", pB_no)]:
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
        rows.append(
            dict(
                Threshold=round(t, 2),
                Features=tag,
                **{k: round(v, 4) for k, v in metrics(y, p, t).items()},
                TP=tp,
                FP=fp,
                FN=fn,
                TN=tn,
                Flagged=tp + fp,
            )
        )
sw = pd.DataFrame(rows)
sw.to_csv(RESULTS / "threshold_sweep_no_lateness.csv", index=False)
print(sw[sw.Features == "B"][["Threshold", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN", "Flagged"]].to_string(index=False))

print("\n" + "=" * 60)
print(f"2. REVIEW LIST AT {OPERATING_THRESHOLD:.2f}: full Model B versus Model B without lateness")
print("=" * 60)
flag_full = pB_full >= OPERATING_THRESHOLD
flag_no = pB_no >= OPERATING_THRESHOLD
both = flag_full & flag_no
rows = [
    dict(Model="Model B, full", Flagged=int(flag_full.sum()), TP=int((flag_full & (y == 1)).sum()), FP=int((flag_full & (y == 0)).sum())),
    dict(Model="Model B, without payment lateness", Flagged=int(flag_no.sum()), TP=int((flag_no & (y == 1)).sum()), FP=int((flag_no & (y == 0)).sum())),
    dict(Model="Flagged by both", Flagged=int(both.sum()), TP=int((both & (y == 1)).sum()), FP=int((both & (y == 0)).sum())),
    dict(Model="Flagged only by the full model", Flagged=int((flag_full & ~flag_no).sum()), TP=int((flag_full & ~flag_no & (y == 1)).sum()), FP=int((flag_full & ~flag_no & (y == 0)).sum())),
    dict(Model="Flagged only without payment lateness", Flagged=int((flag_no & ~flag_full).sum()), TP=int((flag_no & ~flag_full & (y == 1)).sum()), FP=int((flag_no & ~flag_full & (y == 0)).sum())),
]
ov = pd.DataFrame(rows)
ov["Share_of_sample_pct"] = (100 * ov.Flagged / len(y)).round(2)
ov.to_csv(RESULTS / "review_list_overlap_no_lateness.csv", index=False)
print(ov.to_string(index=False))
jacc = both.sum() / (flag_full | flag_no).sum()
print(f"\n  Jaccard overlap of the two review lists: {jacc:.4f}")
