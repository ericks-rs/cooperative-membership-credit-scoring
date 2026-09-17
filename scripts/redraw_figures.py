"""Redraws the three source figures at high resolution.

This script exists so the figure resolution can be raised without rerunning the
whole experiment, which takes far longer. The plotting code is copied unchanged
from main_experiment.py and inference_and_checks.py, so the figures have the
same shape and only the pixel density changes.

The out-of-fold probabilities are recomputed with the same protocol, logistic
regression at seed 42, so the curves are exactly those the main script draws.

Run:      python scripts/redraw_figures.py
Writes:   figures/roc_curves.png, figures/calibration_curves.png,
          figures/cost_surface.png
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

matplotlib.use("Agg")

DPI = 600
# Fig. 2 is printed 12.6 cm wide, so each 5.5 in panel is scaled to about 0.45.
# These sizes give roughly 8 pt labels, 7 pt tick labels, and 7 pt legend text on the page.
LABEL, TICK, LEGEND = 18, 16, 15
PRIMARY_SEED = 42
OPERATING_THRESHOLD = 0.30
TOLERANCE = 0.01

raw = pd.read_csv(DATA / "dataset.csv")
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

# Same pinned category order as main_experiment.py (see that file for why).
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
        pipe = Pipeline(
            [
                ("pre", preprocess(cols)),
                ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
            ]
        )
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


pA = oof(FEATURES_A, PRIMARY_SEED)
pB = oof(FEATURES_B, PRIMARY_SEED)
print(f"AUC Model A {roc_auc_score(y, pA):.4f}, Model B {roc_auc_score(y, pB):.4f}")

# --------------------------------------------------------------- ROC curves
fa, ta, _ = roc_curve(y, pA)
fb, tb, _ = roc_curve(y, pB)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fa, ta, "--", lw=2.6, label=f"Model A (AUC {roc_auc_score(y, pA):.4f})")
ax.plot(fb, tb, "-", lw=2.6, label=f"Model B (AUC {roc_auc_score(y, pB):.4f})")
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1.5)
ax.set_xlabel("False positive rate", fontsize=LABEL)
ax.set_ylabel("True positive rate", fontsize=LABEL)
ax.tick_params(labelsize=TICK)
ax.legend(loc="lower right", fontsize=LEGEND)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "roc_curves.png", dpi=DPI)
plt.close(fig)

# ------------------------------------------------------- calibration curves
fig, ax = plt.subplots(figsize=(5.5, 5))
for p, lab, st in [(pA, "Model A", "--"), (pB, "Model B", "-")]:
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, st, marker="o", ms=7, lw=2.4, label=lab)
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1.5, label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability", fontsize=LABEL)
ax.set_ylabel("Observed default rate", fontsize=LABEL)
ax.tick_params(labelsize=TICK)
ax.legend(fontsize=LEGEND)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "calibration_curves.png", dpi=DPI)
plt.close(fig)

# ------------------------------------------------------------ cost surface
grid = np.arange(0.05, 0.96, 0.01)
flagged = pB[:, None] >= grid[None, :]
pos = (y == 1)[:, None]
fn = (~flagged & pos).sum(axis=0).astype(float)
fp = (flagged & ~pos).sum(axis=0).astype(float)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
for ratio in [1, 2, 3, 5, 10]:
    cost = (fn * ratio + fp) / len(y)
    ax.plot(grid, cost / cost.min(), lw=1.6, label=f"cost ratio {ratio}:1")
ax.axvline(OPERATING_THRESHOLD, color="gray", ls=":", lw=1.2)
ax.axhline(1 + TOLERANCE, color="gray", ls="--", lw=0.9)
ax.annotate(
    "operating threshold 0.30",
    xy=(OPERATING_THRESHOLD, 1.35),
    xytext=(0.42, 1.42),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
)
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Cost relative to the minimum for that ratio")
ax.set_ylim(0.98, 1.6)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "cost_surface.png", dpi=DPI)
plt.close(fig)

print(f"Three figures redrawn at {DPI} DPI.")
