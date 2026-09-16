"""Main experiment: what cooperative membership variables add to default prediction.

Model A = 14 conventional borrower and loan variables
Model B = Model A + 9 cooperative membership variables (23 variables)

Data definitions (final):
  t0                   = application date
  observation window   = 36 months before t0 (all predictors, see the paper for
                         the one exception, payment lateness)
  performance window   = 24 months after t0 (outcome)
  default              = arrears of 90 days or more
  sample default rate  = 30.4 percent (a property of the research sample, not
                         of the cooperative's portfolio)

Run:      python scripts/main_experiment.py
Needs:    scikit-learn 1.7.2, imbalanced-learn, pandas, numpy, matplotlib, scipy
          (versions pinned in requirements.txt; the numbers depend on them)
Writes:   15 CSV files to results/ and 2 PNG files to figures/
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

matplotlib.use("Agg")

DATA_FILE = DATA / "dataset.csv"
PRIMARY_SEED = 42
SEEDS = [42, 1, 7, 13, 99]
OPERATING_THRESHOLD = 0.30

raw = pd.read_csv(DATA_FILE)
df = raw.drop(columns=["member_id"])
y = df.pop("default").values
city = raw["city"].values

# Model A: variables available to any lender. Model B adds the cooperative
# membership block. The split was fixed before the experiment was run.
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

# Category order for the one-hot encoder, fixed to the alphabetical order of the
# original Indonesian labels (see scripts/translate_dataset.py). The encoder
# drops the first category of each variable as the reference level, so the order
# decides which level is dropped and how the columns are laid out. Pinning it
# keeps every fitted model identical to a run on the original labels.
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
for col, levels in CATEGORY_ORDER.items():
    assert set(df[col].unique()) == set(levels), f"{col}: labels differ from CATEGORY_ORDER"

# scikit-learn defaults except where stated. No tuning: the estimand is
# performance under one shared protocol, and both feature sets get identical
# treatment.
MODELS = {
    "Logistic Regression": lambda s: LogisticRegression(max_iter=5000, random_state=s),
    "Decision Tree": lambda s: DecisionTreeClassifier(max_depth=5, random_state=s),
    "Random Forest": lambda s: RandomForestClassifier(n_estimators=300, random_state=s),
    "Gradient Boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "SVM (RBF)": lambda s: SVC(probability=True, random_state=s),
}


def preprocess(cols, data=None):
    """ColumnTransformer: one-hot (drop first) for categoricals, standardisation for numerics."""
    data = df if data is None else data
    cat = [c for c in cols if data[c].dtype.name in ("object", "str")]
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


def oof(cols, make_model, seed, smote=False, data=None):
    """Out-of-fold probabilities from 5-fold stratified cross-validation.

    Preprocessing sits inside the pipeline and is fitted on the training folds
    only, so nothing leaks into the held-out fold. Each member is scored exactly
    once per seed.
    """
    data = df if data is None else data
    X = data[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        steps = [("pre", preprocess(cols, data))]
        if smote:
            steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", make_model(seed)))
        pipe = ImbPipeline(steps) if smote else Pipeline(steps)
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def metrics(y_true, p, t=0.5):
    """Six measures at threshold t. AUC and the Brier score do not depend on t."""
    pred = (p >= t).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, pred),
        Precision=precision_score(y_true, pred, zero_division=0),
        Recall=recall_score(y_true, pred),
        F1=f1_score(y_true, pred),
        AUC=roc_auc_score(y_true, p),
        Brier=brier_score_loss(y_true, p),
    )


def calibration(y_true, p, n_bin=10):
    """Direct calibration evidence, kept separate from discrimination.

    Brier = REL - RES + UNC (Murphy decomposition on quantile bins). REL is the
    calibration component, RES the discrimination component. A Brier score that
    falls because RES rises does NOT show better calibration, so the calibration
    slope (ideal 1) and intercept (ideal 0) are reported alongside.
    """
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    logit = np.log(pc / (1 - pc))
    binom = sm.families.Binomial()
    slope = np.asarray(
        sm.GLM(y_true, sm.add_constant(logit), family=binom).fit().params
    )[1]
    ones = np.ones((len(y_true), 1))
    intercept = np.asarray(
        sm.GLM(y_true, ones, family=binom, offset=logit).fit().params
    )[0]
    edges = np.quantile(pc, np.linspace(0, 1, n_bin + 1))
    idx = np.clip(np.digitize(pc, edges[1:-1]), 0, n_bin - 1)
    obar = y_true.mean()
    rel = res = ece = 0.0
    for k in range(n_bin):
        m = idx == k
        if not m.any():
            continue
        w = m.sum() / len(y_true)
        pbar = pc[m].mean()
        obs = y_true[m].mean()
        rel += w * (pbar - obs) ** 2
        res += w * (obs - obar) ** 2
        ece += w * abs(pbar - obs)
    return dict(
        REL=rel,
        RES=res,
        UNC=obar * (1 - obar),
        slope=slope,
        intercept=intercept,
        ECE=ece,
    )


def fmt_number(v):
    """Whole numbers with thousands separators, fractions with three decimals."""
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 1000 else f"{v:.3f}"


print("=" * 60)
print("0. DESCRIPTIVE STATISTICS")
print("=" * 60)
NUMERIC_DESC = [c for c in FEATURES_B if df[c].dtype.name != "object"]
CATEGORICAL_DESC = [c for c in FEATURES_B if df[c].dtype.name == "object"]
rows = []
for c in NUMERIC_DESC:
    col = df[c]
    rows.append(
        dict(
            variable=c,
            type="numeric",
            level="",
            n=len(col),
            percent="",
            summary=(
                f"{fmt_number(col.median())} "
                f"({fmt_number(col.quantile(0.25))} to {fmt_number(col.quantile(0.75))})"
            ),
            default_rate="",
        )
    )
for c in CATEGORICAL_DESC:
    for lv in CATEGORY_ORDER[c]:
        m = (df[c] == lv).values
        rows.append(
            dict(
                variable=c,
                type="categorical",
                level=str(lv),
                n=int(m.sum()),
                percent=round(m.sum() / len(df) * 100, 1),
                summary="",
                default_rate=round(float(y[m].mean()), 3),
            )
        )
t0 = pd.DataFrame(rows)
t0.to_csv(RESULTS / "descriptive_statistics.csv", index=False)
print(t0.to_string(index=False))
print()
print(
    f"  overall: n={len(df)}, "
    f"defaults={int(y.sum())} ({y.mean() * 100:.1f} percent)"
)

print("=" * 60)
print("1. MODEL A vs MODEL B  (primary seed, threshold 0.5)")
print("=" * 60)
rows = []
for model_name, make_model in MODELS.items():
    for tag, cols in [("A", FEATURES_A), ("B", FEATURES_B)]:
        m = metrics(y, oof(cols, make_model, PRIMARY_SEED))
        rows.append(
            dict(Model=model_name, Features=tag, **{k: round(v, 4) for k, v in m.items()})
        )
t1 = pd.DataFrame(rows)
t1.to_csv(RESULTS / "results_A_vs_B_seed42.csv", index=False)
print(t1.to_string(index=False))

print("\n" + "=" * 60)
print("2. MEAN OVER 5 SEEDS, all measures  -> MAIN TABLE OF THE PAPER")
print("=" * 60)
rows = []
rows_cal = []
for model_name, make_model in MODELS.items():
    for tag, cols in [("A", FEATURES_A), ("B", FEATURES_B)]:
        # Out-of-fold probabilities are computed once per seed and reused for
        # the measures and for the calibration evidence, so nothing is refitted.
        P = [oof(cols, make_model, s) for s in SEEDS]
        M = [metrics(y, p) for p in P]
        agg = {k: round(np.mean([x[k] for x in M]), 4) for k in M[0]}
        agg["sd_AUC"] = round(np.std([x["AUC"] for x in M]), 4)
        rows.append(dict(Model=model_name, Features=tag, **agg))
        K = [calibration(y, p) for p in P]
        # REL, RES, and UNC are small, so they need 5 decimals. Slope, intercept,
        # and ECE are reported with 4 decimals, the same precision as the
        # measures in the main table, so the paper needs no re-rounding.
        decimals = {"REL": 5, "RES": 5, "UNC": 5, "slope": 4, "intercept": 4, "ECE": 4}
        aggk = {k: round(float(np.mean([x[k] for x in K])), decimals[k]) for k in K[0]}
        rows_cal.append(dict(Model=model_name, Features=tag, **aggk))
t2 = pd.DataFrame(rows)
t2.to_csv(RESULTS / "mean_over_five_seeds.csv", index=False)
print(t2.to_string(index=False))
print("largest between-seed sd of AUC:", t2.sd_AUC.max())

print("\n" + "=" * 60)
print("2b. DIRECT CALIBRATION EVIDENCE  (Brier decomposition, slope, intercept)")
print("=" * 60)
cal = pd.DataFrame(rows_cal)
cal.to_csv(RESULTS / "calibration_detail.csv", index=False)
print(cal.to_string(index=False))
print(
    "\nDifference B - A. REL down = better calibration, RES up = better discrimination:"
)
for model_name in cal.Model.unique():
    a = cal[(cal.Model == model_name) & (cal.Features == "A")].iloc[0]
    b = cal[(cal.Model == model_name) & (cal.Features == "B")].iloc[0]
    print(
        f"  {model_name:<20} dREL={b.REL - a.REL:+.5f}  dRES={b.RES - a.RES:+.5f}  "
        f"dECE={b.ECE - a.ECE:+.5f}  slope {a.slope:.3f}->{b.slope:.3f}  "
        f"intercept {a.intercept:+.3f}->{b.intercept:+.3f}"
    )

print("\n" + "=" * 60)
print("3. THRESHOLD SWEEP  (Logistic Regression)")
print("=" * 60)
pA = oof(FEATURES_A, MODELS["Logistic Regression"], PRIMARY_SEED)
pB = oof(FEATURES_B, MODELS["Logistic Regression"], PRIMARY_SEED)
rows = []
for t in np.arange(0.20, 0.65, 0.05):
    for tag, p in [("A", pA), ("B", pB)]:
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
            )
        )
sw = pd.DataFrame(rows)
sw.to_csv(RESULTS / "threshold_sweep.csv", index=False)
columns = ["Threshold", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN"]
print(sw[sw.Features == "B"][columns].to_string(index=False))

print("\n" + "=" * 60)
print("4. SMOTE vs NO SMOTE  (SMOTE on the training folds only)")
print("=" * 60)
rows = []
rows_cal = []
for model_name, make_model in MODELS.items():
    for smote in [False, True]:
        # out-of-fold probabilities are computed once and used for both the
        # measures and the calibration, so both tables describe the same predictions
        P = [oof(FEATURES_B, make_model, s, smote=smote) for s in SEEDS[:3]]
        M = [metrics(y, p) for p in P]
        K = [calibration(y, p) for p in P]
        label = "Yes" if smote else "No"
        rows.append(
            dict(
                Model=model_name,
                SMOTE=label,
                **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
            )
        )
        rows_cal.append(
            dict(
                Model=model_name,
                SMOTE=label,
                Brier=round(np.mean([x["Brier"] for x in M]), 5),
                **{k: round(np.mean([x[k] for x in K]), 5) for k in K[0]},
            )
        )
sm_df = pd.DataFrame(rows)
sm_df.to_csv(RESULTS / "smote_vs_none.csv", index=False)
print(sm_df.to_string(index=False))

# Calibration under SMOTE. The clipping constant in calibration() is unchanged,
# so the note about the decision tree's slope sensitivity applies unchanged.
print("\n4b. CALIBRATION UNDER SMOTE  (the same three seeds)")
cal_sm = pd.DataFrame(rows_cal)
cal_sm.to_csv(RESULTS / "calibration_smote.csv", index=False)
print(cal_sm.to_string(index=False))
for model_name in MODELS:
    a = cal_sm[(cal_sm.Model == model_name) & (cal_sm.SMOTE == "No")].iloc[0]
    b = cal_sm[(cal_sm.Model == model_name) & (cal_sm.SMOTE == "Yes")].iloc[0]
    print(
        f"  {model_name:20} REL {b.REL - a.REL:+.5f}  ECE {b.ECE - a.ECE:+.5f}  "
        f"|slope-1| {abs(b.slope - 1) - abs(a.slope - 1):+.4f}  "
        f"Brier {b.Brier - a.Brier:+.5f}"
    )

# The direct comparison quoted in the paper: moving the threshold without SMOTE
# against SMOTE at the default threshold. Specificity is reported too, because
# the replication claim concerns the balance of sensitivity and specificity,
# not F1 alone.
print("\n4c. THRESHOLD 0.30 WITHOUT SMOTE  vs  THRESHOLD 0.50 WITH SMOTE  (logistic regression)")
rows_b = []
for label, smote, t in [
    ("Without SMOTE", False, OPERATING_THRESHOLD),
    ("With SMOTE", True, 0.50),
]:
    M = []
    for s in SEEDS[:3]:
        p = oof(FEATURES_B, MODELS["Logistic Regression"], s, smote=smote)
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        M.append(
            dict(
                Recall=recall_score(y, pred),
                Specificity=tn / (tn + fp),
                Precision=precision_score(y, pred, zero_division=0),
                F1=f1_score(y, pred),
                TP=tp,
                FP=fp,
                FN=fn,
                TN=tn,
            )
        )
    rows_b.append(
        dict(
            Condition=label,
            Threshold=t,
            **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
        )
    )
thr_sm = pd.DataFrame(rows_b)
thr_sm.to_csv(RESULTS / "threshold_vs_smote.csv", index=False)
print(thr_sm.to_string(index=False))

print("\n" + "=" * 60)
print("5. COST RATIO SWEEP  (cost of FN : cost of FP)")
print("=" * 60)
grid = np.arange(0.05, 0.96, 0.01)
rows = []
for ratio in [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]:
    best = {}
    for tag, p in [("A", pA), ("B", pB)]:
        cost = []
        for t in grid:
            tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
            cost.append((fn * ratio + fp) / len(y))
        i = int(np.argmin(cost))
        best[tag] = (round(grid[i], 2), cost[i])
    rows.append(
        dict(
            ratio=ratio,
            threshold_A=best["A"][0],
            cost_A=round(best["A"][1], 4),
            threshold_B=best["B"][0],
            cost_B=round(best["B"][1], 4),
            saving=round(best["A"][1] - best["B"][1], 4),
        )
    )
co = pd.DataFrame(rows)
co.to_csv(RESULTS / "cost_ratio_sweep.csv", index=False)
print(co.to_string(index=False))

print("\n" + "=" * 60)
print(f"6. GROUP DIFFERENCES  (threshold {OPERATING_THRESHOLD})")
print("=" * 60)
rows = []
for attr in ["gender", "marital_status", "education"]:
    for g in raw[attr].unique():
        mask = (raw[attr] == g).values
        n = int(mask.sum())
        if n < 30:
            continue
        out = {}
        for tag, p in [("A", pA), ("B", pB)]:
            pr = (p[mask] >= OPERATING_THRESHOLD).astype(int)
            yy = y[mask]
            tp = int(((pr == 1) & (yy == 1)).sum())
            fn = int(((pr == 0) & (yy == 1)).sum())
            fp = int(((pr == 1) & (yy == 0)).sum())
            tn = int(((pr == 0) & (yy == 0)).sum())
            recall = tp / (tp + fn) if tp + fn else np.nan
            fpr = fp / (fp + tn) if fp + tn else np.nan
            out[tag] = (pr.mean(), recall, fpr)
        rows.append(
            dict(
                attribute=attr,
                group=g,
                n=n,
                default_rate=round(y[mask].mean(), 3),
                flagged_A=round(out["A"][0], 3),
                recall_A=round(out["A"][1], 3),
                FPR_A=round(out["A"][2], 3),
                flagged_B=round(out["B"][0], 3),
                recall_B=round(out["B"][1], 3),
                FPR_B=round(out["B"][2], 3),
            )
        )
fair = pd.DataFrame(rows)
fair.to_csv(RESULTS / "group_differences.csv", index=False)
print(fair.to_string(index=False))
print("\nRANGE across groups (max - min), smaller is more even:")
for attr in fair.attribute.unique():
    s = fair[fair.attribute == attr]
    for met in ["flagged", "recall", "FPR"]:
        ra = s[f"{met}_A"].max() - s[f"{met}_A"].min()
        rb = s[f"{met}_B"].max() - s[f"{met}_B"].min()
        print(f"  {attr:<18} {met:<10} A={ra:.3f}  B={rb:.3f}  diff {rb - ra:+.3f}")

print("\n" + "=" * 60)
print("7. LEAVE-ONE-CITY-OUT VALIDATION  (city removed from the features)")
print("=" * 60)
A_nc = [c for c in FEATURES_A if c != "city"]
B_nc = A_nc + FEATURES_MEMBERSHIP
rows = []
for model_name in ["Logistic Regression", "Gradient Boosting"]:
    make_model = MODELS[model_name]
    for k in sorted(set(city)):
        te = city == k
        tr = ~te
        res = {}
        for tag, cols in [("A", A_nc), ("B", B_nc)]:
            X = df[cols]
            pipe = Pipeline([("pre", preprocess(cols)), ("clf", make_model(PRIMARY_SEED))])
            pipe.fit(X[tr], y[tr])
            res[tag] = roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])
        rows.append(
            dict(
                Model=model_name,
                city=k,
                n_test=int(te.sum()),
                AUC_A=round(res["A"], 4),
                AUC_B=round(res["B"], 4),
                diff=round(res["B"] - res["A"], 4),
            )
        )
vk = pd.DataFrame(rows)
vk.to_csv(RESULTS / "leave_one_city_out.csv", index=False)
print(vk.to_string(index=False))
for model_name in vk.Model.unique():
    s = vk[vk.Model == model_name]
    print(
        f"  {model_name}: mean AUC A={s.AUC_A.mean():.4f} B={s.AUC_B.mean():.4f} "
        f"diff {s['diff'].mean():+.4f} | "
        f"B higher in {(s['diff'] > 0).sum()} of {len(s)} cities"
    )

print("\n" + "=" * 60)
print("8. COEFFICIENTS AND FIGURES")
print("=" * 60)
pipe = Pipeline(
    [
        ("pre", preprocess(FEATURES_B)),
        ("clf", LogisticRegression(max_iter=5000, random_state=PRIMARY_SEED)),
    ]
)
pipe.fit(df[FEATURES_B], y)
feature_names = [
    x.replace("cat__", "").replace("num__", "")
    for x in pipe.named_steps["pre"].get_feature_names_out()
]
cf = pipe.named_steps["clf"].coef_[0]
imp = pd.DataFrame(
    {"feature": feature_names, "coefficient": cf.round(4), "abs": np.abs(cf).round(4)}
)
imp["block"] = [
    "membership" if any(k in x for k in FEATURES_MEMBERSHIP) else "conventional"
    for x in imp.feature
]
imp.sort_values("abs", ascending=False).to_csv(RESULTS / "logreg_coefficients_B.csv", index=False)

fa, ta, _ = roc_curve(y, pA)
fb, tb, _ = roc_curve(y, pB)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fa, ta, "--", lw=1.8, label=f"Model A (AUC {roc_auc_score(y, pA):.4f})")
ax.plot(fb, tb, "-", lw=1.8, label=f"Model B (AUC {roc_auc_score(y, pB):.4f})")
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "roc_curves.png", dpi=600)

fig, ax = plt.subplots(figsize=(5.5, 5))
for p, lab, st in [(pA, "Model A", "--"), (pB, "Model B", "-")]:
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, st, marker="o", ms=4, lw=1.6, label=lab)
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1, label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed default rate")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "calibration_curves.png", dpi=600)

for t in (0.50, OPERATING_THRESHOLD):
    tn, fp, fn, tp = confusion_matrix(y, (pB >= t).astype(int)).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(
        f"  confusion matrix at threshold {t:.2f} -> TP {tp} FP {fp} FN {fn} TN {tn} | "
        f"recall {recall:.4f} precision {precision:.4f}"
    )
print("\n" + "=" * 60)
print("9. SENSITIVITY OF THE COEFFICIENT RANKING TO THE PENALTY")
print("=" * 60)


def logreg_coefficients(C):
    """Model B logistic regression coefficients at a given penalty strength."""
    pipe_c = Pipeline(
        [
            ("pre", preprocess(FEATURES_B)),
            ("clf", LogisticRegression(max_iter=5000, random_state=PRIMARY_SEED, C=C)),
        ]
    )
    pipe_c.fit(df[FEATURES_B], y)
    names = [
        x.replace("cat__", "").replace("num__", "")
        for x in pipe_c.named_steps["pre"].get_feature_names_out()
    ]
    return pd.Series(pipe_c.named_steps["clf"].coef_[0], index=names)


# scikit-learn's LogisticRegression uses an L2 penalty with C=1.0 by default.
# The penalty shrinks the coefficients, and it does NOT shrink standardised
# numeric variables and unscaled 0/1 dummies equally. This block measures how
# much the size and ranking of the coefficients depend on the penalty.
numeric = set(df[FEATURES_B].select_dtypes("number").columns)
k_used = logreg_coefficients(1.0)
k_none = logreg_coefficients(1e6)
sens = pd.DataFrame(
    {
        "coef_C1": k_used.round(4),
        "coef_no_penalty": k_none.round(4),
        "shrinkage_pct": ((1 - k_used.abs() / k_none.abs()) * 100).round(1),
        "type": ["numeric" if f in numeric else "dummy" for f in k_used.index],
    }
)
sens = sens.sort_values("coef_C1", key=lambda s: s.abs(), ascending=False)
sens.index.name = "feature"
sens.to_csv(RESULTS / "sensitivity_penalty.csv")
print(sens.head(8).to_string())
print("\n  top five WITH the penalty    :", list(k_used.abs().nlargest(5).index))
print("  top five WITHOUT the penalty :", list(k_none.abs().nlargest(5).index))

print("\n" + "=" * 60)
print("10. SENSITIVITY TO THE DEFINITION OF payment_lateness_months")
print("=" * 60)
# Payment lateness counts the 12 months before data extraction AND includes the
# loan under evaluation, so it may carry information from the outcome window.
# This block repeats the A versus B comparison without that variable, to test
# whether the membership block's contribution depends on a variable whose
# definition is problematic.
A_NO_LATENESS = [c for c in FEATURES_A if c != "payment_lateness_months"]
B_NO_LATENESS = A_NO_LATENESS + FEATURES_MEMBERSHIP
rows = []
for model_name, make_model in MODELS.items():
    for tag, cols in [("A", A_NO_LATENESS), ("B", B_NO_LATENESS)]:
        M = [metrics(y, oof(cols, make_model, s)) for s in SEEDS]
        rows.append(
            dict(
                Model=model_name,
                Features=tag,
                **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
            )
        )
sens_l = pd.DataFrame(rows)
sens_l.to_csv(RESULTS / "sensitivity_payment_lateness.csv", index=False)
print(sens_l.to_string(index=False))
print("\nAUC difference B - A, full feature sets versus without payment lateness:")
p_full = t2.pivot(index="Model", columns="Features", values="AUC")
p_without = sens_l.pivot(index="Model", columns="Features", values="AUC")
for model_name in MODELS:
    d_full = p_full["B"][model_name] - p_full["A"][model_name]
    d_without = p_without["B"][model_name] - p_without["A"][model_name]
    print(
        f"  {model_name:<20} full {d_full:+.4f}   without {d_without:+.4f}   "
        f"change {d_without - d_full:+.4f}"
    )

print("\n" + "=" * 60)
print("11. BLOCK PERMUTATION TEST, 200 REPLICATES")
print("=" * 60)
# The permutation is applied to the BLOCK, not column by column. One random
# order is used for all nine membership columns at once, so the dependence
# structure WITHIN the block stays intact while its link to the member, to the
# conventional variables, and to the outcome is broken. Permuting column by
# column would also destroy that internal structure, so the null hypothesis
# would no longer be about the block's added information. The fold split is
# HELD FIXED (seed 42) so that the measured variation comes from the permutation
# alone, not mixed with fold-split variation.
N_PERM = 200
auc_A_fixed = roc_auc_score(y, oof(FEATURES_A, MODELS["Logistic Regression"], PRIMARY_SEED))
auc_B_fixed = roc_auc_score(y, oof(FEATURES_B, MODELS["Logistic Regression"], PRIMARY_SEED))
observed_diff = auc_B_fixed - auc_A_fixed
null = []
for i in range(N_PERM):
    rng_i = np.random.default_rng(1000 + i)
    dd = df.copy()
    order = rng_i.permutation(len(dd))
    for c in FEATURES_MEMBERSHIP:
        dd[c] = dd[c].values[order]
    null.append(
        roc_auc_score(y, oof(FEATURES_B, MODELS["Logistic Regression"], PRIMARY_SEED, data=dd))
        - auc_A_fixed
    )
null = np.array(null)
n_ge = int((null >= observed_diff).sum())
p_emp = (n_ge + 1) / (N_PERM + 1)
pd.DataFrame(
    {"replicate": np.arange(1, N_PERM + 1), "auc_diff": null.round(6)}
).to_csv(RESULTS / "permutation_null_logreg.csv", index=False)
print(f"  observed difference (seed {PRIMARY_SEED}) = {observed_diff:+.4f}")
print(f"  null distribution from {N_PERM} permutations:")
print(f"    mean {null.mean():+.4f}  sd {null.std(ddof=1):.4f}")
print(f"    min {null.min():+.4f}  max {null.max():+.4f}")
print(
    f"  distance of the observed value from the null mean = {(observed_diff - null.mean()) / null.std(ddof=1):.1f} sd"
)
print(f"  permutations >= observed = {n_ge} of {N_PERM},  empirical p = {p_emp:.4f}")

print("\nDONE. All CSV files and 2 figures written.")
