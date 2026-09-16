"""Statistical inference and the additional checks behind the paper's claims.

This script does NOT modify and does NOT rerun main_experiment.py. The data and
preprocessing helpers are copied from that script unchanged, and Section 0 tests
the copy against the stored CSV. If the copy drifts by even a fourth decimal the
script stops, so the duplicated code cannot diverge silently.

Contents:
  0. Parity guard against results/mean_over_five_seeds.csv
  1. Audit of member_id, the 25th column, which is never used
  2. Statistical inference for the AUC difference between Model A and Model B
  3. Permutation test: the algorithm made explicit, and replicated in a second
     model family
  4. Membership-only model (exploratory)
  5. Shape of the cost surface and stability of the optimal threshold
  6. Univariate AUC for all 23 variables

Run:      python scripts/inference_and_checks.py
Writes:   7 CSV files to results/ and 1 PNG file to figures/
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
N_BOOT = 2000  # bootstrap replicates for the AUC confidence interval
N_BOOT_THRESHOLD = 1000  # bootstrap replicates for the optimal-threshold distribution
SEED_BOOT = 20260802

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
for col, levels in CATEGORY_ORDER.items():
    assert set(df[col].unique()) == set(levels), f"{col}: labels differ from CATEGORY_ORDER"

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


def oof(cols, make_model, seed, data=None):
    """Out-of-fold probabilities from 5-fold stratified CV, identical to the main script."""
    data = df if data is None else data
    X = data[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline([("pre", preprocess(cols, data)), ("clf", make_model(seed))])
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def fast_auc(y_true, score):
    """AUC through the Mann-Whitney rank statistic, with midranks for ties.

    The value equals roc_auc_score, but it is far cheaper when called thousands
    of times inside a bootstrap.
    """
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def midrank(x):
    """Midranks for tied values, used by the DeLong computation."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong(y_true, p1, p2):
    """DeLong test for two correlated ROC curves on the same sample.

    Returns both AUCs, the difference, its standard error, the z statistic, and
    the two-sided p-value. The variance components follow the fast form of Sun
    and Xu.

    CAVEAT that must be reported with the result: the test assumes both score
    vectors are fixed functions evaluated on a random sample. The scores here
    are out-of-fold probabilities from five fold models, so each member is
    scored by a model trained on the other four folds. The scores therefore do
    not come from one fixed model and are weakly dependent through overlapping
    training data. The DeLong variance under this structure is an approximation.
    """
    pos = y_true == 1
    m = int(pos.sum())
    n = len(y_true) - m
    score = np.vstack([np.r_[p1[pos], p1[~pos]], np.r_[p2[pos], p2[~pos]]])
    tx = np.vstack([midrank(score[k, :m]) for k in range(2)])
    ty = np.vstack([midrank(score[k, m:]) for k in range(2)])
    tz = np.vstack([midrank(score[k]) for k in range(2)])
    auc = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    diff = auc[0] - auc[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else np.nan
    return dict(
        AUC_1=auc[0],
        AUC_2=auc[1],
        diff=diff,
        se=se,
        z=z,
        p=2 * norm.sf(abs(z)) if se > 0 else np.nan,
    )


def bootstrap_indices(y_true, n_rep, seed):
    """Bootstrap indices stratified by outcome, so the number of defaults is fixed.

    Without stratification some replicates could lose the whole minority class
    and leave the AUC undefined. Stratification also makes the measured
    variation come purely from the composition of members within each class.
    """
    rng = np.random.default_rng(seed)
    i1 = np.flatnonzero(y_true == 1)
    i0 = np.flatnonzero(y_true == 0)
    for _ in range(n_rep):
        yield np.r_[
            rng.choice(i1, len(i1), replace=True), rng.choice(i0, len(i0), replace=True)
        ]


line = "=" * 70
print(line)
print("0. PARITY GUARD  (copied helpers tested against the stored CSV)")
print(line)
try:
    t2 = pd.read_csv(RESULTS / "mean_over_five_seeds.csv")
except FileNotFoundError:
    sys.exit("results/mean_over_five_seeds.csv not found. Run main_experiment.py first.")

OOF = {}  # (model_name, tag) -> list of out-of-fold probabilities per seed
for model_name in ["Logistic Regression", "Gradient Boosting"]:
    for tag, cols in [("A", FEATURES_A), ("B", FEATURES_B)]:
        OOF[(model_name, tag)] = [oof(cols, MODELS[model_name], s) for s in SEEDS]
        auc_new = round(
            float(np.mean([fast_auc(y, p) for p in OOF[(model_name, tag)]])), 4
        )
        auc_stored = float(
            t2[(t2.Model == model_name) & (t2.Features == tag)].iloc[0]["AUC"]
        )
        status = "match" if auc_new == auc_stored else "MISMATCH"
        print(f"  {model_name:<20} {tag}  AUC stored {auc_stored:.4f}  recomputed {auc_new:.4f}  {status}")
        if auc_new != auc_stored:
            sys.exit(
                "Parity failed. The copied helpers are no longer identical to the main "
                "script, so nothing below would be comparable. Script stopped."
            )
print("  parity passed, continuing.")

print("\n" + line)
print("1. AUDIT OF member_id  (the 25th column)")
print(line)
# The paper says the file has 25 columns, while 23 variables + 1 outcome = 24.
# The remaining column is member_id. This section proves that it never enters
# the feature matrix on any path, and also measures whether it would carry any
# signal if it did leak.
A_nc = [c for c in FEATURES_A if c != "city"]
A_NO_LATENESS = [c for c in FEATURES_A if c != "payment_lateness_months"]
ALL_PATHS = {
    "Model A": FEATURES_A,
    "Model B": FEATURES_B,
    "Model A without city (leave-one-city-out)": A_nc,
    "Model B without city (leave-one-city-out)": A_nc + FEATURES_MEMBERSHIP,
    "Model A without payment lateness": A_NO_LATENESS,
    "Model B without payment lateness": A_NO_LATENESS + FEATURES_MEMBERSHIP,
    "Membership block only": FEATURES_MEMBERSHIP,
}
audit_rows = []
print(f"  columns in the raw file          : {raw.shape[1]}")
print(f"  columns after id and outcome     : {df.shape[1]}  (= 23 predictors)")
print(f"  member_id present in df?         : {'member_id' in df.columns}")
for path_name, cols in ALL_PATHS.items():
    in_list = "member_id" in cols
    # binding evidence: the feature names that actually come out of the fitted preprocessing
    pre = preprocess(cols)
    pre.fit(df[cols])
    names_out = list(pre.get_feature_names_out())
    in_matrix = any("member_id" in x for x in names_out)
    audit_rows.append(
        dict(
            path=path_name,
            n_variables=len(cols),
            n_matrix_columns=len(names_out),
            id_in_feature_list=in_list,
            id_in_fitted_matrix=in_matrix,
        )
    )
    print(
        f"  {path_name:<42} {len(cols):2d} variables -> {len(names_out):2d} columns | "
        f"id in matrix: {in_matrix}"
    )
id_auc = fast_auc(y, raw["member_id"].values)
corr = float(np.corrcoef(raw["member_id"].values, y)[0, 1])
print(f"\n  univariate AUC of member_id      : {id_auc:.4f}  (0.5 = no information)")
print(f"  correlation of member_id with y  : {corr:+.4f}")
pd.DataFrame(audit_rows).to_csv(RESULTS / "member_id_audit.csv", index=False)
if any(b["id_in_fitted_matrix"] for b in audit_rows):
    sys.exit("LEAK: member_id entered the feature matrix. Stop and fix.")
print("  verdict: member_id never enters the feature matrix on any path.")

print("\n" + line)
print("2. STATISTICAL INFERENCE FOR THE AUC DIFFERENCE")
print(line)
# The between-seed SD already reported in the paper measures FOLD-SPLIT
# variation. This section measures MEMBER-SAMPLING variation, a different
# question. Both are reported side by side, not as substitutes.
#
# DECISION RULE fixed in advance: Model B is said to improve discrimination for
# an algorithm ONLY IF the 95 percent bootstrap interval of the seed-averaged
# AUC difference excludes zero AND the DeLong test at seed 42 gives p < 0.05.
# If the two disagree, the claim is downgraded to not established.
for model_name in MODELS:
    if (model_name, "A") not in OOF:
        OOF[(model_name, "A")] = [oof(FEATURES_A, MODELS[model_name], s) for s in SEEDS]
        OOF[(model_name, "B")] = [oof(FEATURES_B, MODELS[model_name], s) for s in SEEDS]

inference_rows = []
for model_name in MODELS:
    PA, PB = OOF[(model_name, "A")], OOF[(model_name, "B")]
    d_seed = np.array([fast_auc(y, b) - fast_auc(y, a) for a, b in zip(PA, PB)])
    # bootstrap: one resample of members is applied to ALL FIVE seeds at once,
    # so the interval covers the seed-averaged difference, not a single seed
    boot = []
    for idx in bootstrap_indices(y, N_BOOT, SEED_BOOT):
        yb = y[idx]
        boot.append(
            np.mean([fast_auc(yb, b[idx]) - fast_auc(yb, a[idx]) for a, b in zip(PA, PB)])
        )
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dl = delong(y, PB[0], PA[0])  # seed 42 = SEEDS[0]
    supported = (lo > 0 or hi < 0) and dl["p"] < 0.05
    inference_rows.append(
        dict(
            Model=model_name,
            mean_diff_over_seeds=round(float(d_seed.mean()), 4),
            sd_across_seeds=round(float(d_seed.std(ddof=0)), 4),
            boot_lo=round(float(lo), 4),
            boot_hi=round(float(hi), 4),
            boot_se=round(float(boot.std(ddof=1)), 4),
            delong_diff_seed42=round(float(dl["diff"]), 4),
            delong_se=round(float(dl["se"]), 4),
            delong_z=round(float(dl["z"]), 3),
            delong_p=float(f"{dl['p']:.3e}"),
            supported=supported,
        )
    )
    print(
        f"  {model_name:<20} d={d_seed.mean():+.4f}  "
        f"CI95 [{lo:+.4f}, {hi:+.4f}]  DeLong z={dl['z']:.2f} p={dl['p']:.2e}  "
        f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
    )
inf = pd.DataFrame(inference_rows)
inf.to_csv(RESULTS / "auc_difference_inference.csv", index=False)

print("\n" + line)
print("3. PERMUTATION TEST: ALGORITHM MADE EXPLICIT, REPLICATED IN A SECOND MODEL")
print(line)
# An earlier draft wrote "the AUC difference between the two feature sets"
# without naming the model. The configuration the main script actually uses is
# logistic regression at seed 42 with the fold split held fixed. Below, that
# configuration is stated explicitly and then repeated for Gradient Boosting, a
# model family of a different paradigm, with exactly the same permutation design.
N_PERM = 200
N_PERM_FINE = 2000  # logistic regression only, to sharpen the resolution of p


def permutation_test(model_name, n_perm):
    make_model = MODELS[model_name]
    auc_A = fast_auc(y, oof(FEATURES_A, make_model, PRIMARY_SEED))
    auc_B = fast_auc(y, oof(FEATURES_B, make_model, PRIMARY_SEED))
    observed = auc_B - auc_A
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng_i = np.random.default_rng(1000 + i)
        dd = df.copy()
        order = rng_i.permutation(len(dd))
        for c in FEATURES_MEMBERSHIP:
            dd[c] = dd[c].values[order]
        null[i] = fast_auc(y, oof(FEATURES_B, make_model, PRIMARY_SEED, data=dd)) - auc_A
    n_ge = int((null >= observed).sum())
    return dict(
        Model=model_name,
        n_permutations=n_perm,
        AUC_A=round(float(auc_A), 4),
        AUC_B=round(float(auc_B), 4),
        observed_diff=round(float(observed), 6),
        null_mean=round(float(null.mean()), 5),
        null_sd=round(float(null.std(ddof=1)), 5),
        null_min=round(float(null.min()), 5),
        null_max=round(float(null.max()), 5),
        deviation_in_sd=round(float((observed - null.mean()) / null.std(ddof=1)), 2),
        n_ge_observed=n_ge,
        p_empirical=round((n_ge + 1) / (n_perm + 1), 5),
    ), null


perm_rows = []
null_dist = {}
for model_name, n_perm in [
    ("Logistic Regression", N_PERM),
    ("Logistic Regression", N_PERM_FINE),
    ("Gradient Boosting", N_PERM),
]:
    result, null = permutation_test(model_name, n_perm)
    perm_rows.append(result)
    null_dist[(model_name, n_perm)] = null
    print(
        f"  {model_name:<20} n={n_perm:5d}  observed {result['observed_diff']:+.6f}  "
        f"null {result['null_mean']:+.5f} sd {result['null_sd']:.5f}  "
        f"{result['deviation_in_sd']:.2f} SD  p={result['p_empirical']:.5f}"
    )
pd.DataFrame(perm_rows).to_csv(RESULTS / "permutation_two_models.csv", index=False)

print("\n" + line)
print("4. MEMBERSHIP-ONLY MODEL  (EXPLORATORY)")
print(line)
# The permutation test settles the dimensionality argument. This section asks
# a different question: how much standalone signal is there in the nine
# membership variables. Its status is EXPLORATORY. The paper's main comparison
# stays Model A versus Model B, and the numbers below must not be used to move
# that claim in either direction.
membership_rows = []
for model_name in MODELS:
    PM = [oof(FEATURES_MEMBERSHIP, MODELS[model_name], s) for s in SEEDS]
    auc_M = np.array([fast_auc(y, p) for p in PM])
    auc_A = np.array([fast_auc(y, p) for p in OOF[(model_name, "A")]])
    auc_B = np.array([fast_auc(y, p) for p in OOF[(model_name, "B")]])
    boot = []
    for idx in bootstrap_indices(y, N_BOOT, SEED_BOOT + 1):
        yb = y[idx]
        boot.append(np.mean([fast_auc(yb, p[idx]) for p in PM]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    membership_rows.append(
        dict(
            Model=model_name,
            AUC_membership_only=round(float(auc_M.mean()), 4),
            sd_across_seeds=round(float(auc_M.std(ddof=0)), 4),
            boot_lo=round(float(lo), 4),
            boot_hi=round(float(hi), 4),
            AUC_A=round(float(auc_A.mean()), 4),
            AUC_B=round(float(auc_B.mean()), 4),
        )
    )
    print(
        f"  {model_name:<20} membership only {auc_M.mean():.4f} "
        f"CI95 [{lo:.4f}, {hi:.4f}] | A {auc_A.mean():.4f} | B {auc_B.mean():.4f}"
    )
pd.DataFrame(membership_rows).to_csv(RESULTS / "membership_only_model.csv", index=False)

print("\n" + line)
print("5. SHAPE OF THE COST SURFACE AND STABILITY OF THE OPTIMAL THRESHOLD")
print(line)
# The paper reports only the argmin per cost ratio. An argmin that jumps
# (0.48, then 0.46, then 0.34) is a symptom of a flat surface, not of a
# threshold that truly moves far. This section reports the shape of the curve:
# the width of the region whose cost is within 1 percent of the minimum, the
# cost at the fixed operating threshold 0.30, and the spread of the argmin
# under a member bootstrap.
pA = OOF[("Logistic Regression", "A")][0]
pB = OOF[("Logistic Regression", "B")][0]
grid = np.arange(0.05, 0.96, 0.01)
RATIOS = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
TOLERANCE = 0.01  # "practically equivalent" = cost at most 1 percent above the minimum


def count_fn_fp(y_true, p, thresholds):
    """Numbers of false negatives and false positives at each threshold of the grid."""
    flagged = p[:, None] >= thresholds[None, :]
    pos = (y_true == 1)[:, None]
    fn = (~flagged & pos).sum(axis=0)
    fp = (flagged & ~pos).sum(axis=0)
    return fn.astype(float), fp.astype(float)


fnA, fpA = count_fn_fp(y, pA, grid)
fnB, fpB = count_fn_fp(y, pB, grid)
i_operating = int(np.argmin(np.abs(grid - OPERATING_THRESHOLD)))
cost_rows = []
curves = {}
for ratio in RATIOS:
    for tag, fn, fp in [("A", fnA, fpA), ("B", fnB, fpB)]:
        cost = (fn * ratio + fp) / len(y)
        i = int(np.argmin(cost))
        flat = grid[cost <= cost[i] * (1 + TOLERANCE)]
        curves[(ratio, tag)] = cost
        cost_rows.append(
            dict(
                ratio=ratio,
                Features=tag,
                threshold_argmin=round(float(grid[i]), 2),
                cost_min=round(float(cost[i]), 4),
                flat_from=round(float(flat.min()), 2),
                flat_to=round(float(flat.max()), 2),
                flat_width=round(float(flat.max() - flat.min()), 2),
                cost_at_030=round(float(cost[i_operating]), 4),
                excess_cost_at_030_pct=round(
                    float((cost[i_operating] / cost[i] - 1) * 100), 2
                ),
                threshold_030_in_flat_region=bool(
                    cost[i_operating] <= cost[i] * (1 + TOLERANCE)
                ),
            )
        )
cs = pd.DataFrame(cost_rows)
cs.to_csv(RESULTS / "cost_surface.csv", index=False)
print(cs[cs.Features == "B"].to_string(index=False))

# Spread of the argmin under a member bootstrap. If the surface is flat, the
# argmin scatters widely even though the cost barely changes.
print("\n5b. DISTRIBUTION OF THE OPTIMAL THRESHOLD UNDER A MEMBER BOOTSTRAP (Model B)")
argmin_boot = {r: [] for r in RATIOS}
for idx in bootstrap_indices(y, N_BOOT_THRESHOLD, SEED_BOOT + 2):
    fn_b, fp_b = count_fn_fp(y[idx], pB[idx], grid)
    for ratio in RATIOS:
        argmin_boot[ratio].append(grid[int(np.argmin((fn_b * ratio + fp_b) / len(idx)))])
stability_rows = []
for ratio in RATIOS:
    a = np.array(argmin_boot[ratio])
    q = np.percentile(a, [5, 25, 50, 75, 95])
    stability_rows.append(
        dict(
            ratio=ratio,
            argmin_full_sample=round(
                float(cs[(cs.ratio == ratio) & (cs.Features == "B")].iloc[0]["threshold_argmin"]), 2
            ),
            boot_median=round(float(q[2]), 2),
            boot_p5=round(float(q[0]), 2),
            boot_p25=round(float(q[1]), 2),
            boot_p75=round(float(q[3]), 2),
            boot_p95=round(float(q[4]), 2),
            range_p5_p95=round(float(q[4] - q[0]), 2),
        )
    )
    print(
        f"  ratio {ratio:<4} full-sample argmin {stability_rows[-1]['argmin_full_sample']:.2f}  "
        f"bootstrap median {q[2]:.2f}  p5-p95 [{q[0]:.2f}, {q[4]:.2f}]  "
        f"width {q[4] - q[0]:.2f}"
    )
pd.DataFrame(stability_rows).to_csv(RESULTS / "argmin_stability.csv", index=False)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
for ratio in [1, 2, 3, 5, 10]:
    b = curves[(ratio, "B")]
    ax.plot(grid, b / b.min(), lw=1.6, label=f"cost ratio {ratio}:1")
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
fig.savefig(FIGURES / "cost_surface.png", dpi=600)
print("  cost_surface.png written")

print("\n" + line)
print("6. UNIVARIATE AUC, 23 VARIABLES")
print(line)
# Three definitions, stated explicitly so they can be reused:
#   raw univariate AUC       : the variable's value is used directly as the
#                              score, larger taken to indicate default. Defined
#                              only for numeric variables, because categorical
#                              variables have no inherent order.
#   directed univariate AUC  : max(raw, 1 - raw). The variable is oriented in
#                              the direction that indicates default, so the value
#                              is always >= 0.5 and measures signal strength
#                              regardless of direction.
#   single-variable model AUC: AUC of the out-of-fold probabilities of a logistic
#                              regression trained on that variable ALONE, with
#                              the same preprocessing and fold splits as the main
#                              experiment. This measure is UNIFORM across numeric
#                              and categorical variables, and does not use the
#                              outcome to order the levels.
uni_rows = []
for c in FEATURES_B:
    categorical = df[c].dtype.name == "object"
    if categorical:
        raw_auc = directed = np.nan
    else:
        raw_auc = fast_auc(y, df[c].values.astype(float))
        directed = max(raw_auc, 1 - raw_auc)
    p_single = np.mean(
        [
            fast_auc(y, oof([c], MODELS["Logistic Regression"], s))
            for s in SEEDS
        ]
    )
    uni_rows.append(
        dict(
            variable=c,
            block="membership" if c in FEATURES_MEMBERSHIP else "conventional",
            type="categorical" if categorical else "numeric",
            n_levels=int(df[c].nunique()) if categorical else "",
            auc_raw=round(float(raw_auc), 4) if not categorical else "",
            auc_directed=round(float(directed), 4) if not categorical else "",
            auc_single_variable_model=round(float(p_single), 4),
        )
    )
uni = pd.DataFrame(uni_rows).sort_values(
    "auc_single_variable_model", ascending=False, kind="mergesort"
)
uni.to_csv(RESULTS / "univariate_auc.csv", index=False)
print(uni.to_string(index=False))

print("\n  CHECK OF THREE VALUES QUOTED IN EARLIER DRAFTS:")
for variable, label, quoted, column in [
    ("principal_savings_idr", "principal savings", 0.5205, "auc_directed"),
    ("prior_loans", "prior-loan history, directed", 0.5040, "auc_directed"),
    ("prior_loans", "prior-loan history, raw", 0.4960, "auc_raw"),
    ("payment_lateness_months", "payment lateness", 0.5443, "auc_directed"),
]:
    r = uni[uni.variable == variable].iloc[0]
    computed = r[column]
    flag = "match" if abs(float(computed) - quoted) < 5e-5 else "MISMATCH"
    print(f"    {label:<32} quoted {quoted:.4f}  computed {float(computed):.4f}  {flag}")

print("\nDONE. Seven CSV files and one figure written.")
