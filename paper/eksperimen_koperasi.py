"""Eksperimen credit scoring koperasi simpan pinjam.

Model A = peubah pinjaman konvensional + demografi (14 peubah)
Model B = Model A + peubah keanggotaan koperasi (23 peubah)

Definisi data (final):
  t0                 = tanggal pengajuan
  jendela pengamatan = 36 bulan sebelum t0  (seluruh peubah penjelas)
  jendela kinerja    = 24 bulan setelah t0  (luaran)
  gagal bayar        = tunggakan 90 hari atau lebih
  gagal bayar sampel = 30,4 persen (angka SAMPEL PENELITIAN, bukan tingkat portofolio)

Jalankan:  python eksperimen_koperasi.py
Butuh:     scikit-learn 1.7.2, imbalanced-learn, pandas, numpy, matplotlib, scipy
           (versi di-pin pada requirements.txt; vonis numerik bergantung versi)
Keluaran:  15 berkas CSV + 2 gambar PNG, ditulis ke direktori kerja.
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

matplotlib.use("Agg")

BERKAS = "dataset.csv"
SEED_UTAMA = 42
SEEDS = [42, 1, 7, 13, 99]
AMBANG_OPERASI = 0.30

raw = pd.read_csv(BERKAS)
df = raw.drop(columns=["id_anggota"])
y = df.pop("status_pinjaman").values
kota = raw["kota"].values

# Model A: peubah yang tersedia di lembaga keuangan mana pun. Model B menambahkan
# blok keanggotaan koperasi. Pembagian ini ditetapkan sebelum eksperimen dijalankan.
FIT_A = [
    "usia",
    "jenis_kelamin",
    "status_perkawinan",
    "pendidikan",
    "pekerjaan",
    "kota",
    "jumlah_tanggungan",
    "penghasilan_bulanan",
    "jumlah_pinjaman",
    "tenor_pinjaman",
    "suku_bunga",
    "rasio_hutang",
    "riwayat_pinjaman",
    "keterlambatan_bayar",
]
FIT_KOP = [
    "lama_keanggotaan",
    "simpanan_pokok",
    "simpanan_wajib",
    "simpanan_sukarela",
    "jenis_anggota",
    "partisipasi_rat",
    "sistem_potong_gaji",
    "status_keaktifan",
    "transaksi_koperasi",
]
FIT_B = FIT_A + FIT_KOP

# Hiperparameter default scikit-learn kecuali yang disebut. Penyetelan tidak
# dilakukan: estimand penelitian adalah kinerja di bawah satu protokol tunggal,
# dan kedua set fitur menerima perlakuan identik.
MODEL = {
    "Logistic Regression": lambda s: LogisticRegression(max_iter=5000, random_state=s),
    "Decision Tree": lambda s: DecisionTreeClassifier(max_depth=5, random_state=s),
    "Random Forest": lambda s: RandomForestClassifier(n_estimators=300, random_state=s),
    "Gradient Boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "SVM (RBF)": lambda s: SVC(probability=True, random_state=s),
}


def pra(cols, data=None):
    """ColumnTransformer: one-hot (drop pertama) kategorikal, standardisasi numerik."""
    data = df if data is None else data
    cat = [c for c in cols if data[c].dtype.name in ("object", "str")]
    num = [c for c in cols if c not in cat]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
            ("num", StandardScaler(), num),
        ]
    )


def oof(cols, make_model, seed, smote=False, data=None):
    """Peluang out-of-fold dari 5-fold stratified CV.

    Praproses dipasang di dalam pipeline dan dilatih hanya pada lipatan latih,
    sehingga tidak ada kebocoran ke lipatan uji. Tiap anggota dinilai tepat
    sekali per seed.
    """
    data = df if data is None else data
    X = data[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        steps = [("pra", pra(cols, data))]
        if smote:
            steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", make_model(seed)))
        pipe = ImbPipeline(steps) if smote else Pipeline(steps)
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def metrik(y_true, p, t=0.5):
    """Enam metrik pada ambang t. AUC dan Brier tidak bergantung ambang."""
    pred = (p >= t).astype(int)
    return dict(
        Accuracy=accuracy_score(y_true, pred),
        Precision=precision_score(y_true, pred, zero_division=0),
        Recall=recall_score(y_true, pred),
        F1=f1_score(y_true, pred),
        AUC=roc_auc_score(y_true, p),
        Brier=brier_score_loss(y_true, p),
    )


def kalibrasi(y_true, p, n_bin=10):
    """Bukti kalibrasi langsung, dipisahkan dari diskriminasi.

    Brier = REL - RES + UNC (dekomposisi Murphy pada bin kuantil). REL adalah
    komponen kalibrasi, RES komponen diskriminasi. Brier yang turun karena RES
    naik TIDAK membuktikan kalibrasi membaik, sehingga calibration slope
    (ideal 1) dan calibration intercept (ideal 0) dilaporkan berdampingan.
    """
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    logit = np.log(pc / (1 - pc))
    binom = sm.families.Binomial()
    slope = np.asarray(
        sm.GLM(y_true, sm.add_constant(logit), family=binom).fit().params
    )[1]
    satu = np.ones((len(y_true), 1))
    intercept = np.asarray(
        sm.GLM(y_true, satu, family=binom, offset=logit).fit().params
    )[0]
    tepi = np.quantile(pc, np.linspace(0, 1, n_bin + 1))
    idx = np.clip(np.digitize(pc, tepi[1:-1]), 0, n_bin - 1)
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


def fmt_angka(v):
    """Angka utuh diberi pemisah ribuan, pecahan diberi tiga desimal."""
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 1000 else f"{v:.3f}"


print("=" * 60)
print("0. STATISTIK DESKRIPTIF  -> TABLE 1 NASKAH")
print("=" * 60)
NUM_DESK = [c for c in FIT_B if df[c].dtype.name != "object"]
KAT_DESK = [c for c in FIT_B if df[c].dtype.name == "object"]
rows = []
for c in NUM_DESK:
    kol = df[c]
    rows.append(
        dict(
            peubah=c,
            jenis="numerik",
            taraf="",
            n=len(kol),
            persen="",
            ringkasan=(
                f"{fmt_angka(kol.median())} "
                f"({fmt_angka(kol.quantile(0.25))} to {fmt_angka(kol.quantile(0.75))})"
            ),
            gagal_bayar="",
        )
    )
for c in KAT_DESK:
    for lv in sorted(df[c].unique()):
        m = (df[c] == lv).values
        rows.append(
            dict(
                peubah=c,
                jenis="kategorikal",
                taraf=str(lv),
                n=int(m.sum()),
                persen=round(m.sum() / len(df) * 100, 1),
                ringkasan="",
                gagal_bayar=round(float(y[m].mean()), 3),
            )
        )
t0 = pd.DataFrame(rows)
t0.to_csv("tabel1_deskriptif.csv", index=False)
print(t0.to_string(index=False))
print()
print(
    f"  keseluruhan: n={len(df)}, "
    f"gagal bayar={int(y.sum())} ({y.mean() * 100:.1f} persen)"
)

print("=" * 60)
print("1. MODEL A vs MODEL B  (seed utama, ambang 0,5)")
print("=" * 60)
rows = []
for nama_model, make_model in MODEL.items():
    for tag, cols in [("A", FIT_A), ("B", FIT_B)]:
        m = metrik(y, oof(cols, make_model, SEED_UTAMA))
        rows.append(
            dict(Model=nama_model, Fitur=tag, **{k: round(v, 4) for k, v in m.items()})
        )
t1 = pd.DataFrame(rows)
t1.to_csv("hasil_A_vs_B.csv", index=False)
print(t1.to_string(index=False))

print("\n" + "=" * 60)
print("2. RERATA 5 SEED, semua metrik  -> TABEL UTAMA NASKAH")
print("=" * 60)
rows = []
rows_kal = []
for nama_model, make_model in MODEL.items():
    for tag, cols in [("A", FIT_A), ("B", FIT_B)]:
        # Peluang OOF dihitung sekali per seed, lalu dipakai untuk metrik dan
        # untuk bukti kalibrasi, sehingga tidak ada pelatihan ulang.
        P = [oof(cols, make_model, s) for s in SEEDS]
        M = [metrik(y, p) for p in P]
        agg = {k: round(np.mean([x[k] for x in M]), 4) for k in M[0]}
        agg["sd_AUC"] = round(np.std([x["AUC"] for x in M]), 4)
        rows.append(dict(Model=nama_model, Fitur=tag, **agg))
        K = [kalibrasi(y, p) for p in P]
        # REL, RES, UNC bernilai kecil sehingga perlu 5 desimal. Slope, intercept,
        # dan ECE dilaporkan 4 desimal, sama dengan presisi metrik di Table 2,
        # supaya angka di naskah tidak perlu dibulatkan ulang.
        desimal = {"REL": 5, "RES": 5, "UNC": 5, "slope": 4, "intercept": 4, "ECE": 4}
        aggk = {k: round(float(np.mean([x[k] for x in K])), desimal[k]) for k in K[0]}
        rows_kal.append(dict(Model=nama_model, Fitur=tag, **aggk))
t2 = pd.DataFrame(rows)
t2.to_csv("tabel2_rerata5seed.csv", index=False)
print(t2.to_string(index=False))
print("sd AUC antar-seed maksimum:", t2.sd_AUC.max())

print("\n" + "=" * 60)
print("2b. BUKTI KALIBRASI LANGSUNG  (dekomposisi Brier, slope, intercept)")
print("=" * 60)
kal = pd.DataFrame(rows_kal)
kal.to_csv("kalibrasi_detail.csv", index=False)
print(kal.to_string(index=False))
print(
    "\nSelisih B - A. REL turun = kalibrasi membaik, RES naik = diskriminasi membaik:"
)
for nama_model in kal.Model.unique():
    a = kal[(kal.Model == nama_model) & (kal.Fitur == "A")].iloc[0]
    b = kal[(kal.Model == nama_model) & (kal.Fitur == "B")].iloc[0]
    print(
        f"  {nama_model:<20} dREL={b.REL - a.REL:+.5f}  dRES={b.RES - a.RES:+.5f}  "
        f"dECE={b.ECE - a.ECE:+.5f}  slope {a.slope:.3f}->{b.slope:.3f}  "
        f"intercept {a.intercept:+.3f}->{b.intercept:+.3f}"
    )

print("\n" + "=" * 60)
print("3. SAPUAN AMBANG  (Logistic Regression)")
print("=" * 60)
pA = oof(FIT_A, MODEL["Logistic Regression"], SEED_UTAMA)
pB = oof(FIT_B, MODEL["Logistic Regression"], SEED_UTAMA)
rows = []
for t in np.arange(0.20, 0.65, 0.05):
    for tag, p in [("A", pA), ("B", pB)]:
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
        rows.append(
            dict(
                Ambang=round(t, 2),
                Fitur=tag,
                **{k: round(v, 4) for k, v in metrik(y, p, t).items()},
                TP=tp,
                FP=fp,
                FN=fn,
                TN=tn,
            )
        )
sw = pd.DataFrame(rows)
sw.to_csv("sapuan_ambang.csv", index=False)
kolom = ["Ambang", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN"]
print(sw[sw.Fitur == "B"][kolom].to_string(index=False))

print("\n" + "=" * 60)
print("4. SMOTE vs TANPA SMOTE  (SMOTE hanya di lipatan latih)")
print("=" * 60)
rows = []
rows_kal = []
for nama_model, make_model in MODEL.items():
    for smote in [False, True]:
        # peluang out-of-fold dihitung sekali, dipakai untuk metrik dan kalibrasi,
        # sehingga kedua tabel menggambarkan prediksi yang benar-benar sama
        P = [oof(FIT_B, make_model, s, smote=smote) for s in SEEDS[:3]]
        M = [metrik(y, p) for p in P]
        K = [kalibrasi(y, p) for p in P]
        label = "Ya" if smote else "Tidak"
        rows.append(
            dict(
                Model=nama_model,
                SMOTE=label,
                **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
            )
        )
        rows_kal.append(
            dict(
                Model=nama_model,
                SMOTE=label,
                Brier=round(np.mean([x["Brier"] for x in M]), 5),
                **{k: round(np.mean([x[k] for x in K]), 5) for k in K[0]},
            )
        )
sm_df = pd.DataFrame(rows)
sm_df.to_csv("smote_vs_tanpa.csv", index=False)
print(sm_df.to_string(index=False))

# Kalibrasi di bawah SMOTE. Konstanta klip di kalibrasi() tidak diubah, jadi
# catatan tentang kepekaan slope decision tree tetap berlaku sama persis.
print("\n4b. KALIBRASI DI BAWAH SMOTE  (tiga seed yang sama)")
kal_sm = pd.DataFrame(rows_kal)
kal_sm.to_csv("kalibrasi_smote.csv", index=False)
print(kal_sm.to_string(index=False))
for nama_model in MODEL:
    a = kal_sm[(kal_sm.Model == nama_model) & (kal_sm.SMOTE == "Tidak")].iloc[0]
    b = kal_sm[(kal_sm.Model == nama_model) & (kal_sm.SMOTE == "Ya")].iloc[0]
    print(
        f"  {nama_model:20} REL {b.REL - a.REL:+.5f}  ECE {b.ECE - a.ECE:+.5f}  "
        f"|slope-1| {abs(b.slope - 1) - abs(a.slope - 1):+.4f}  "
        f"Brier {b.Brier - a.Brier:+.5f}"
    )

# Perbandingan langsung yang dikutip di naskah: menggeser ambang tanpa SMOTE
# melawan SMOTE pada ambang bawaan. Specificity ikut dilaporkan karena klaim
# replikasi menyangkut keseimbangan sensitivity dan specificity, bukan F1 saja.
print("\n4c. AMBANG 0.30 TANPA SMOTE  vs  AMBANG 0.50 DENGAN SMOTE  (regresi logistik)")
rows_b = []
for label, smote, t in [
    ("Tanpa SMOTE", False, AMBANG_OPERASI),
    ("Dengan SMOTE", True, 0.50),
]:
    M = []
    for s in SEEDS[:3]:
        p = oof(FIT_B, MODEL["Logistic Regression"], s, smote=smote)
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
            Kondisi=label,
            Ambang=t,
            **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
        )
    )
amb_sm = pd.DataFrame(rows_b)
amb_sm.to_csv("ambang_vs_smote.csv", index=False)
print(amb_sm.to_string(index=False))

print("\n" + "=" * 60)
print("5. SAPUAN RASIO ONGKOS  (biaya FN : biaya FP)")
print("=" * 60)
grid = np.arange(0.05, 0.96, 0.01)
rows = []
for rasio in [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]:
    best = {}
    for tag, p in [("A", pA), ("B", pB)]:
        biaya = []
        for t in grid:
            tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
            biaya.append((fn * rasio + fp) / len(y))
        i = int(np.argmin(biaya))
        best[tag] = (round(grid[i], 2), biaya[i])
    rows.append(
        dict(
            rasio=rasio,
            ambang_A=best["A"][0],
            biaya_A=round(best["A"][1], 4),
            ambang_B=best["B"][0],
            biaya_B=round(best["B"][1], 4),
            hemat=round(best["A"][1] - best["B"][1], 4),
        )
    )
co = pd.DataFrame(rows)
co.to_csv("sapuan_rasio_ongkos.csv", index=False)
print(co.to_string(index=False))

print("\n" + "=" * 60)
print(f"6. KEADILAN ANTAR KELOMPOK  (ambang {AMBANG_OPERASI})")
print("=" * 60)
rows = []
for atr in ["jenis_kelamin", "status_perkawinan", "pendidikan"]:
    for g in raw[atr].unique():
        mask = (raw[atr] == g).values
        n = int(mask.sum())
        if n < 30:
            continue
        out = {}
        for tag, p in [("A", pA), ("B", pB)]:
            pr = (p[mask] >= AMBANG_OPERASI).astype(int)
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
                atribut=atr,
                kelompok=g,
                n=n,
                base_rate=round(y[mask].mean(), 3),
                ditandai_A=round(out["A"][0], 3),
                recall_A=round(out["A"][1], 3),
                FPR_A=round(out["A"][2], 3),
                ditandai_B=round(out["B"][0], 3),
                recall_B=round(out["B"][1], 3),
                FPR_B=round(out["B"][2], 3),
            )
        )
fair = pd.DataFrame(rows)
fair.to_csv("keadilan_kelompok.csv", index=False)
print(fair.to_string(index=False))
print("\nRENTANG antar kelompok (maks - min), makin kecil makin merata:")
for atr in fair.atribut.unique():
    s = fair[fair.atribut == atr]
    for met in ["ditandai", "recall", "FPR"]:
        ra = s[f"{met}_A"].max() - s[f"{met}_A"].min()
        rb = s[f"{met}_B"].max() - s[f"{met}_B"].min()
        print(f"  {atr:<18} {met:<10} A={ra:.3f}  B={rb:.3f}  selisih {rb - ra:+.3f}")

print("\n" + "=" * 60)
print("7. VALIDASI SATU-KOTA-DITINGGALKAN  (kota dibuang dari fitur)")
print("=" * 60)
A_nk = [c for c in FIT_A if c != "kota"]
B_nk = A_nk + FIT_KOP
rows = []
for nama_model in ["Logistic Regression", "Gradient Boosting"]:
    make_model = MODEL[nama_model]
    for k in sorted(set(kota)):
        te = kota == k
        tr = ~te
        res = {}
        for tag, cols in [("A", A_nk), ("B", B_nk)]:
            X = df[cols]
            pipe = Pipeline([("pra", pra(cols)), ("clf", make_model(SEED_UTAMA))])
            pipe.fit(X[tr], y[tr])
            res[tag] = roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])
        rows.append(
            dict(
                Model=nama_model,
                kota=k,
                n_uji=int(te.sum()),
                AUC_A=round(res["A"], 4),
                AUC_B=round(res["B"], 4),
                selisih=round(res["B"] - res["A"], 4),
            )
        )
vk = pd.DataFrame(rows)
vk.to_csv("validasi_kota.csv", index=False)
print(vk.to_string(index=False))
for nama_model in vk.Model.unique():
    s = vk[vk.Model == nama_model]
    print(
        f"  {nama_model}: rerata AUC A={s.AUC_A.mean():.4f} B={s.AUC_B.mean():.4f} "
        f"selisih {s.selisih.mean():+.4f} | "
        f"B unggul di {(s.selisih > 0).sum()} dari {len(s)} kota"
    )

print("\n" + "=" * 60)
print("8. KOEFISIEN & GAMBAR")
print("=" * 60)
pipe = Pipeline(
    [
        ("pra", pra(FIT_B)),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED_UTAMA)),
    ]
)
pipe.fit(df[FIT_B], y)
nama_fitur = [
    x.replace("cat__", "").replace("num__", "")
    for x in pipe.named_steps["pra"].get_feature_names_out()
]
cf = pipe.named_steps["clf"].coef_[0]
imp = pd.DataFrame(
    {"fitur": nama_fitur, "koefisien": cf.round(4), "abs": np.abs(cf).round(4)}
)
imp["blok"] = [
    "KOPERASI" if any(k in x for k in FIT_KOP) else "KONVENSIONAL" for x in imp.fitur
]
imp.sort_values("abs", ascending=False).to_csv("koefisien_logreg_B.csv", index=False)

fa, ta, _ = roc_curve(y, pA)
fb, tb, _ = roc_curve(y, pB)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fa, ta, "--", lw=1.8, label=f"Model A (AUC {roc_auc_score(y, pA):.4f})")
ax.plot(fb, tb, "-", lw=1.8, label=f"Model B (AUC {roc_auc_score(y, pB):.4f})")
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve, logistic regression")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("gambar_roc.png", dpi=330)

fig, ax = plt.subplots(figsize=(5.5, 5))
for p, lab, st in [(pA, "Model A", "--"), (pB, "Model B", "-")]:
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, st, marker="o", ms=4, lw=1.6, label=lab)
ax.plot([0, 1], [0, 1], ":", color="gray", lw=1, label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed default rate")
ax.set_title("Calibration curve, logistic regression")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("gambar_kalibrasi.png", dpi=330)

for t in (0.50, AMBANG_OPERASI):
    tn, fp, fn, tp = confusion_matrix(y, (pB >= t).astype(int)).ravel()
    recall = tp / (tp + fn)
    presisi = tp / (tp + fp)
    print(
        f"  confusion matrix ambang {t:.2f} -> TP {tp} FP {fp} FN {fn} TN {tn} | "
        f"recall {recall:.4f} presisi {presisi:.4f}"
    )
print("\n" + "=" * 60)
print("9. SENSITIVITAS PERINGKAT KOEFISIEN TERHADAP PENALTI")
print("=" * 60)


def koef_logreg(C):
    """Koefisien regresi logistik Model B pada kekuatan penalti tertentu."""
    pipe_c = Pipeline(
        [
            ("pra", pra(FIT_B)),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED_UTAMA, C=C)),
        ]
    )
    pipe_c.fit(df[FIT_B], y)
    nama = [
        x.replace("cat__", "").replace("num__", "")
        for x in pipe_c.named_steps["pra"].get_feature_names_out()
    ]
    return pd.Series(pipe_c.named_steps["clf"].coef_[0], index=nama)


# LogisticRegression scikit-learn memakai penalti L2 dengan C=1,0 secara default.
# Penalti menyusutkan koefisien, dan penyusutannya TIDAK sama antara peubah
# numerik yang distandarkan dan dummy 0/1 yang tidak diskalakan. Blok ini
# mengukur seberapa jauh besar dan peringkat koefisien bergantung pada penalti.
numerik = set(df[FIT_B].select_dtypes("number").columns)
k_pakai = koef_logreg(1.0)
k_tanpa = koef_logreg(1e6)
sens = pd.DataFrame(
    {
        "koef_C1": k_pakai.round(4),
        "koef_tanpa_penalti": k_tanpa.round(4),
        "susut_persen": ((1 - k_pakai.abs() / k_tanpa.abs()) * 100).round(1),
        "jenis": ["NUMERIK" if f in numerik else "DUMMY" for f in k_pakai.index],
    }
)
sens = sens.sort_values("koef_C1", key=lambda s: s.abs(), ascending=False)
sens.index.name = "fitur"
sens.to_csv("sensitivitas_penalti.csv")
print(sens.head(8).to_string())
print("\n  lima besar DENGAN penalti :", list(k_pakai.abs().nlargest(5).index))
print("  lima besar TANPA penalti  :", list(k_tanpa.abs().nlargest(5).index))

print("\n" + "=" * 60)
print("10. SENSITIVITAS TERHADAP DEFINISI keterlambatan_bayar")
print("=" * 60)
# keterlambatan_bayar dihitung atas 12 bulan terakhir DAN mencakup pinjaman yang
# sedang dinilai, sehingga berpotensi memuat informasi dari jendela luaran.
# Blok ini mengulang perbandingan A lawan B tanpa peubah itu, untuk menguji apakah
# sumbangan blok keanggotaan bergantung pada peubah yang definisinya bermasalah.
A_TANPA = [c for c in FIT_A if c != "keterlambatan_bayar"]
B_TANPA = A_TANPA + FIT_KOP
rows = []
for nama_model, make_model in MODEL.items():
    for tag, cols in [("A", A_TANPA), ("B", B_TANPA)]:
        M = [metrik(y, oof(cols, make_model, s)) for s in SEEDS]
        rows.append(
            dict(
                Model=nama_model,
                Fitur=tag,
                **{k: round(np.mean([x[k] for x in M]), 4) for k in M[0]},
            )
        )
sens_k = pd.DataFrame(rows)
sens_k.to_csv("sensitivitas_keterlambatan.csv", index=False)
print(sens_k.to_string(index=False))
print("\nSelisih AUC B - A, set fitur penuh lawan tanpa keterlambatan_bayar:")
p_penuh = t2.pivot(index="Model", columns="Fitur", values="AUC")
p_tanpa = sens_k.pivot(index="Model", columns="Fitur", values="AUC")
for nama_model in MODEL:
    d_penuh = p_penuh["B"][nama_model] - p_penuh["A"][nama_model]
    d_tanpa = p_tanpa["B"][nama_model] - p_tanpa["A"][nama_model]
    print(
        f"  {nama_model:<20} penuh {d_penuh:+.4f}   tanpa {d_tanpa:+.4f}   "
        f"ubah {d_tanpa - d_penuh:+.4f}"
    )

print("\n" + "=" * 60)
print("11. UJI PERMUTASI BLOK, 200 REPLIKASI")
print("=" * 60)
# Permutasi dilakukan pada BLOK, bukan per kolom. Satu urutan acak dipakai untuk
# seluruh sembilan kolom keanggotaan sekaligus, sehingga struktur ketergantungan
# DI DALAM blok tetap utuh sementara kaitannya dengan anggota, peubah konvensional,
# dan luaran terputus. Permutasi per kolom akan merusak struktur internal itu juga,
# sehingga hipotesis nol yang diuji bukan lagi soal informasi tambahan blok.
# Pembagian lipatan DITAHAN TETAP (seed 42) supaya ragam yang terukur murni berasal
# dari permutasi, bukan campuran dengan ragam pembagian lipatan.
N_PERM = 200
auc_A_tetap = roc_auc_score(y, oof(FIT_A, MODEL["Logistic Regression"], SEED_UTAMA))
auc_B_tetap = roc_auc_score(y, oof(FIT_B, MODEL["Logistic Regression"], SEED_UTAMA))
selisih_teramati = auc_B_tetap - auc_A_tetap
nol = []
for i in range(N_PERM):
    rng_i = np.random.default_rng(1000 + i)
    dd = df.copy()
    urutan = rng_i.permutation(len(dd))
    for c in FIT_KOP:
        dd[c] = dd[c].values[urutan]
    nol.append(
        roc_auc_score(y, oof(FIT_B, MODEL["Logistic Regression"], SEED_UTAMA, data=dd))
        - auc_A_tetap
    )
nol = np.array(nol)
n_ge = int((nol >= selisih_teramati).sum())
p_emp = (n_ge + 1) / (N_PERM + 1)
pd.DataFrame(
    {"replikasi": np.arange(1, N_PERM + 1), "selisih_AUC": nol.round(6)}
).to_csv("uji_permutasi.csv", index=False)
print(f"  selisih teramati (seed {SEED_UTAMA}) = {selisih_teramati:+.4f}")
print(f"  sebaran nol dari {N_PERM} permutasi:")
print(f"    rerata {nol.mean():+.4f}  sd {nol.std(ddof=1):.4f}")
print(f"    min {nol.min():+.4f}  maks {nol.max():+.4f}")
print(
    f"  jarak teramati dari rerata nol = {(selisih_teramati - nol.mean()) / nol.std(ddof=1):.1f} sd"
)
print(f"  permutasi >= teramati = {n_ge} dari {N_PERM},  p empiris = {p_emp:.4f}")

print("\nSELESAI. Seluruh CSV dan 2 gambar tersimpan.")
