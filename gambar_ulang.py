"""Bangun ulang ketiga gambar naskah pada resolusi tinggi.

Skrip ini ada supaya resolusi gambar dapat dinaikkan tanpa menjalankan ulang
seluruh eksperimen, yang memakan waktu jauh lebih lama. Kode penggambarannya
disalin apa adanya dari eksperimen_koperasi.py dan eksperimen_lanjutan.py,
sehingga bentuk gambarnya identik dan hanya kerapatan pikselnya yang berubah.

Peluang out-of-fold yang dipakai dihitung ulang dengan protokol yang sama,
yaitu regresi logistik pada seed 42, jadi kurvanya persis sama dengan gambar
yang dihasilkan skrip utama.

Jalankan:  .venv\\Scripts\\python.exe gambar_ulang.py
Keluaran:  gambar_roc.png, gambar_kalibrasi.png, gambar_permukaan_biaya.png
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

matplotlib.use("Agg")

DPI = 600
SEED_UTAMA = 42
AMBANG_OPERASI = 0.30
TOLERANSI = 0.01

raw = pd.read_csv("dataset.csv")
df = raw.drop(columns=["id_anggota"])
y = df.pop("status_pinjaman").values

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


def pra(cols):
    cat = [c for c in cols if df[c].dtype.name in ("object", "str")]
    num = [c for c in cols if c not in cat]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
            ("num", StandardScaler(), num),
        ]
    )


def oof(cols, seed):
    X = df[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline(
            [
                ("pra", pra(cols)),
                ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
            ]
        )
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


pA = oof(FIT_A, SEED_UTAMA)
pB = oof(FIT_B, SEED_UTAMA)
print(f"AUC Model A {roc_auc_score(y, pA):.4f}, Model B {roc_auc_score(y, pB):.4f}")

# ------------------------------------------------------------------ Fig. 1 ROC
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
fig.savefig("gambar_roc.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------- Fig. 2 kalibrasi
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
fig.savefig("gambar_kalibrasi.png", dpi=DPI)
plt.close(fig)

# ------------------------------------------------------ Fig. 3 permukaan biaya
grid = np.arange(0.05, 0.96, 0.01)
tandai = pB[:, None] >= grid[None, :]
pos = (y == 1)[:, None]
fn = (~tandai & pos).sum(axis=0).astype(float)
fp = (tandai & ~pos).sum(axis=0).astype(float)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
for rasio in [1, 2, 3, 5, 10]:
    biaya = (fn * rasio + fp) / len(y)
    ax.plot(grid, biaya / biaya.min(), lw=1.6, label=f"cost ratio {rasio}:1")
ax.axvline(AMBANG_OPERASI, color="gray", ls=":", lw=1.2)
ax.axhline(1 + TOLERANSI, color="gray", ls="--", lw=0.9)
ax.annotate(
    "operating threshold 0.30",
    xy=(AMBANG_OPERASI, 1.35),
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
fig.savefig("gambar_permukaan_biaya.png", dpi=DPI)
plt.close(fig)

print(f"Tiga gambar tersimpan ulang pada {DPI} DPI.")
