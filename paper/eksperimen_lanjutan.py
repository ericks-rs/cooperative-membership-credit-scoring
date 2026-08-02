"""Eksperimen lanjutan untuk menutup temuan review naskah JISEBI.

Skrip ini TIDAK mengubah dan TIDAK menjalankan ulang eksperimen_koperasi.py.
Helper data dan praproses disalin apa adanya dari skrip itu, lalu Bagian 0
menguji salinan tersebut terhadap CSV yang sudah tersimpan. Kalau salinan
menyimpang walau satu desimal keempat, skrip berhenti. Dengan begitu duplikasi
kode tidak bisa melenceng diam-diam.

Isi:
  0. Penjaga parity terhadap tabel2_rerata5seed.csv
  1. Audit id_anggota, kolom ke-25 yang tidak terpakai
  2. Inferensi statistik untuk selisih AUC Model A lawan Model B
  3. Uji permutasi: penegasan algoritma dan replikasi di keluarga model kedua
  4. Model blok keanggotaan saja (eksploratori)
  5. Bentuk permukaan biaya dan kestabilan ambang optimal
  6. AUC univariat lengkap untuk 23 peubah

Jalankan:  .venv\\Scripts\\python.exe eksperimen_lanjutan.py
Keluaran:  7 berkas CSV + 1 gambar PNG, ditulis ke direktori kerja.
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

matplotlib.use("Agg")

BERKAS = "dataset.csv"
SEED_UTAMA = 42
SEEDS = [42, 1, 7, 13, 99]
AMBANG_OPERASI = 0.30
N_BOOT = 2000  # replikasi bootstrap untuk selang kepercayaan AUC
N_BOOT_AMBANG = 1000  # replikasi bootstrap untuk sebaran ambang optimal
SEED_BOOT = 20260802

raw = pd.read_csv(BERKAS)
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


def oof(cols, make_model, seed, data=None):
    """Peluang out-of-fold dari 5-fold stratified CV, identik dengan skrip utama."""
    data = df if data is None else data
    X = data[cols]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline([("pra", pra(cols, data)), ("clf", make_model(seed))])
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def auc_cepat(y_true, skor):
    """AUC lewat statistik peringkat Mann-Whitney. Seri ditangani dengan midrank.

    Nilainya sama dengan roc_auc_score, tetapi jauh lebih murah ketika dipanggil
    ribuan kali di dalam bootstrap.
    """
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(skor)
    return (r[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def midrank(x):
    """Peringkat tengah untuk nilai seri, dipakai oleh perhitungan DeLong."""
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
    """Uji DeLong untuk dua kurva ROC berkorelasi pada sampel yang sama.

    Mengembalikan AUC keduanya, selisih, galat baku selisih, statistik z, dan
    nilai p dua sisi. Komponen ragam mengikuti bentuk cepat Sun dan Xu.

    KAVEAT yang wajib ikut dilaporkan: uji ini mengandaikan kedua vektor skor
    adalah fungsi tetap yang dievaluasi pada sampel acak. Skor di sini adalah
    peluang out-of-fold dari lima model lipatan, sehingga tiap anggota dinilai
    oleh model yang dilatih pada empat lipatan lain. Skor karena itu tidak
    berasal dari satu model tetap dan saling bergantung lemah lewat data latih
    yang beririsan. Ragam DeLong di bawah struktur ini adalah hampiran.
    """
    pos = y_true == 1
    m = int(pos.sum())
    n = len(y_true) - m
    skor = np.vstack([np.r_[p1[pos], p1[~pos]], np.r_[p2[pos], p2[~pos]]])
    tx = np.vstack([midrank(skor[k, :m]) for k in range(2)])
    ty = np.vstack([midrank(skor[k, m:]) for k in range(2)])
    tz = np.vstack([midrank(skor[k]) for k in range(2)])
    auc = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    beda = auc[0] - auc[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = beda / se if se > 0 else np.nan
    return dict(
        AUC_1=auc[0],
        AUC_2=auc[1],
        selisih=beda,
        se=se,
        z=z,
        p=2 * norm.sf(abs(z)) if se > 0 else np.nan,
    )


def indeks_bootstrap(y_true, n_ulang, seed):
    """Indeks bootstrap berstrata luaran, supaya jumlah gagal bayar tetap.

    Tanpa stratifikasi sebagian ulangan bisa kehilangan seluruh kelas minoritas
    sehingga AUC tidak terdefinisi. Stratifikasi juga membuat ragam yang terukur
    murni berasal dari komposisi anggota di dalam tiap kelas.
    """
    rng = np.random.default_rng(seed)
    i1 = np.flatnonzero(y_true == 1)
    i0 = np.flatnonzero(y_true == 0)
    for _ in range(n_ulang):
        yield np.r_[
            rng.choice(i1, len(i1), replace=True), rng.choice(i0, len(i0), replace=True)
        ]


garis = "=" * 70
print(garis)
print("0. PENJAGA PARITY  (salinan helper diuji terhadap CSV tersimpan)")
print(garis)
try:
    t2 = pd.read_csv("tabel2_rerata5seed.csv")
except FileNotFoundError:
    sys.exit("tabel2_rerata5seed.csv tidak ditemukan. Jalankan eksperimen_koperasi.py dulu.")

OOF = {}  # (nama_model, tag) -> daftar peluang OOF per seed
for nama_model in ["Logistic Regression", "Gradient Boosting"]:
    for tag, cols in [("A", FIT_A), ("B", FIT_B)]:
        OOF[(nama_model, tag)] = [oof(cols, MODEL[nama_model], s) for s in SEEDS]
        auc_baru = round(
            float(np.mean([auc_cepat(y, p) for p in OOF[(nama_model, tag)]])), 4
        )
        auc_lama = float(
            t2[(t2.Model == nama_model) & (t2.Fitur == tag)].iloc[0]["AUC"]
        )
        status = "cocok" if auc_baru == auc_lama else "MELENCENG"
        print(f"  {nama_model:<20} {tag}  AUC tersimpan {auc_lama:.4f}  ulang {auc_baru:.4f}  {status}")
        if auc_baru != auc_lama:
            sys.exit(
                "Parity gagal. Salinan helper tidak lagi identik dengan skrip utama, "
                "sehingga seluruh hasil di bawah tidak sebanding. Skrip dihentikan."
            )
print("  parity lolos, lanjut.")

print("\n" + garis)
print("1. AUDIT id_anggota  (kolom ke-25)")
print(garis)
# Naskah menyatakan berkas punya 25 kolom, sedangkan 23 peubah + 1 luaran = 24.
# Kolom yang menggantung adalah id_anggota. Bagian ini membuktikan kolom itu tidak
# pernah masuk matriks fitur di jalur mana pun, dan sekaligus mengukur apakah ia
# membawa sinyal seandainya bocor.
A_nk = [c for c in FIT_A if c != "kota"]
A_TANPA = [c for c in FIT_A if c != "keterlambatan_bayar"]
SEMUA_JALUR = {
    "Model A": FIT_A,
    "Model B": FIT_B,
    "Model A tanpa kota (validasi kota)": A_nk,
    "Model B tanpa kota (validasi kota)": A_nk + FIT_KOP,
    "Model A tanpa keterlambatan_bayar": A_TANPA,
    "Model B tanpa keterlambatan_bayar": A_TANPA + FIT_KOP,
    "Blok keanggotaan saja": FIT_KOP,
}
baris_audit = []
print(f"  kolom berkas mentah           : {raw.shape[1]}")
print(f"  kolom setelah id dan luaran   : {df.shape[1]}  (= 23 peubah penjelas)")
print(f"  id_anggota ada di df?         : {'id_anggota' in df.columns}")
for nama_jalur, cols in SEMUA_JALUR.items():
    ada_di_daftar = "id_anggota" in cols
    # bukti yang mengikat: nama fitur yang benar-benar keluar dari praproses terlatih
    pipa = pra(cols)
    pipa.fit(df[cols])
    nama_keluar = list(pipa.get_feature_names_out())
    ada_di_matriks = any("id_anggota" in x for x in nama_keluar)
    baris_audit.append(
        dict(
            jalur=nama_jalur,
            n_peubah=len(cols),
            n_kolom_matriks=len(nama_keluar),
            id_di_daftar_fitur=ada_di_daftar,
            id_di_matriks_terlatih=ada_di_matriks,
        )
    )
    print(
        f"  {nama_jalur:<36} {len(cols):2d} peubah -> {len(nama_keluar):2d} kolom | "
        f"id di matriks: {ada_di_matriks}"
    )
id_auc = auc_cepat(y, raw["id_anggota"].values)
korelasi = float(np.corrcoef(raw["id_anggota"].values, y)[0, 1])
print(f"\n  AUC univariat id_anggota      : {id_auc:.4f}  (0.5 = tanpa informasi)")
print(f"  korelasi id_anggota dengan y  : {korelasi:+.4f}")
pd.DataFrame(baris_audit).to_csv("audit_id_anggota.csv", index=False)
if any(b["id_di_matriks_terlatih"] for b in baris_audit):
    sys.exit("KEBOCORAN: id_anggota masuk matriks fitur. Hentikan dan perbaiki.")
print("  vonis: id_anggota tidak pernah masuk matriks fitur di jalur mana pun.")

print("\n" + garis)
print("2. INFERENSI STATISTIK UNTUK SELISIH AUC")
print(garis)
# SD antar-seed yang sudah dilaporkan naskah mengukur ragam PEMBAGIAN LIPATAN.
# Bagian ini mengukur ragam PENGAMBILAN SAMPEL ANGGOTA, yaitu pertanyaan yang
# berbeda. Keduanya dilaporkan berdampingan, bukan saling menggantikan.
#
# KRITERIA GUGUR ditetapkan di muka: Model B disebut meningkatkan diskriminasi
# untuk sebuah algoritma HANYA JIKA selang bootstrap 95 persen dari selisih AUC
# rerata-seed tidak memuat nol DAN uji DeLong pada seed 42 memberi p < 0,05.
# Kalau keduanya tidak sepakat, klaim diturunkan menjadi belum terbukti.
for nama_model in MODEL:
    if (nama_model, "A") not in OOF:
        OOF[(nama_model, "A")] = [oof(FIT_A, MODEL[nama_model], s) for s in SEEDS]
        OOF[(nama_model, "B")] = [oof(FIT_B, MODEL[nama_model], s) for s in SEEDS]

baris_inf = []
for nama_model in MODEL:
    PA, PB = OOF[(nama_model, "A")], OOF[(nama_model, "B")]
    d_seed = np.array([auc_cepat(y, b) - auc_cepat(y, a) for a, b in zip(PA, PB)])
    # bootstrap: satu ulangan anggota dipakai untuk KELIMA seed sekaligus, sehingga
    # yang diselangi adalah selisih rerata-seed, bukan selisih satu seed saja
    boot = []
    for idx in indeks_bootstrap(y, N_BOOT, SEED_BOOT):
        yb = y[idx]
        boot.append(
            np.mean([auc_cepat(yb, b[idx]) - auc_cepat(yb, a[idx]) for a, b in zip(PA, PB)])
        )
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dl = delong(y, PB[0], PA[0])  # seed 42 = SEEDS[0]
    lolos = (lo > 0 or hi < 0) and dl["p"] < 0.05
    baris_inf.append(
        dict(
            Model=nama_model,
            selisih_rerata_seed=round(float(d_seed.mean()), 4),
            sd_antar_seed=round(float(d_seed.std(ddof=0)), 4),
            boot_lo=round(float(lo), 4),
            boot_hi=round(float(hi), 4),
            boot_se=round(float(boot.std(ddof=1)), 4),
            delong_selisih_seed42=round(float(dl["selisih"]), 4),
            delong_se=round(float(dl["se"]), 4),
            delong_z=round(float(dl["z"]), 3),
            delong_p=float(f"{dl['p']:.3e}"),
            lolos_kriteria=lolos,
        )
    )
    print(
        f"  {nama_model:<20} d={d_seed.mean():+.4f}  "
        f"IK95 [{lo:+.4f}, {hi:+.4f}]  DeLong z={dl['z']:.2f} p={dl['p']:.2e}  "
        f"{'LOLOS' if lolos else 'TIDAK LOLOS'}"
    )
inf = pd.DataFrame(baris_inf)
inf.to_csv("inferensi_delta_auc.csv", index=False)

print("\n" + garis)
print("3. UJI PERMUTASI: PENEGASAN ALGORITMA DAN REPLIKASI MODEL KEDUA")
print(garis)
# Naskah sebelumnya menulis "selisih AUC antara dua set fitur" tanpa menyebut
# model. Konfigurasi yang benar-benar dipakai skrip utama adalah regresi logistik
# pada seed 42 dengan pembagian lipatan ditahan tetap. Di bawah, konfigurasi itu
# dinyatakan eksplisit lalu diulang pada Gradient Boosting, keluarga model yang
# berbeda paradigma, memakai rancangan permutasi yang sama persis.
N_PERM = 200
N_PERM_HALUS = 2000  # hanya untuk regresi logistik, mempertajam resolusi p


def uji_permutasi(nama_model, n_perm):
    make_model = MODEL[nama_model]
    auc_A = auc_cepat(y, oof(FIT_A, make_model, SEED_UTAMA))
    auc_B = auc_cepat(y, oof(FIT_B, make_model, SEED_UTAMA))
    teramati = auc_B - auc_A
    nol = np.empty(n_perm)
    for i in range(n_perm):
        rng_i = np.random.default_rng(1000 + i)
        dd = df.copy()
        urutan = rng_i.permutation(len(dd))
        for c in FIT_KOP:
            dd[c] = dd[c].values[urutan]
        nol[i] = auc_cepat(y, oof(FIT_B, make_model, SEED_UTAMA, data=dd)) - auc_A
    n_ge = int((nol >= teramati).sum())
    return dict(
        Model=nama_model,
        n_permutasi=n_perm,
        AUC_A=round(float(auc_A), 4),
        AUC_B=round(float(auc_B), 4),
        selisih_teramati=round(float(teramati), 6),
        nol_rerata=round(float(nol.mean()), 5),
        nol_sd=round(float(nol.std(ddof=1)), 5),
        nol_min=round(float(nol.min()), 5),
        nol_maks=round(float(nol.max()), 5),
        deviasi_sd=round(float((teramati - nol.mean()) / nol.std(ddof=1)), 2),
        n_ge_teramati=n_ge,
        p_empiris=round((n_ge + 1) / (n_perm + 1), 5),
    ), nol


baris_perm = []
sebaran = {}
for nama_model, n_perm in [
    ("Logistic Regression", N_PERM),
    ("Logistic Regression", N_PERM_HALUS),
    ("Gradient Boosting", N_PERM),
]:
    hasil, nol = uji_permutasi(nama_model, n_perm)
    baris_perm.append(hasil)
    sebaran[(nama_model, n_perm)] = nol
    print(
        f"  {nama_model:<20} n={n_perm:5d}  teramati {hasil['selisih_teramati']:+.6f}  "
        f"nol {hasil['nol_rerata']:+.5f} sd {hasil['nol_sd']:.5f}  "
        f"{hasil['deviasi_sd']:.2f} SD  p={hasil['p_empiris']:.5f}"
    )
pd.DataFrame(baris_perm).to_csv("permutasi_dua_model.csv", index=False)

print("\n" + garis)
print("4. MODEL BLOK KEANGGOTAAN SAJA  (EKSPLORATORI)")
print(garis)
# Uji permutasi sudah menutup argumen dimensionalitas. Bagian ini menjawab
# pertanyaan yang berbeda: berapa besar sinyal yang berdiri sendiri di dalam
# sembilan peubah keanggotaan. Statusnya EKSPLORATORI. Perbandingan utama naskah
# tetap Model A lawan Model B, dan angka di bawah tidak boleh dipakai untuk
# menggeser klaim itu ke arah mana pun.
baris_kop = []
for nama_model in MODEL:
    PM = [oof(FIT_KOP, MODEL[nama_model], s) for s in SEEDS]
    auc_M = np.array([auc_cepat(y, p) for p in PM])
    auc_A = np.array([auc_cepat(y, p) for p in OOF[(nama_model, "A")]])
    auc_B = np.array([auc_cepat(y, p) for p in OOF[(nama_model, "B")]])
    boot = []
    for idx in indeks_bootstrap(y, N_BOOT, SEED_BOOT + 1):
        yb = y[idx]
        boot.append(np.mean([auc_cepat(yb, p[idx]) for p in PM]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    baris_kop.append(
        dict(
            Model=nama_model,
            AUC_keanggotaan_saja=round(float(auc_M.mean()), 4),
            sd_antar_seed=round(float(auc_M.std(ddof=0)), 4),
            boot_lo=round(float(lo), 4),
            boot_hi=round(float(hi), 4),
            AUC_A=round(float(auc_A.mean()), 4),
            AUC_B=round(float(auc_B.mean()), 4),
        )
    )
    print(
        f"  {nama_model:<20} keanggotaan saja {auc_M.mean():.4f} "
        f"IK95 [{lo:.4f}, {hi:.4f}] | A {auc_A.mean():.4f} | B {auc_B.mean():.4f}"
    )
pd.DataFrame(baris_kop).to_csv("model_keanggotaan_saja.csv", index=False)

print("\n" + garis)
print("5. BENTUK PERMUKAAN BIAYA DAN KESTABILAN AMBANG OPTIMAL")
print(garis)
# Naskah hanya melaporkan argmin per rasio ongkos. Argmin yang melompat
# (0,48 lalu 0,46 lalu 0,34) adalah gejala permukaan yang datar, bukan gejala
# ambang yang benar-benar berpindah jauh. Bagian ini melaporkan bentuk kurvanya:
# lebar daerah yang biayanya berada dalam 1 persen dari minimum, biaya di ambang
# operasi tetap 0,30, dan sebaran argmin di bawah bootstrap anggota.
pA = OOF[("Logistic Regression", "A")][0]
pB = OOF[("Logistic Regression", "B")][0]
grid = np.arange(0.05, 0.96, 0.01)
RASIO = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
TOLERANSI = 0.01  # "praktis setara" = biaya paling banyak 1 persen di atas minimum


def hitung_fn_fp(y_true, p, ambang):
    """Jumlah negatif palsu dan positif palsu untuk tiap ambang di grid."""
    tandai = p[:, None] >= ambang[None, :]
    pos = (y_true == 1)[:, None]
    fn = (~tandai & pos).sum(axis=0)
    fp = (tandai & ~pos).sum(axis=0)
    return fn.astype(float), fp.astype(float)


fnA, fpA = hitung_fn_fp(y, pA, grid)
fnB, fpB = hitung_fn_fp(y, pB, grid)
i_operasi = int(np.argmin(np.abs(grid - AMBANG_OPERASI)))
baris_biaya = []
kurva = {}
for rasio in RASIO:
    for tag, fn, fp in [("A", fnA, fpA), ("B", fnB, fpB)]:
        biaya = (fn * rasio + fp) / len(y)
        i = int(np.argmin(biaya))
        datar = grid[biaya <= biaya[i] * (1 + TOLERANSI)]
        kurva[(rasio, tag)] = biaya
        baris_biaya.append(
            dict(
                rasio=rasio,
                Fitur=tag,
                ambang_argmin=round(float(grid[i]), 2),
                biaya_min=round(float(biaya[i]), 4),
                datar_dari=round(float(datar.min()), 2),
                datar_sampai=round(float(datar.max()), 2),
                lebar_datar=round(float(datar.max() - datar.min()), 2),
                biaya_di_030=round(float(biaya[i_operasi]), 4),
                kelebihan_biaya_030_persen=round(
                    float((biaya[i_operasi] / biaya[i] - 1) * 100), 2
                ),
                ambang_030_di_daerah_datar=bool(
                    biaya[i_operasi] <= biaya[i] * (1 + TOLERANSI)
                ),
            )
        )
pb = pd.DataFrame(baris_biaya)
pb.to_csv("permukaan_biaya.csv", index=False)
print(pb[pb.Fitur == "B"].to_string(index=False))

# Sebaran argmin di bawah bootstrap anggota. Kalau permukaannya datar, argmin
# akan berpencar lebar walaupun biayanya nyaris tidak berubah.
print("\n5b. SEBARAN AMBANG OPTIMAL DI BAWAH BOOTSTRAP ANGGOTA (Model B)")
argmin_boot = {r: [] for r in RASIO}
for idx in indeks_bootstrap(y, N_BOOT_AMBANG, SEED_BOOT + 2):
    fn_b, fp_b = hitung_fn_fp(y[idx], pB[idx], grid)
    for rasio in RASIO:
        argmin_boot[rasio].append(grid[int(np.argmin((fn_b * rasio + fp_b) / len(idx)))])
baris_stab = []
for rasio in RASIO:
    a = np.array(argmin_boot[rasio])
    q = np.percentile(a, [5, 25, 50, 75, 95])
    baris_stab.append(
        dict(
            rasio=rasio,
            argmin_sampel_penuh=round(
                float(pb[(pb.rasio == rasio) & (pb.Fitur == "B")].iloc[0]["ambang_argmin"]), 2
            ),
            boot_median=round(float(q[2]), 2),
            boot_p5=round(float(q[0]), 2),
            boot_p25=round(float(q[1]), 2),
            boot_p75=round(float(q[3]), 2),
            boot_p95=round(float(q[4]), 2),
            rentang_p5_p95=round(float(q[4] - q[0]), 2),
        )
    )
    print(
        f"  rasio {rasio:<4} argmin sampel {baris_stab[-1]['argmin_sampel_penuh']:.2f}  "
        f"bootstrap median {q[2]:.2f}  p5-p95 [{q[0]:.2f}, {q[4]:.2f}]  "
        f"lebar {q[4] - q[0]:.2f}"
    )
pd.DataFrame(baris_stab).to_csv("stabilitas_argmin.csv", index=False)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
for rasio in [1, 2, 3, 5, 10]:
    b = kurva[(rasio, "B")]
    ax.plot(grid, b / b.min(), lw=1.6, label=f"cost ratio {rasio}:1")
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
ax.set_title("Cost surface, logistic regression Model B")
ax.set_ylim(0.98, 1.6)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("gambar_permukaan_biaya.png", dpi=330)
print("  gambar_permukaan_biaya.png tersimpan")

print("\n" + garis)
print("6. AUC UNIVARIAT, 23 PEUBAH")
print(garis)
# Tiga definisi, dinyatakan tegas supaya bisa dipakai ulang:
#   AUC univariat mentah  : nilai peubah dipakai langsung sebagai skor, nilai
#                           lebih besar dianggap menandakan gagal bayar. Hanya
#                           terdefinisi untuk peubah numerik, sebab peubah
#                           kategorikal tidak punya urutan bawaan.
#   AUC univariat terarah : maks(mentah, 1 - mentah). Peubah diorientasikan ke
#                           arah yang memang menandakan gagal bayar, sehingga
#                           nilainya selalu >= 0,5 dan mengukur kekuatan sinyal
#                           terlepas dari arahnya.
#   AUC model satu peubah : AUC dari peluang out-of-fold regresi logistik yang
#                           dilatih HANYA pada peubah itu, memakai praproses dan
#                           pembagian lipatan yang sama dengan eksperimen utama.
#                           Ini ukuran yang SERAGAM untuk numerik dan kategorikal,
#                           dan tidak memakai luaran untuk mengurutkan taraf.
baris_uni = []
for c in FIT_B:
    kategorikal = df[c].dtype.name == "object"
    if kategorikal:
        mentah = terarah = np.nan
    else:
        mentah = auc_cepat(y, df[c].values.astype(float))
        terarah = max(mentah, 1 - mentah)
    p_satu = np.mean(
        [
            auc_cepat(y, oof([c], MODEL["Logistic Regression"], s))
            for s in SEEDS
        ]
    )
    baris_uni.append(
        dict(
            peubah=c,
            blok="KOPERASI" if c in FIT_KOP else "KONVENSIONAL",
            jenis="kategorikal" if kategorikal else "numerik",
            n_taraf=int(df[c].nunique()) if kategorikal else "",
            auc_mentah=round(float(mentah), 4) if not kategorikal else "",
            auc_terarah=round(float(terarah), 4) if not kategorikal else "",
            auc_model_satu_peubah=round(float(p_satu), 4),
        )
    )
uni = pd.DataFrame(baris_uni).sort_values(
    "auc_model_satu_peubah", ascending=False, kind="mergesort"
)
uni.to_csv("auc_univariat.csv", index=False)
print(uni.to_string(index=False))

print("\n  VERIFIKASI TIGA ANGKA YANG SUDAH DIKUTIP NASKAH:")
for peubah, label, nilai_naskah, kolom in [
    ("simpanan_pokok", "principal savings", 0.5205, "auc_terarah"),
    ("riwayat_pinjaman", "prior-loan history terarah", 0.5040, "auc_terarah"),
    ("riwayat_pinjaman", "prior-loan history mentah", 0.4960, "auc_mentah"),
    ("keterlambatan_bayar", "payment lateness", 0.5443, "auc_terarah"),
]:
    r = uni[uni.peubah == peubah].iloc[0]
    hitung = r[kolom]
    tanda = "cocok" if abs(float(hitung) - nilai_naskah) < 5e-5 else "TIDAK COCOK"
    print(f"    {label:<32} naskah {nilai_naskah:.4f}  hitung {float(hitung):.4f}  {tanda}")

print("\nSELESAI. Tujuh CSV dan satu gambar tersimpan.")
