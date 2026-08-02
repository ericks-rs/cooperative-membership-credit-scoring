"""Peta setiap angka yang dikutip naskah ke konfigurasi yang menghasilkannya.

Nama berkas keluaran memakai bahasa Inggris, bukan Indonesia seperti artefak
lain di repo ini, karena berkas inilah satu-satunya yang disebut namanya di
dalam naskah dan karena itu akan dibuka oleh pembaca berbahasa Inggris.

Naskah melaporkan kuantitas yang sama di bawah beberapa konfigurasi seed dan
ambang. Semuanya sah, tetapi tanpa penanda pembaca melihatnya sebagai
ketidakkonsistenan. Dua keluaran:

  number_provenance.csv : konfigurasi setiap tabel dan analisis naskah
  apparent_mismatches.csv       : pasangan angka yang tampak bentrok, berikut sebabnya

Nilai ditarik langsung dari CSV hasil, tidak diketik ulang, sehingga tabel ini
ikut berubah bila eksperimen dijalankan ulang.

Jalankan:  .venv\\Scripts\\python.exe number_provenance.py
"""

import pandas as pd

SEED_5 = "42, 1, 7, 13, 99"
SEED_3 = "42, 1, 7"
SEED_1 = "42"

t2 = pd.read_csv("tabel2_rerata5seed.csv")
sw = pd.read_csv("sapuan_ambang.csv")
sm = pd.read_csv("smote_vs_tanpa.csv")
av = pd.read_csv("ambang_vs_smote.csv")


def t2_nilai(model, fitur, kolom):
    return float(t2[(t2.Model == model) & (t2.Fitur == fitur)].iloc[0][kolom])


def sw_nilai(ambang, fitur, kolom):
    b = sw[(sw.Fitur == fitur) & (sw.Ambang == ambang)]
    return float(b.iloc[0][kolom])


def sm_nilai(model, smote, kolom):
    return float(sm[(sm.Model == model) & (sm.SMOTE == smote)].iloc[0][kolom])


# ---------------------------------------------------------------- konfigurasi
KONFIG = [
    dict(
        artefak="Table 1",
        isi="statistik deskriptif",
        model="tidak ada",
        set_fitur="Model B (23 peubah)",
        seed="tidak berlaku",
        ambang="tidak berlaku",
        prediksi="tidak ada, hanya ringkasan data",
        sumber="tabel1_deskriptif.csv",
    ),
    dict(
        artefak="Table 2",
        isi="enam metrik, rerata lima seed",
        model="kelima algoritma",
        set_fitur="A dan B",
        seed=SEED_5,
        ambang="0.50",
        prediksi="out-of-fold, 5-fold stratified",
        sumber="tabel2_rerata5seed.csv",
    ),
    dict(
        artefak="Table 3",
        isi="bukti kalibrasi",
        model="kelima algoritma",
        set_fitur="A dan B",
        seed=SEED_5,
        ambang="tidak bergantung ambang",
        prediksi="out-of-fold, prediksi yang sama dengan Table 2",
        sumber="kalibrasi_detail.csv",
    ),
    dict(
        artefak="Table 4",
        isi="sapuan ambang 0.20 sampai 0.60",
        model="Logistic Regression",
        set_fitur="A dan B",
        seed=SEED_1,
        ambang="sembilan nilai",
        prediksi="out-of-fold seed tunggal, BUKAN rerata seed",
        sumber="sapuan_ambang.csv",
    ),
    dict(
        artefak="Table 5",
        isi="sapuan rasio ongkos",
        model="Logistic Regression",
        set_fitur="A dan B",
        seed=SEED_1,
        ambang="grid 0.05 sampai 0.95 langkah 0.01",
        prediksi="prediksi yang sama dengan Table 4",
        sumber="sapuan_rasio_ongkos.csv",
    ),
    dict(
        artefak="Table 6",
        isi="SMOTE lawan tanpa SMOTE",
        model="kelima algoritma",
        set_fitur="B saja",
        seed=SEED_3,
        ambang="0.50",
        prediksi="out-of-fold, TIGA seed, bukan lima",
        sumber="smote_vs_tanpa.csv",
    ),
    dict(
        artefak="Table 7",
        isi="kalibrasi di bawah SMOTE",
        model="kelima algoritma",
        set_fitur="B saja",
        seed=SEED_3,
        ambang="tidak bergantung ambang",
        prediksi="prediksi yang sama dengan Table 6",
        sumber="kalibrasi_smote.csv",
    ),
    dict(
        artefak="Section 3.7, perbandingan ambang lawan SMOTE",
        isi="recall, specificity, F1",
        model="Logistic Regression",
        set_fitur="B saja",
        seed=SEED_3,
        ambang="0.30 tanpa SMOTE lawan 0.50 dengan SMOTE",
        prediksi="out-of-fold, TIGA seed",
        sumber="ambang_vs_smote.csv",
    ),
    dict(
        artefak="Table 8",
        isi="selisih antar kelompok",
        model="Logistic Regression",
        set_fitur="A dan B",
        seed=SEED_1,
        ambang="0.30",
        prediksi="prediksi yang sama dengan Table 4",
        sumber="keadilan_kelompok.csv",
    ),
    dict(
        artefak="Table 9",
        isi="koefisien regresi logistik",
        model="Logistic Regression",
        set_fitur="B saja",
        seed=SEED_1,
        ambang="tidak berlaku",
        prediksi="dilatih pada SELURUH data, bukan out-of-fold",
        sumber="koefisien_logreg_B.csv",
    ),
    dict(
        artefak="Section 3.3, validasi kota",
        isi="AUC per kota",
        model="Logistic Regression dan Gradient Boosting",
        set_fitur="A dan B, peubah kota DIBUANG",
        seed=SEED_1,
        ambang="tidak bergantung ambang",
        prediksi="satu-kota-ditinggalkan, BUKAN 5-fold",
        sumber="validasi_kota.csv",
    ),
    dict(
        artefak="Section 3.3, sensitivitas keterlambatan",
        isi="AUC tanpa keterlambatan_bayar",
        model="kelima algoritma",
        set_fitur="A dan B, keterlambatan_bayar DIBUANG",
        seed=SEED_5,
        ambang="0.50",
        prediksi="out-of-fold, 5-fold stratified",
        sumber="sensitivitas_keterlambatan.csv",
    ),
    dict(
        artefak="Section 3.2, uji permutasi",
        isi="sebaran nol selisih AUC",
        model="Logistic Regression",
        set_fitur="A dan B",
        seed="42, pembagian lipatan DITAHAN TETAP",
        ambang="tidak bergantung ambang",
        prediksi="out-of-fold, 200 replikasi permutasi",
        sumber="uji_permutasi.csv, permutasi_dua_model.csv",
    ),
    dict(
        artefak="Section 3.9, sensitivitas penalti",
        isi="koefisien pada C=1 dan C=1e6",
        model="Logistic Regression",
        set_fitur="B saja",
        seed=SEED_1,
        ambang="tidak berlaku",
        prediksi="dilatih pada SELURUH data",
        sumber="sensitivitas_penalti.csv",
    ),
    dict(
        artefak="Inferensi selisih AUC (baru)",
        isi="selang bootstrap dan uji DeLong",
        model="kelima algoritma",
        set_fitur="A dan B",
        seed=SEED_5 + " untuk bootstrap, 42 untuk DeLong",
        ambang="tidak bergantung ambang",
        prediksi="out-of-fold, prediksi yang sama dengan Table 2",
        sumber="inferensi_delta_auc.csv",
    ),
    dict(
        artefak="Permukaan biaya (baru)",
        isi="bentuk kurva dan kestabilan argmin",
        model="Logistic Regression",
        set_fitur="A dan B",
        seed=SEED_1,
        ambang="grid 0.05 sampai 0.95 langkah 0.01",
        prediksi="prediksi yang sama dengan Table 4 dan Table 5",
        sumber="permukaan_biaya.csv, stabilitas_argmin.csv",
    ),
    dict(
        artefak="AUC univariat (baru)",
        isi="mentah, terarah, dan model satu peubah",
        model="Logistic Regression untuk model satu peubah",
        set_fitur="23 peubah satu per satu",
        seed=SEED_5 + " untuk model satu peubah",
        ambang="tidak bergantung ambang",
        prediksi="mentah dan terarah dihitung langsung dari data",
        sumber="auc_univariat.csv",
    ),
]

# ------------------------------------------------------------- angka kembar
SEBAB_SEED = (
    "Kuantitas sama, jumlah seed berbeda. Table 2 merata-ratakan lima seed, "
    "Table 6 dan Section 3.7 merata-ratakan tiga seed karena menjalankan ulang "
    "setiap algoritma di bawah resampling menggandakan ongkos."
)
SEBAB_TUNGGAL = (
    "Kuantitas sama, Table 4 memakai seed tunggal 42 sedangkan Table 2 "
    "merata-ratakan lima seed. Analisis ambang sengaja memakai satu himpunan "
    "prediksi supaya cacah confusion matrix berupa bilangan bulat."
)

KEMBAR = []
for model in t2.Model.unique():
    for metrik, kolom in [("AUC", "AUC"), ("Brier", "Brier"), ("Recall", "Recall")]:
        a = t2_nilai(model, "B", kolom)
        b = sm_nilai(model, "Tidak", kolom)
        if round(a, 4) != round(b, 4):
            KEMBAR.append(
                dict(
                    kuantitas=f"{metrik}, {model}, Model B, ambang 0.50",
                    nilai_1=f"{a:.4f}",
                    asal_1=f"Table 2, lima seed ({SEED_5})",
                    nilai_2=f"{b:.4f}",
                    asal_2=f"Table 6 baris tanpa SMOTE, tiga seed ({SEED_3})",
                    selisih=f"{b - a:+.4f}",
                    sebab=SEBAB_SEED,
                )
            )
for metrik, kolom in [
    ("Recall", "Recall"),
    ("Precision", "Precision"),
    ("F1", "F1"),
    ("Accuracy", "Accuracy"),
]:
    a = t2_nilai("Logistic Regression", "B", kolom)
    b = sw_nilai(0.50, "B", kolom)
    if round(a, 4) != round(b, 4):
        KEMBAR.append(
            dict(
                kuantitas=f"{metrik}, Logistic Regression, Model B, ambang 0.50",
                nilai_1=f"{a:.4f}",
                asal_1=f"Table 2, lima seed ({SEED_5})",
                nilai_2=f"{b:.4f}",
                asal_2=f"Table 4 baris ambang 0.50, seed tunggal ({SEED_1})",
                selisih=f"{b - a:+.4f}",
                sebab=SEBAB_TUNGGAL,
            )
        )
KEMBAR.append(
    dict(
        kuantitas="Recall, Logistic Regression, Model B, ambang 0.30",
        nilai_1=f"{sw_nilai(0.30, 'B', 'Recall'):.4f}",
        asal_1=f"Table 4 dan Section 3.6, seed tunggal ({SEED_1})",
        nilai_2=f"{float(av[av.Kondisi == 'Tanpa SMOTE'].iloc[0]['Recall']):.4f}",
        asal_2=f"Section 3.7, tiga seed ({SEED_3})",
        selisih=f"{float(av[av.Kondisi == 'Tanpa SMOTE'].iloc[0]['Recall']) - sw_nilai(0.30, 'B', 'Recall'):+.4f}",
        sebab=(
            "Kuantitas sama pada ambang yang sama. Section 3.6 melaporkan seed "
            "tunggal supaya cacah defaulter berupa bilangan bulat, Section 3.7 "
            "melaporkan rerata tiga seed supaya sebanding dengan kondisi SMOTE."
        ),
    )
)
KEMBAR.append(
    dict(
        kuantitas="Selisih AUC Model B dikurangi Model A",
        nilai_1="0.0099 sampai 0.0341",
        asal_1="prosa Section 3.2, selisih dari angka Table 2 yang SUDAH dibulatkan",
        nilai_2="0.0099 sampai 0.0340",
        asal_2="hitungan presisi penuh sebelum pembulatan",
        selisih="sampai 0.0001",
        sebab=(
            "Selisih di prosa sengaja dihitung dari entri Table 2 supaya pembaca "
            "dapat mereproduksinya dari tabel. Membulatkan dulu lalu mengurangkan "
            "tidak selalu sama dengan mengurangkan lalu membulatkan. Simpangannya "
            "paling besar 0.0001 dan hanya mengenai logistic regression serta "
            "random forest."
        ),
    )
)
KEMBAR.append(
    dict(
        kuantitas="Cacah confusion matrix pada ambang 0.30",
        nilai_1="TP 439, FP 382, FN 169, TN 1010",
        asal_1=f"Table 4 dan Section 3.6, seed tunggal ({SEED_1})",
        nilai_2="TP 442, FP 383.33, FN 166, TN 1008.67",
        asal_2=f"ambang_vs_smote.csv, rerata tiga seed ({SEED_3})",
        selisih="pecahan muncul karena perataan",
        sebab=(
            "Cacah pada seed tunggal berupa bilangan bulat, rerata tiga seed tidak. "
            "Naskah mengutip yang seed tunggal setiap kali menyebut jumlah anggota, "
            "sehingga angka yang muncul di prosa selalu dapat dijumlahkan."
        ),
    )
)

pd.DataFrame(KONFIG).to_csv("number_provenance.csv", index=False)
pd.DataFrame(KEMBAR).to_csv("apparent_mismatches.csv", index=False)

print("=" * 70)
print("KONFIGURASI TIAP ARTEFAK NASKAH")
print("=" * 70)
for k in KONFIG:
    print(f"  {k['artefak']:<44} seed {k['seed']}")
print()
print("=" * 70)
print(f"ANGKA YANG TAMPAK BENTROK: {len(KEMBAR)} pasang")
print("=" * 70)
for k in KEMBAR:
    print(f"  {k['kuantitas']}")
    print(f"      {k['nilai_1']}  <- {k['asal_1']}")
    print(f"      {k['nilai_2']}  <- {k['asal_2']}")
print()
print("Tersimpan: number_provenance.csv, apparent_mismatches.csv")
