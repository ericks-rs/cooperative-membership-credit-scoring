"""Writes data/dataset_en.csv, a copy of data/dataset.csv with English column names and
English category labels, for readers who do not read Indonesian.

The analysis scripts read dataset.csv, not this copy. Nothing else changes:
the rows are in the same order, numeric values are untouched, and the mapping
is the one in DATA_DICTIONARY.md. Run from the repository folder:

    python scripts/make_english_copy.py
"""

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

COLUMNS = {
    "id_anggota": "member_id",
    "usia": "age",
    "jenis_kelamin": "gender",
    "status_perkawinan": "marital_status",
    "pendidikan": "education",
    "pekerjaan": "occupation",
    "penghasilan_bulanan": "monthly_income_idr",
    "lama_keanggotaan": "membership_tenure_months",
    "jumlah_pinjaman": "loan_amount_idr",
    "tenor_pinjaman": "loan_term_months",
    "suku_bunga": "interest_rate_pct_per_year",
    "jumlah_tanggungan": "number_of_dependents",
    "riwayat_pinjaman": "prior_loans",
    "keterlambatan_bayar": "payment_lateness_months",
    "rasio_hutang": "debt_ratio",
    "simpanan_wajib": "mandatory_savings_idr",
    "simpanan_sukarela": "voluntary_savings_idr",
    "status_pinjaman": "default",
    "kota": "city",
    "simpanan_pokok": "principal_savings_idr",
    "jenis_anggota": "membership_category",
    "partisipasi_rat": "annual_meeting_participation",
    "transaksi_koperasi": "transaction_activity_12m_idr",
    "sistem_potong_gaji": "payroll_deduction",
    "status_keaktifan": "activity_status",
}

LABELS = {
    "jenis_kelamin": {"P": "female", "L": "male"},
    "status_perkawinan": {"Menikah": "married", "Belum Menikah": "single", "Cerai": "divorced"},
    "pendidikan": {
        "SMA": "senior high school",
        "D3": "diploma",
        "S1": "bachelor's degree",
        "S2": "master's degree",
    },
    "pekerjaan": {
        "Karyawan Swasta": "private-sector employee",
        "UMKM": "micro or small business owner",
        "Pedagang": "trader",
        "PNS": "civil servant",
        "Wiraswasta": "self-employed",
        "Buruh": "laborer",
        "Guru": "teacher",
        "Petani": "farmer",
        "Nelayan": "fisher",
        "Perawat": "nurse",
    },
    "jenis_anggota": {"Biasa": "full member", "Calon": "prospective member"},
    "partisipasi_rat": {"Ya": "yes", "Tidak": "no"},
    "sistem_potong_gaji": {"Ya": "yes", "Tidak": "no"},
    "status_keaktifan": {"Aktif": "active", "Pasif": "inactive"},
}


def main():
    df = pd.read_csv(DATA / "dataset.csv")
    missing = [c for c in df.columns if c not in COLUMNS]
    if missing:
        raise SystemExit(f"columns without an English name: {missing}")
    for col, mapping in LABELS.items():
        unknown = sorted(set(df[col]) - set(mapping))
        if unknown:
            raise SystemExit(f"{col}: labels without a translation: {unknown}")
        df[col] = df[col].map(mapping)
    df = df.rename(columns=COLUMNS)
    df.to_csv(DATA / "dataset_en.csv", index=False)
    print(f"dataset_en.csv written, {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
