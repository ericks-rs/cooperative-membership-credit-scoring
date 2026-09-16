"""Writes data/dataset.csv, the English version of the cooperative's extract.

data/dataset_original.csv is the file as supplied, with Indonesian column names
and category labels that match the cooperative's own information system. This
script renames the columns and translates the labels using the mapping in
data/DATA_DICTIONARY.md. Rows stay in the same order and numeric values are
untouched, so the two files describe exactly the same 2,000 members.

The analysis scripts read data/dataset.csv. To keep their results identical to
a run on the original labels, they pass CATEGORY_ORDER to the one-hot encoder,
which fixes each variable's category order to the alphabetical order of the
original Indonesian labels. Without that, the encoder would sort the English
labels instead, drop a different reference level, and change the fitted models.

Run from anywhere:

    python scripts/translate_dataset.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

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


def category_order(original: pd.DataFrame) -> dict:
    """English labels of every categorical column, in the alphabetical order of
    the original Indonesian labels. This is the order the analysis scripts pin."""
    order = {}
    for col in original.columns:
        if original[col].dtype == object:
            mapping = LABELS.get(col, {})
            order[COLUMNS[col]] = [mapping.get(v, v) for v in sorted(original[col].unique())]
    return order


def main():
    original = pd.read_csv(DATA / "dataset_original.csv")
    missing = [c for c in original.columns if c not in COLUMNS]
    if missing:
        raise SystemExit(f"columns without an English name: {missing}")
    df = original.copy()
    for col, mapping in LABELS.items():
        unknown = sorted(set(df[col]) - set(mapping))
        if unknown:
            raise SystemExit(f"{col}: labels without a translation: {unknown}")
        df[col] = df[col].map(mapping)
    df = df.rename(columns=COLUMNS)
    df.to_csv(DATA / "dataset.csv", index=False)
    print(f"data/dataset.csv written, {df.shape[0]} rows, {df.shape[1]} columns")
    print("CATEGORY_ORDER = {")
    for col, levels in category_order(original).items():
        print(f"    {col!r}: {levels!r},")
    print("}")


if __name__ == "__main__":
    main()
