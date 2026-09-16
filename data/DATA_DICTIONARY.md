# Data dictionary

`dataset.csv` uses the English column names and labels shown in the first
column of each table below. `dataset_original.csv` is the file as supplied, with
the Indonesian names and labels of the cooperative's information system, shown
in the second column. The two files hold the same 2,000 records in the same
order. This file also gives the role of each column in the two feature sets, the
unit, and the meaning of every category label. One row is one member with one loan under evaluation. The file
has 2,000 rows and 25 columns, no missing values, and no duplicate rows.

All amounts are in Indonesian rupiah (IDR). Units for the interest rate,
membership tenure, prior-loan history, and transaction activity were confirmed
with the data provider.

## Identifier and outcome

| Column in `dataset.csv` | Original name | Type | Meaning |
|---|---|---|---|
| `member_id` | `id_anggota` | integer, 1 to 2000 | Excluded from every feature matrix |
| `default` | `status_pinjaman` | 0 or 1 | 1 = arrears of 90 days or more within the 24 months after the application (608 members). 0 = that threshold was not reached by month 24 (1,392 members). A 0 is not a record of eventual repayment. |

## Model A, conventional borrower and loan variables (14)

| Column in `dataset.csv` | Original name | Type and unit | Labels or range |
|---|---|---|---|
| `age` | `usia` | integer, years | 22 to 65 |
| `gender` | `jenis_kelamin` | category | `female` (original `P`, perempuan), `male` (original `L`, laki-laki) |
| `marital_status` | `status_perkawinan` | category | `married` (`Menikah`), `single` (`Belum Menikah`), `divorced` (`Cerai`) |
| `education` | `pendidikan` | category | `senior high school` (`SMA`), `diploma` (`D3`, three-year), `bachelor's degree` (`S1`), `master's degree` (`S2`) |
| `occupation` | `pekerjaan` | category | `private-sector employee` (`Karyawan Swasta`), `micro or small business owner` (`UMKM`), `trader` (`Pedagang`), `civil servant` (`PNS`), `self-employed` (`Wiraswasta`), `laborer` (`Buruh`), `teacher` (`Guru`), `farmer` (`Petani`), `fisher` (`Nelayan`), `nurse` (`Perawat`) |
| `city` | `kota` | category | `Bandung`, `Jakarta`, `Makassar`, `Medan`, `Palembang`, `Semarang`, `Surabaya`, `Yogyakarta` (removed from both feature sets in leave-one-city-out validation) |
| `number_of_dependents` | `jumlah_tanggungan` | integer, persons | 0 to 4 |
| `monthly_income_idr` | `penghasilan_bulanan` | integer, IDR | 2,501,000 to 20,897,000 |
| `loan_amount_idr` | `jumlah_pinjaman` | integer, IDR | 2,894,000 to 150,000,000 |
| `loan_term_months` | `tenor_pinjaman` | integer, months | 12, 24, 36, 48, or 60 |
| `interest_rate_pct_per_year` | `suku_bunga` | integer, percent per year | 10 to 15 |
| `debt_ratio` | `rasio_hutang` | decimal | 0.103 to 0.469, as supplied by the provider |
| `prior_loans` | `riwayat_pinjaman` | integer, number of previous loans | 0 to 6 |
| `payment_lateness_months` | `keterlambatan_bayar` | integer, months | Number of months, out of the 12 before the data were extracted, with at least one late payment, including payments on the loan under evaluation. Because the window is anchored at extraction rather than at the application, the variable may carry information from the outcome window. The paper repeats the comparison without it. |

## Membership variables added in Model B (9)

| Column in `dataset.csv` | Original name | Type and unit | Labels or range |
|---|---|---|---|
| `membership_tenure_months` | `lama_keanggotaan` | integer, months | 3 to 179 |
| `principal_savings_idr` | `simpanan_pokok` | integer, IDR | 25,000, 50,000, 75,000, or 100,000 (one-off deposit paid on joining) |
| `mandatory_savings_idr` | `simpanan_wajib` | integer, IDR | 150,000 to 35,800,000 (accumulated balance) |
| `voluntary_savings_idr` | `simpanan_sukarela` | integer, IDR | 100,000 to 13,617,000 (balance) |
| `membership_category` | `jenis_anggota` | category | `full member` (`Biasa`), `prospective member` (`Calon`) |
| `annual_meeting_participation` | `partisipasi_rat` | category | `yes` = attended, `no` = did not attend (original `Ya`/`Tidak`; RAT = rapat anggota tahunan, the annual member meeting) |
| `payroll_deduction` | `sistem_potong_gaji` | category | `yes` = installments deducted from payroll, `no` = not deducted (original `Ya`/`Tidak`) |
| `activity_status` | `status_keaktifan` | category | `active` (`Aktif`), `inactive` (`Pasif`) |
| `transaction_activity_12m_idr` | `transaksi_koperasi` | integer, IDR | 0 to 45,340,000, value of transactions with the cooperative during the 12 months before extraction |

## Notes for reuse

- The feature lists are hard-coded in `scripts/main_experiment.py` as
  `FEATURES_A` (the 14 Model A columns) and `FEATURES_MEMBERSHIP` (the 9
  membership columns). Model B is `FEATURES_A + FEATURES_MEMBERSHIP`.
- The one-hot encoder is given `CATEGORY_ORDER`, the category order of the
  original Indonesian labels, so results on `dataset.csv` are identical to
  results on `dataset_original.csv`.
- Categorical columns are one-hot encoded with one level omitted, and numeric
  columns are standardised, both fitted inside each training fold.
- Table 6 in the paper reports the rows of `results/group_differences.csv` as
  percentages with one decimal.
- The extract is cross-sectional and carries no calendar dates.
