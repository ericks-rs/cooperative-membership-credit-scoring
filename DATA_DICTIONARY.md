# Data dictionary for `dataset.csv`

Column names and category labels are in Indonesian, matching the cooperative's
own information system. This file gives the English name used in the paper, the
role of each column in the two feature sets, the unit, and the meaning of every
category label. One row is one member with one loan under evaluation. The file
has 2,000 rows and 25 columns, no missing values, and no duplicate rows.

All amounts are in Indonesian rupiah (IDR). Units for the interest rate,
membership tenure, prior-loan history, and transaction activity were confirmed
with the data provider.

## Identifier and outcome

| Column | English name | Type | Meaning |
|---|---|---|---|
| `id_anggota` | member identifier | integer, 1 to 2000 | Excluded from every feature matrix |
| `status_pinjaman` | default label | 0 or 1 | 1 = arrears of 90 days or more within the 24 months after the application (608 members). 0 = that threshold was not reached by month 24 (1,392 members). A 0 is not a record of eventual repayment. |

## Model A, conventional borrower and loan variables (14)

| Column | English name | Type and unit | Labels or range |
|---|---|---|---|
| `usia` | age | integer, years | 22 to 65 |
| `jenis_kelamin` | gender | category | `P` = female (perempuan), `L` = male (laki-laki) |
| `status_perkawinan` | marital status | category | `Menikah` = married, `Belum Menikah` = single, `Cerai` = divorced |
| `pendidikan` | education | category | `SMA` = senior high school, `D3` = diploma (three-year), `S1` = bachelor's degree, `S2` = master's degree |
| `pekerjaan` | occupation | category | `Karyawan Swasta` = private-sector employee, `UMKM` = micro or small business owner, `Pedagang` = trader, `PNS` = civil servant, `Wiraswasta` = self-employed, `Buruh` = laborer, `Guru` = teacher, `Petani` = farmer, `Nelayan` = fisher, `Perawat` = nurse |
| `kota` | city | category | `Bandung`, `Jakarta`, `Makassar`, `Medan`, `Palembang`, `Semarang`, `Surabaya`, `Yogyakarta` (removed from both feature sets in leave-one-city-out validation) |
| `jumlah_tanggungan` | number of dependents | integer, persons | 0 to 4 |
| `penghasilan_bulanan` | monthly income | integer, IDR | 2,501,000 to 20,897,000 |
| `jumlah_pinjaman` | loan amount | integer, IDR | 2,894,000 to 150,000,000 |
| `tenor_pinjaman` | loan term | integer, months | 12, 24, 36, 48, or 60 |
| `suku_bunga` | interest rate | integer, percent per year | 10 to 15 |
| `rasio_hutang` | debt ratio | decimal | 0.103 to 0.469, as supplied by the provider |
| `riwayat_pinjaman` | prior-loan history | integer, number of previous loans | 0 to 6 |
| `keterlambatan_bayar` | payment lateness | integer, months | Number of months, out of the 12 before the data were extracted, with at least one late payment, including payments on the loan under evaluation. Because the window is anchored at extraction rather than at the application, the variable may carry information from the outcome window. The paper repeats the comparison without it. |

## Membership variables added in Model B (9)

| Column | English name | Type and unit | Labels or range |
|---|---|---|---|
| `lama_keanggotaan` | membership tenure | integer, months | 3 to 179 |
| `simpanan_pokok` | principal savings | integer, IDR | 25,000, 50,000, 75,000, or 100,000 (one-off deposit paid on joining) |
| `simpanan_wajib` | mandatory savings | integer, IDR | 150,000 to 35,800,000 (accumulated balance) |
| `simpanan_sukarela` | voluntary savings | integer, IDR | 100,000 to 13,617,000 (balance) |
| `jenis_anggota` | membership category | category | `Biasa` = full member, `Calon` = prospective member |
| `partisipasi_rat` | participation in the annual member meeting | category | `Ya` = attended, `Tidak` = did not attend (RAT = rapat anggota tahunan, the annual member meeting) |
| `sistem_potong_gaji` | payroll deduction | category | `Ya` = installments deducted from payroll, `Tidak` = not deducted |
| `status_keaktifan` | activity status | category | `Aktif` = active, `Pasif` = inactive |
| `transaksi_koperasi` | transaction activity with the cooperative | integer, IDR | 0 to 45,340,000, value of transactions with the cooperative during the 12 months before extraction |

## Notes for reuse

- The feature lists are hard-coded in `eksperimen_koperasi.py` as `FIT_A` (the 14
  Model A columns) and `FIT_KOP` (the 9 membership columns). Model B is
  `FIT_A + FIT_KOP`.
- Categorical columns are one-hot encoded with one level omitted, and numeric
  columns are standardised, both fitted inside each training fold.
- Group labels in `keadilan_kelompok.csv` use the Indonesian labels above
  (`jenis_kelamin` `P`/`L`, `status_perkawinan`, `pendidikan`). Table 6 in the
  paper reports the same rows with the English labels and as percentages.
- The extract is cross-sectional and carries no calendar dates.
