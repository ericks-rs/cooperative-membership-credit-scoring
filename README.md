# Cooperative membership variables in credit scoring

Data and code for the paper *Cooperative membership variables add predictive
value to loan default prediction across algorithms and decision thresholds*,
submitted to Applied Computer Science.

The paper asks a single question. A savings and credit cooperative observes
things a bank does not, such as how long someone has been a member, what they
hold in the three cooperative savings accounts, whether their installments are
deducted from payroll, and whether they attend the annual member meeting. Do
those records add anything to a default model that already has the conventional
borrower and loan variables?

Everything reported in the paper is produced by the scripts here from the
supplied data file. Nothing was typed in by hand.

## What is in this repository

Layout is flat, because the scripts read `dataset.csv` and write their outputs in
the working directory.

| File | What it is |
|---|---|
| `dataset.csv` | 2,000 de-identified member records, 25 columns |
| `eksperimen_koperasi.py` | Main experiment |
| `eksperimen_lanjutan.py` | Inference and the four robustness analyses |
| `gambar_ulang.py` | Redraws the three source figures at 600 dpi |
| `create_fig1_revised.py` | Draws Fig. 1, the research design, as PDF, SVG, and 600 dpi PNG |
| `gambar_dua_panel.py` | Places the ROC and calibration panels side by side for Fig. 2 |
| `*.csv` | Every result file the two experiment scripts write |
| `*.png` | Every figure, plus the PDF and SVG of Fig. 1 |
| `requirements.txt` | Pinned versions |

## Reproducing the paper

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe eksperimen_koperasi.py
.venv\Scripts\python.exe eksperimen_lanjutan.py
.venv\Scripts\python.exe gambar_ulang.py
.venv\Scripts\python.exe create_fig1_revised.py
.venv\Scripts\python.exe gambar_dua_panel.py
```

Run the main script first. Before doing anything of its own,
`eksperimen_lanjutan.py` reloads `tabel2_rerata5seed.csv` and recomputes the AUC
for logistic regression and gradient boosting under both feature sets, exiting if
any value differs from the stored one. The second script carries its own copy of
the fitting helpers, and that guard is what keeps the copy from drifting away
from the main script.

The two experiment scripts take about 3.5 and 10.5 minutes respectively on an
ordinary laptop. The three figure scripts finish in seconds.
`create_fig1_revised.py` reads Times New Roman from the Windows font folder and
needs poppler's `pdftoppm` on the PATH to rasterise the PNG.

This has been checked rather than assumed. The whole repository was copied to an
empty folder, every output was deleted, the five scripts were run in the order
above, and each result file was compared cell by cell against the version used in
the paper. All 22 CSV files matched, and the figure files were identical pixel
for pixel. The two-column Fig. 1 was added later and checked the same way, from
an empty folder, and the PNG matched the one in the manuscript byte for byte.

The versions in `requirements.txt` are pinned, and matching them matters. The
reported numbers come back exactly under those versions, and the figures come
back pixel-identical only under the pinned matplotlib. Robustness to the pinning
was also tested for the numbers themselves. The main comparison was rerun on
scikit-learn 1.6.1 with numpy, pandas, and scipy held fixed, and every AUC and
Brier score in Table 1 reproduced exactly for all five algorithms and both
feature sets.

## Which script produces which table and figure

| Paper | Result file | Written by |
|---|---|---|
| Table 1 | `tabel2_rerata5seed.csv` | `eksperimen_koperasi.py` |
| Table 2 | `inferensi_delta_auc.csv` | `eksperimen_lanjutan.py` |
| Table 3 | `kalibrasi_detail.csv` | `eksperimen_koperasi.py` |
| Table 4 | `sapuan_ambang.csv` | `eksperimen_koperasi.py` |
| Table 5 | `smote_vs_tanpa.csv` | `eksperimen_koperasi.py` |
| Table 6 | `keadilan_kelompok.csv` | `eksperimen_koperasi.py` |
| Fig. 1 | `fig1_alur_penelitian_revised.png` | `create_fig1_revised.py` |
| Fig. 2 | `fig2_roc_kalibrasi.png` | `gambar_dua_panel.py` |

Table 6 in the paper prints the rates in `keadilan_kelompok.csv` as percentages
with one decimal, so 0.298 in the file appears as 29.8 in the table.

The remaining CSV files support statements made in the text rather than a
numbered table. They cover the permutation test, the sensitivity analysis without
payment lateness, leave-one-city-out validation, the membership-only model,
univariate AUC per variable, the logistic regression coefficients, the cost
surface, and the stability of the cost-minimising threshold.

## The data

One row is one member with one loan under evaluation. The file has 2,000 rows,
23 explanatory variables, the outcome `status_pinjaman`, and `id_anggota`, which
is excluded from every feature matrix. There are no missing values and no
duplicate rows.

Column names and category labels are in Indonesian, matching the cooperative's
own information system. The paper gives the English names.

The records come from a savings and credit cooperative in Indonesia, supplied
through a provider of cooperative financial system applications. The provider
removed names, identity numbers, and contact information before releasing them.
The provider stated that the 2,000 records were drawn at random from a
substantially larger source population, but neither the size nor the defining
criteria of that population were disclosed, and the draw cannot be verified from
this file.

Two definitions matter for anyone reusing the data.

`status_pinjaman` is 1 when arrears reached 90 days or more within the 24-month
performance window following the application. It is not a record of eventual
repayment. Of the 2,000 loans, 1,001 had scheduled terms longer than 24 months,
and the file carries no settlement dates, so a 0 means only that the member had
not reached 90 days of arrears by month 24.

`keterlambatan_bayar` counts the months, out of the 12 before the data were
extracted, in which the member was late at least once, including the loan under
evaluation. Because the window is anchored at extraction rather than at the
application, the variable may carry information from the outcome window. The
paper repeats the whole comparison without it.

The extract is cross-sectional and carries no calendar dates, so temporal
validation is not possible with this file.

## Licence

Code is released under the MIT Licence, in `LICENSE`.

The dataset and the derived result files are released under the Creative Commons
Attribution 4.0 International Licence, in `LICENSE-DATA`. Reuse for any purpose
is fine, with attribution given by citing the article.
