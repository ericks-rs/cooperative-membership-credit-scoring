# Cooperative membership variables in credit scoring

Data and code for the paper *Cooperative membership variables add predictive
value to early loan default prediction*, submitted to Applied Computer Science.

The paper asks a single question. A savings and credit cooperative observes
things a bank does not, such as how long someone has been a member, what they
hold in the three cooperative savings accounts, whether their installments are
deducted from payroll, and whether they attend the annual member meeting. Do
those records add anything to a default model that already has the conventional
borrower and loan variables?

Everything reported in the paper is produced by the scripts here from the
supplied data file. Nothing was typed in by hand.

## What is in this repository

Four folders. The scripts locate the others relative to their own location, so
they can be run from anywhere.

| Path | What it is |
|---|---|
| `data/dataset.csv` | 2,000 de-identified member records, 25 columns, English names and labels. The file the scripts read |
| `data/dataset_original.csv` | The same records as supplied, with the Indonesian names and labels of the cooperative's information system |
| `data/DATA_DICTIONARY.md` | Every column: English name, original name, unit, and the meaning of each category label |
| `scripts/translate_dataset.py` | Writes `data/dataset.csv` from `data/dataset_original.csv` |
| `scripts/main_experiment.py` | Main experiment |
| `scripts/inference_and_checks.py` | Inference and the additional analyses |
| `scripts/redraw_figures.py` | Redraws the three source figures at 600 dpi |
| `scripts/draw_fig1.py` | Draws Fig. 1, the research design, as PDF, SVG, and 600 dpi PNG |
| `scripts/compose_fig2.py` | Places the ROC and calibration panels side by side for Fig. 2 |
| `results/*.csv` | Every result file the two experiment scripts write |
| `figures/` | Every figure, plus the PDF and SVG of Fig. 1 |
| `requirements.txt` | Pinned versions |

## Reproducing the paper

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\translate_dataset.py
.venv\Scripts\python.exe scripts\main_experiment.py
.venv\Scripts\python.exe scripts\inference_and_checks.py
.venv\Scripts\python.exe scripts\redraw_figures.py
.venv\Scripts\python.exe scripts\draw_fig1.py
.venv\Scripts\python.exe scripts\compose_fig2.py
```

Run the main script first. Before doing anything of its own,
`inference_and_checks.py` reloads `results/mean_over_five_seeds.csv` and recomputes the AUC
for logistic regression and gradient boosting under both feature sets, exiting if
any value differs from the stored one. The second script carries its own copy of
the fitting helpers, and that guard is what keeps the copy from drifting away
from the main script.

The two experiment scripts take about 3.5 and 10.5 minutes respectively on an
ordinary laptop. The three figure scripts finish in seconds.
`draw_fig1.py` reads Times New Roman from the Windows font folder and needs
poppler's `pdftoppm` on the PATH to rasterise the PNG.

This has been checked rather than assumed. The whole repository was copied to an
empty folder, `results/`, `figures/`, and `data/dataset.csv` were deleted, the
six scripts were run in the order above, and each output was compared against
the committed version. All 22 CSV
files matched byte for byte and every PNG matched pixel for pixel. Only the PDF
of Fig. 1 differs between runs, in its embedded creation timestamp.

The versions in `requirements.txt` are pinned, and matching them matters. The
reported numbers come back exactly under those versions, and the figures come
back pixel-identical only under the pinned matplotlib. Robustness to the pinning
was also tested for the numbers themselves. The main comparison was rerun on
scikit-learn 1.6.1 with numpy, pandas, and scipy held fixed, and every AUC and
Brier score in Table 1 reproduced exactly for all five algorithms and both
feature sets.

## Which script produces which table and figure

Result files are in `results/`, figures in `figures/`, scripts in `scripts/`.

| Paper | Result file | Written by |
|---|---|---|
| Table 1 | `mean_over_five_seeds.csv` | `main_experiment.py` |
| Table 2 | `auc_difference_inference.csv` | `inference_and_checks.py` |
| Table 3 | `calibration_detail.csv` | `main_experiment.py` |
| Table 4 | `threshold_sweep.csv` | `main_experiment.py` |
| Table 5 | `smote_vs_none.csv` | `main_experiment.py` |
| Table 6 | `group_differences.csv` | `main_experiment.py` |
| Fig. 1 | `fig1_research_design.png` | `draw_fig1.py` |
| Fig. 2 | `fig2_roc_calibration.png` | `compose_fig2.py` |

Table 6 in the paper prints the rates in `group_differences.csv` as percentages
with one decimal, so 0.298 in the file appears as 29.8 in the table.

The remaining CSV files support statements made in the text rather than a
numbered table. They cover the permutation test, the sensitivity analysis without
payment lateness, leave-one-city-out validation, the membership-only model,
univariate AUC per variable, the logistic regression coefficients, the cost
surface, and the stability of the cost-minimising threshold.

## The data

One row is one member with one loan under evaluation. The file has 2,000 rows,
23 explanatory variables, the outcome `default`, and `member_id`, which is
excluded from every feature matrix. There are no missing values and no
duplicate rows.

`data/dataset_original.csv` is the file as supplied, with Indonesian column
names and category labels that match the cooperative's own information system.
`data/dataset.csv` is the same file with English names and labels, written by
`scripts/translate_dataset.py`, and is what the analysis scripts read.
`data/DATA_DICTIONARY.md` lists every column with both names, the unit, and the
meaning of each label.

Translating the labels does not change any result. The one-hot encoder in the
scripts is given the category order of the original Indonesian labels, so the
reference level and the column layout of every model are the same as in a run
on the original file. This was checked by running the whole pipeline on both
files and comparing every result cell.

The records come from a savings and credit cooperative in Indonesia, supplied
through a provider of cooperative financial system applications. The provider
removed names, identity numbers, and contact information before releasing them.
The provider stated that the 2,000 records were drawn at random from a
substantially larger source population, but neither the size nor the defining
criteria of that population were disclosed, and the draw cannot be verified from
this file.

Two definitions matter for anyone reusing the data.

`default` is 1 when arrears reached 90 days or more within the 24-month
performance window following the application. It is not a record of eventual
repayment. Of the 2,000 loans, 1,001 had scheduled terms longer than 24 months,
and the file carries no settlement dates, so a 0 means only that the member had
not reached 90 days of arrears by month 24.

`payment_lateness_months` counts the months, out of the 12 before the data were
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
