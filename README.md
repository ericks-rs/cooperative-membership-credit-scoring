# The Incremental Predictive Value of Cooperative Membership Variables for Loan Default Classification

Reproducibility code for the study on whether cooperative membership variables add
predictive value for loan default classification, evaluated across five algorithms
and nine decision thresholds. Manuscript under review.

The experiment compares two feature sets under one shared protocol:

- **Model A** — 14 conventional loan and demographic variables, available at any
  financial institution.
- **Model B** — Model A plus 9 cooperative membership variables (membership tenure,
  three deposit types, member class, RAT participation, payroll deduction, activity
  status, transaction activity).

Two scripts run the full analysis. `eksperimen_koperasi.py` produces the main
comparison, covering 5-fold stratified cross-validation over five seeds, a threshold
sweep, a cost-ratio sweep, a SMOTE comparison, probability calibration, group-level
fairness, leave-one-city-out robustness, and logistic regression coefficients.
`eksperimen_lanjutan.py` adds the statistical inference and the follow-up analyses. It
begins by reproducing the AUC values of the main table and halts if they do not match,
so the two scripts cannot drift apart unnoticed.

## Repository layout

```
.
├── README.md                  this file
├── requirements.txt           pinned dependencies (exact versions matter)
├── pyproject.toml             black and ruff configuration
└── paper/
    ├── eksperimen_koperasi.py generates every table and figure
    ├── dataset.csv            input data (see "Data availability")
    ├── hasil_A_vs_B.csv       single-seed A vs B (seed 42)
    ├── tabel2_rerata5seed.csv main table, 5-seed mean, six metrics
    ├── kalibrasi_detail.csv   Brier decomposition, calibration slope and intercept
    ├── sapuan_ambang.csv      threshold sweep, 0.20 to 0.60
    ├── smote_vs_tanpa.csv     SMOTE vs no SMOTE
    ├── kalibrasi_smote.csv    calibration under both SMOTE conditions
    ├── ambang_vs_smote.csv    threshold shift vs SMOTE, recall and specificity
    ├── sapuan_rasio_ongkos.csv cost-ratio sweep
    ├── keadilan_kelompok.csv  group fairness at the operating threshold
    ├── koefisien_logreg_B.csv logistic regression coefficients
    ├── sensitivitas_penalti.csv how far those coefficients depend on the L2 penalty
    ├── sensitivitas_keterlambatan.csv main comparison without the payment-lateness variable
    ├── uji_permutasi.csv      null distribution from 200 permutations of the membership block
    ├── validasi_kota.csv      leave-one-city-out robustness
    ├── tabel1_deskriptif.csv  descriptive statistics of the sample
    ├── gambar_roc.png         ROC curves
    ├── gambar_kalibrasi.png   calibration curves
    ├── eksperimen_lanjutan.py inference and follow-up analyses, guarded by a parity check
    ├── audit_id_anggota.csv   proof that the member identifier never enters a feature matrix
    ├── inferensi_delta_auc.csv paired bootstrap and DeLong test for the AUC difference
    ├── permutasi_dua_model.csv permutation test under two algorithm families
    ├── model_keanggotaan_saja.csv exploratory membership-only model
    ├── permukaan_biaya.csv    shape of the cost surface around its minimum
    ├── stabilitas_argmin.csv  bootstrap spread of the cost-minimizing threshold
    ├── auc_univariat.csv      univariate AUC for all 23 variables
    ├── gambar_permukaan_biaya.png cost surface figure
    ├── gambar_ulang.py        rebuilds the three figures at 600 DPI without rerunning the experiments
    ├── number_provenance.py   maps every reported figure to the run that produced it
    ├── number_provenance.csv   configuration behind each table and analysis
    └── apparent_mismatches.csv pairs of figures that look inconsistent, and why they are not
```

## Data availability

The de-identified dataset used in the study is included in this repository as
`paper/dataset.csv`, so the analysis reproduces out of the box. It was obtained from a
savings and credit cooperative in Indonesia through a provider of cooperative
financial-system applications, and de-identified by the provider before analysis, with
no names, identity numbers, or contact information. Neither the cooperative nor the provider is
named. Only the de-identified member-level records are shared here.

The file has 2,000 rows and the following 25 columns (one row per member, no missing
values):

| Column | Type | Notes |
|---|---|---|
| `id_anggota` | id | dropped before modeling |
| `usia` | numeric | age |
| `jenis_kelamin` | categorical | gender |
| `status_perkawinan` | categorical | marital status |
| `pendidikan` | categorical | education level |
| `pekerjaan` | categorical | occupation |
| `penghasilan_bulanan` | numeric | monthly income |
| `lama_keanggotaan` | numeric | membership tenure (cooperative) |
| `jumlah_pinjaman` | numeric | loan amount |
| `tenor_pinjaman` | numeric | loan term |
| `suku_bunga` | numeric | interest rate |
| `jumlah_tanggungan` | numeric | number of dependents |
| `riwayat_pinjaman` | numeric | prior-loan history |
| `keterlambatan_bayar` | numeric | payment lateness |
| `rasio_hutang` | numeric | debt ratio |
| `simpanan_wajib` | numeric | mandatory savings (cooperative) |
| `simpanan_sukarela` | numeric | voluntary savings (cooperative) |
| `simpanan_pokok` | numeric | principal savings (cooperative) |
| `status_pinjaman` | target | 1 = reached 90 days of arrears within 24 months, 0 = did not. A 0 does **not** mean the loan was repaid |
| `kota` | categorical | city |
| `jenis_anggota` | categorical | member class (cooperative) |
| `partisipasi_rat` | categorical | RAT participation (cooperative) |
| `transaksi_koperasi` | numeric | transaction activity with the cooperative, values are multiples of 1,000 and reach 45,340,000, so most likely a monetary total (cooperative) |
| `sistem_potong_gaji` | categorical | payroll deduction (cooperative) |
| `status_keaktifan` | categorical | activity status (cooperative) |

The target `status_pinjaman` marks a member as default when arrears reach 90 days or
more within a 24-month performance window. That window matters. Scheduled loan terms
take five values from 12 to 60 months, and 1,001 of the 2,000 loans run past the end of
the window, so a label of 0 records the absence of 90-day arrears by month 24 rather
than a completed repayment. Treat every result as evidence about early default, not
about eventual repayment. The default rate in the study sample is 30.4 percent, which is
a research-sample rate and not a portfolio-level rate.

## Environment

Python 3.10.9. Dependency versions are pinned in `requirements.txt` because every
verdict in the study rests on numeric thresholds, and a different scikit-learn or numpy
version can shift the fourth decimal for some models. Do not loosen the pins without
rerunning the full analysis.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux or macOS
.venv/bin/python -m pip install -r requirements.txt
```

## Reproduce

```bash
cd paper
python eksperimen_koperasi.py
python eksperimen_lanjutan.py
python number_provenance.py
```

Run them in that order. `eksperimen_lanjutan.py` reads the main table produced by the
first script and refuses to continue if it cannot reproduce it, and
`number_provenance.py` reads the outputs of both. The first script takes a few minutes
on a laptop, with the SVM stage slowest because `probability=True` fits an internal
calibration. The second takes longer, dominated by 2,400 permutation replicates. Every
CSV and PNG is written to the working directory, overwriting the versions shipped
here.

## Output to paper mapping

| File | Paper exhibit |
|---|---|
| `tabel2_rerata5seed.csv` | main results table, six metrics including Brier |
| `kalibrasi_detail.csv` | calibration evidence for both feature sets |
| `sapuan_ambang.csv` | threshold sweep, F1 for both feature sets |
| `smote_vs_tanpa.csv` | SMOTE comparison, discrimination metrics |
| `kalibrasi_smote.csv` | calibration with and without SMOTE |
| `ambang_vs_smote.csv` | threshold shift versus SMOTE, sensitivity and specificity |
| `sapuan_rasio_ongkos.csv` | cost-ratio sweep |
| `keadilan_kelompok.csv` | group differences |
| `koefisien_logreg_B.csv` | logistic regression coefficients |
| `sensitivitas_penalti.csv` | penalty-sensitivity check reported as prose in the results |
| `sensitivitas_keterlambatan.csv` | robustness check excluding the payment-lateness variable |
| `uji_permutasi.csv` | permutation null distribution reported as prose in the results |
| `validasi_kota.csv` | leave-one-city-out robustness |
| `gambar_roc.png` | ROC curve figure |
| `gambar_kalibrasi.png` | calibration curve figure |
| `tabel1_deskriptif.csv` | descriptive statistics table |
| `inferensi_delta_auc.csv` | appendix table, bootstrap interval and DeLong test per algorithm |
| `permukaan_biaya.csv`, `stabilitas_argmin.csv` | appendix table, cost-surface shape and threshold stability |
| `auc_univariat.csv` | appendix table, univariate AUC for all 23 variables |
| `gambar_permukaan_biaya.png` | cost surface figure |
| `permutasi_dua_model.csv` | permutation replication reported as prose in the results |
| `model_keanggotaan_saja.csv` | exploratory membership-only model, reported as prose |
| `audit_id_anggota.csv` | identifier audit reported as prose in the methods |
| `number_provenance.csv`, `apparent_mismatches.csv` | provenance of every reported figure |

## Reproducibility notes

- **Seeds.** Cross-validation uses five seeds (42, 1, 7, 13, 99). Reported numbers are
  the mean across seeds, with the between-seed standard deviation of AUC recorded in
  `tabel2_rerata5seed.csv`.
- **No leakage.** One-hot encoding and standardization live inside the pipeline and are
  fit on the training fold only. SMOTE, when applied, also runs inside the pipeline and
  only on the training fold.
- **Out-of-fold scoring.** Probabilities are collected from held-out folds, so each
  member is scored exactly once per seed.
- **No hyperparameter tuning.** Defaults are used except `max_iter=5000` for logistic
  regression, `max_depth=5` for the decision tree, `n_estimators=300` for the random
  forest, and an RBF kernel with probability estimates for the SVM. The estimand is
  performance under one shared protocol, and both feature sets receive identical
  treatment. Note that the scikit-learn defaults are not neutral. Logistic regression
  applies an L2 penalty with `C=1.0`, and the SVM applies `C=1.0` with `gamma='scale'`,
  so both models are regularized.
- **Coefficients are penalty-dependent.** Because the L2 penalty shrinks unscaled
  one-hot dummies and standardized numeric variables to different degrees, the ranking
  in `koefisien_logreg_B.csv` is not a penalty-independent importance ordering.
  `sensitivitas_penalti.csv` quantifies this by refitting with the penalty effectively
  removed.
- **Operating threshold.** Fairness and the confusion-matrix report use a 0.30 operating
  threshold, chosen where F1 peaks and confirmed independently by the cost-ratio sweep.
  The cost surface is flat near its minimum, so the fixed 0.30 costs under 1 percent more
  than the ratio-specific optimum when a missed default is treated as two to two-and-a-half
  times as costly as a false alarm. Outside roughly two to three times, that penalty grows
  quickly, so the threshold should not be carried to a different cost structure.
- **Inference is separate from seed variation.** The between-seed standard deviation
  describes the fold split, not the member sample. `inferensi_delta_auc.csv` tests the AUC
  difference against sampling noise with a paired bootstrap over members and the DeLong
  test. Four of the five algorithms clear both. The decision tree does not, so its gain is
  a point estimate without inferential support.
- **Calibration is measured separately from discrimination.** The Brier score mixes the
  two, so a lower Brier does not by itself demonstrate better calibration. The script
  reports the Murphy decomposition of the Brier score into reliability, resolution, and
  uncertainty, alongside the calibration slope and the calibration intercept.
- **Style.** The script passes `ruff` and `black` under the configuration in
  `pyproject.toml`.

## Citation

```bibtex
@article{loancreditscoring,
  title   = {The Incremental Predictive Value of Cooperative Membership Variables
             for Loan Default Classification},
  author  = {Indosatrya, Akbar and Swedia, Ericks Rachmat},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

The code is released under the MIT License, reproduced in `LICENSE`.

The data are released under CC BY 4.0, set out in `DATA-LICENSE.md`. You are free to use,
share, and adapt `paper/dataset.csv` for any purpose, including commercially, and you do
not need to ask us first. All we ask in return is a citation, using the BibTeX entry
below.

Two notes. The cooperative and the application provider are not named anywhere in this
repository, so
please leave them unnamed in anything you publish from these data. The records also carry
no names, identity numbers, or contact details, so please do not try to re-identify
individual members.
