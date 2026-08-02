# Data license

The MIT License in `LICENSE` covers the code in this repository. It does not cover the
data. This file states the terms for the data.

## What is covered

`paper/dataset.csv`, the de-identified member-level records of 2,000 cooperative
members, together with the derived result files in `paper/` that are computed from it.

## License

The data are released under the **Creative Commons Attribution 4.0 International
licence (CC BY 4.0)**.

- Human-readable summary: https://creativecommons.org/licenses/by/4.0/
- Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to share and adapt the data for any purpose, including commercially,
provided you give appropriate credit.

## Required attribution

Anyone who uses these data must cite the source publication. Until the article appears,
cite the manuscript and this repository:

> Indosatrya, A., and Swedia, E. R. The Incremental Predictive Value of Cooperative
> Membership Variables for Loan Default Classification: Consistent Gains Across Five
> Algorithms and Nine Decision Thresholds. Manuscript under review. Data and code:
> https://github.com/ericks-rs/cooperative-membership-credit-scoring

A BibTeX entry is given in `README.md`. Replace it with the published reference once the
article carries a DOI.

## Two requests that sit outside the licence

Neither is a copyright term. Both follow from the agreement under which the records were
obtained and from ordinary research ethics, and neither limits what CC BY 4.0 allows you
to do with the data.

**Please leave the source institution unnamed.** The cooperative and the application
provider are not identified anywhere in this repository, under a confidentiality
agreement.

**Please do not attempt re-identification.** The provider removed names, identity
numbers, and contact details before the data reached the authors.

## What the labels mean

`status_pinjaman` records whether a member reached 90 days of arrears within a 24-month
performance window. A value of 0 does not mean the loan was repaid, because 1,001 of the
2,000 loans have a scheduled term that runs past the end of that window. Read the data as
evidence about early default, not about eventual repayment. `README.md` describes every
column.
