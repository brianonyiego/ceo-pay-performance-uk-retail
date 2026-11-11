# CEO Pay and Firm Performance: UK Retail Firms (2015-2024)

Panel regression analysis of the relationship between CEO pay and firm performance (ROA and ROE) for UK-listed retail firms, using WRDS director remuneration and financial data.

---

## Contents

- [Overview](#overview)
- [Research design](#research-design)
- [Data](#data)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Configuration](#configuration)
- [Variables](#variables)
- [Models](#models)
- [Robustness checks](#robustness-checks)
- [Diagnostics and plots](#diagnostics-and-plots)
- [Summary of results](#summary-of-results)
- [Correlation heatmap](#correlation-heatmap)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project tests whether CEO pay is associated with firm performance, both contemporaneously and with a one-year lag. The pipeline:

1. Loads and cleans two sheets from a WRDS Excel workbook.
2. Identifies CEO rows and builds one CEO pay observation per company-year.
3. Merges pay data with firm financials into a firm-year panel.
4. Constructs ROA, ROE, log CEO pay, firm size and leverage.
5. Estimates pooled OLS and company fixed-effects models with robust and clustered standard errors.
6. Runs wild-cluster bootstrap tests, regression diagnostics, and robustness checks.

Final panel: **15 firms, 115 firm-year observations, 2015-2024.**

---

## Research design

| Item | Description |
|------|-------------|
| Dependent variables | ROA, ROE |
| Main regressor | Log CEO total remuneration (current year and lagged one year) |
| Controls | Firm size (log total assets), leverage, year dummies |
| Estimators | Pooled OLS (HC1 and firm-clustered), company fixed effects (LSDV) |
| Inference | Firm-clustered SEs and wild-cluster (Rademacher) bootstrap, B = 4,999 |
| Robustness | Winsorised sample (1%/99%), excluding 2020-2021 |

Four model specifications are estimated for each sample:

| Model | Formula |
|-------|---------|
| `ROA_contemporaneous` | `ROA ~ CEOPay + FirmSize + Leverage + C(Year_cat)` |
| `ROE_contemporaneous` | `ROE ~ CEOPay + FirmSize + Leverage + C(Year_cat)` |
| `ROA_lagged` | `ROA ~ LaggedCEOPay + FirmSize + Leverage + C(Year_cat)` |
| `ROE_lagged` | `ROE ~ LaggedCEOPay + FirmSize + Leverage + C(Year_cat)` |

---

## Data

The data come from WRDS and are **not included** in this repository (licensed data). Place your own copy locally and set the path in the configuration.

| Sheet | Content |
|-------|---------|
| `Sheet1` | Board and director remuneration (one row per director-year, plus board-level aggregate rows) |
| `Sheet2` | Firm-year financials (profit after tax, total assets, equity, total debt, revenue, operating profit) |

Firms in the final panel: AO World, Card Factory, Dunelm, Greggs, Halfords, Inchcape, JD Sports Fashion, Kingfisher, Mulberry, Ocado, Pets at Home, Travis Perkins, Victorian Plumbing, Watches of Switzerland, WH Smith.

### CEO identification

- Only named individual rows are used. Aggregate rows such as `ED Average` or `Board Total` are excluded.
- A row is treated as a CEO if `rolename` matches `CEO_ROLE_PATTERN` (for example "CEO", "Group CEO", "Chief Executive") and does not match `EXCLUDE_ROLE_PATTERN` (for example "Deputy", "Former").
- Where more than one CEO served in a financial year, their remuneration is summed.

### Merging

Company names are normalised (case, punctuation, legal suffixes such as PLC, LTD, GROUP) to create a merge key, and Sheet1 is joined to Sheet2 on company key and year. Unmatched firm-years on either side are printed in the `[data check]` output.

---

## Repository structure

> TODO: adjust file names to match your repository.

```
.
├── README.md
├── requirements.txt
├── data/
│   └── WRDS DATA.xlsx              # not committed (licensed data)
├── ceo_pay_analysis.py             # data pipeline, models, diagnostics, plots
├── correlation_heatmap.py          # terminal and seaborn correlation heatmap
└── correlation_heatmap.png         # generated output
```

---

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `openpyxl`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `colorama`

```bash
pip install numpy pandas openpyxl matplotlib seaborn statsmodels colorama
```

---

## How to run

1. Put the WRDS workbook in the data folder.
2. Set `FILE_PATH` in the configuration block (use a relative path such as `data/WRDS DATA.xlsx`).
3. Run the analysis:

```bash
python ceo_pay_analysis.py
```

4. Run the correlation heatmap:

```bash
python correlation_heatmap.py
```

All results are printed to the terminal. The analysis script writes no files; the heatmap script saves `correlation_heatmap.png`.

---

## Configuration

Settings are at the top of the analysis script.

| Setting | Description | Default |
|---------|-------------|---------|
| `FILE_PATH` | Path to the WRDS workbook | local path (change this) |
| `COMPENSATION_SHEET` | Sheet with director remuneration | `Sheet1` |
| `FINANCIAL_SHEET` | Sheet with firm financials | `Sheet2` |
| `CEO_PAY_COLUMN` | Pay measure used | `totalcompensation` |
| `CEO_PAY_CURRENCY` | Expected pay currency (mismatches are flagged, not converted) | `GBP` |
| `CEO_ROLE_PATTERN` | Regex for CEO role names | `chief executive` / `ceo` |
| `EXCLUDE_ROLE_PATTERN` | Regex for roles to exclude | `deputy` / `former` |
| `RELAX_ROLE_MATCHING` | Also treat "Managing Director" and "President" as CEO-equivalent | `True` |
| `ENABLE_FUZZY_COMPANY_MATCH` | Second-pass fuzzy name matching within year | `True` |
| `FUZZY_MATCH_THRESHOLD` | Minimum difflib similarity for a fuzzy match | `0.87` |
| `WINSOR_LIMITS` | Winsorisation percentiles | `(0.01, 0.01)` |
| `EXCLUDE_YEARS` | Years dropped in the COVID robustness check | `[2020, 2021]` |
| `N_BOOTSTRAP` | Wild-cluster bootstrap replications | `4999` |
| `RANDOM_SEED` | Random seed | `42` |
| `SHOW_PLOTS` | Turn all plotting on or off | `True` |
| `FILL_ZERO_COLS` | Variables whose missing values are filled with 0 | see [Limitations](#limitations) |

---

## Variables

| Variable | Definition |
|----------|------------|
| `ROA` | Profit after tax / total assets |
| `ROE` | Profit after tax / equity |
| `CEOPay` | Natural log of CEO total remuneration |
| `LaggedCEOPay` | `CEOPay` lagged one year within company |
| `FirmSize` | Natural log of total assets |
| `Leverage` | Total debt / total assets |
| `Year_cat` | Year as a string, used for year dummies |

Recomputed ROA, ROE and firm size are cross-checked against the pre-computed values in Sheet2.

---

## Models

For each specification the script reports:

1. **Pooled OLS** with HC1 heteroscedasticity-robust standard errors.
2. **Pooled OLS** with firm-clustered standard errors.
3. **Company fixed effects** (company dummies, LSDV) with firm-clustered standard errors.
4. **Wild-cluster bootstrap** p-value for the pay coefficient, using the restricted (null-imposed) Rademacher bootstrap, which suits a small number of clusters (15 firms).

---

## Robustness checks

| Run | Description |
|-----|-------------|
| Baseline | Full sample (115 firm-years) |
| Winsorised | All key variables winsorised at the 1st and 99th percentiles |
| Excluding 2020-2021 | Drops COVID-19 years (89 firm-years) |

---

## Diagnostics and plots

**Numerical diagnostics** (for each pooled OLS model): variance inflation factors, Breusch-Pagan heteroscedasticity test, Ramsey RESET test, and Cook's distance (count of observations above 4/n).

**Plots** (values are printed to the terminal before each plot):

- Histograms and boxplots of the key variables
- Scatter plots of pay vs performance (ROA/ROE against CEOPay/LaggedCEOPay)
- Year-average trends of ROA, ROE and CEOPay
- Residuals vs fitted, Q-Q plot and Cook's distance for the baseline models

---

## Summary of results

Coefficient on CEO pay from the company fixed-effects models. P-values are shown for firm-clustered SEs and the wild-cluster bootstrap (WCB).

| Sample | Model | Coefficient | Clustered p | WCB p |
|--------|-------|-------------|-------------|-------|
| Baseline | ROA, contemporaneous | 0.0104 | 0.012 | 0.088 |
| Baseline | ROE, contemporaneous | 0.0452 | 0.062 | 0.008 |
| Baseline | ROA, lagged | 0.0036 | 0.278 | 0.538 |
| Baseline | ROE, lagged | -0.0043 | 0.838 | 0.618 |
| Winsorised | ROA, contemporaneous | 0.0106 | 0.007 | 0.039 |
| Winsorised | ROE, contemporaneous | 0.0349 | 0.008 | 0.017 |
| Excl. 2020-21 | ROA, contemporaneous | 0.0051 | 0.216 | 0.118 |
| Excl. 2020-21 | ROE, contemporaneous | 0.0364 | 0.263 | 0.059 |

**Main findings**

- Contemporaneous CEO pay is positively associated with ROA and ROE in the baseline and winsorised samples, but the evidence is weaker with the wild-cluster bootstrap for ROA and disappears for ROA once 2020-2021 are excluded.
- Lagged CEO pay is not significantly related to either performance measure in any sample.
- Simple correlations between pay and performance are weak and negative (see the heatmap), so the positive regression coefficients appear only after controlling for firm size, leverage, year effects and company fixed effects.
- Ramsey RESET rejects the linear specification in some ROE models, and Cook's distance flags influential observations (notably Mulberry in 2023-2024).

These results should be read with the limitations below.

---

## Correlation heatmap

`correlation_heatmap.py` prints a colour-coded correlation matrix in the terminal and saves a seaborn heatmap to `correlation_heatmap.png`.

| Pair | Correlation |
|------|-------------|
| ROA and ROE | 0.793 |
| ROA and FirmSize | -0.533 |
| ROE and FirmSize | -0.425 |
| CEOPay and LaggedCEOPay | 0.480 |
| CEOPay and FirmSize | 0.281 |
| CEOPay and ROA | -0.127 |
| LaggedCEOPay and ROA | -0.279 |

The correlation values are hard-coded in the script. Update them if you change the sample or variable construction.

---

## Limitations

- **Small sample.** 15 firms and 115 firm-years. Clustered standard errors can be unreliable with so few clusters, which is why wild-cluster bootstrap p-values are reported.
- **Missing values filled with zero.** Missing ROA, ROE, CEOPay, LaggedCEOPay, FirmSize and Leverage values are set to 0 instead of dropped. Zero is not a true value for log pay or log assets, so these rows introduce artificial outliers (for example `CEOPay = 0`, `FirmSize = 0`) and can distort coefficients, diagnostics and descriptive statistics. Consider dropping these rows or using a missing-value indicator as a robustness check.
- **Mixed currencies.** All 260 CEO pay rows are reported in USD, but the expected currency is GBP, and no FX conversion is applied. Pay levels are therefore not in pounds. Year dummies absorb common currency movements but not firm-specific effects.
- **Unmatched firm-years.** The merge dropped many firm-years because company names differ between sheets (133 firm-years in Sheet2 and 95 CEO firm-years in Sheet1 in the logged run). Fuzzy matching (`ENABLE_FUZZY_COMPANY_MATCH`) is intended to recover these. Check the `[data check]` output after each run.
- **Role matching.** CEO identification is regex-based. Check the printed `rolename` listing, especially if `RELAX_ROLE_MATCHING` pulls in managing directors or presidents.
- **Extreme values.** ROE is highly skewed and heavy-tailed (Mulberry 2023-2024 in particular), so ROE results are sensitive to outliers.
- **Association, not causation.** The models control for company and year effects but do not address reverse causality or omitted time-varying factors.

---

## Troubleshooting

- **`FileNotFoundError`:** check `FILE_PATH`. Use a relative path or a raw string for Windows paths.
- **Fewer firms than expected:** inspect the `[data check]` messages for dropped firm-years and unmatched company names, and adjust the name normalisation or fuzzy-match threshold.
- **Plots do not appear:** set `SHOW_PLOTS = True` and run in an environment with a display, or use a non-interactive backend and save figures.
- **Slow runtime:** the wild-cluster bootstrap runs 4,999 replications per model. Lower `N_BOOTSTRAP` while testing.
- **Terminal colours look wrong:** the heatmap script uses `colorama`; some terminals do not support background colours.

---

## Data and disclaimer

The WRDS data are licensed and must not be redistributed. This project is for academic research and is not investment or remuneration advice.
