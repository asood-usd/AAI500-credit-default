# AAI500-credit-default

**Predicting Credit Card Payment Default Using Client Demographics and Payment History**
AAI-500 Probability and Statistics for AI, University of San Diego, Final Team Project

## Team

| Member | Primary responsibilities |
|---|---|
| Abhineet Sood | Repository and codebase setup: data loading, cleaning, feature engineering, modeling and evaluation notebooks |
| Ravi Kanth Vuddagiri | Technical report and video presentation |

Group 02. This was a team effort: we shared ideas, made the main analysis decisions together, and reviewed each other's work.

## Project overview

Lenders lose money when a client stops paying their card balance. This project uses six months of
billing and repayment history, plus basic demographics, to estimate the probability that a client
will default on next month's payment. We also want to know which factors drive that risk.

**Objectives**

1. Clean and prepare the data, documenting every decision.
2. Explore how default relates to credit limit, repayment history, bill and payment amounts, and demographics, using descriptive statistics and hypothesis tests (chi-squared test, z test, t tests and confidence intervals).
3. Fit a logistic regression as the main, interpretable model (odds ratios, p-values, confidence intervals) and compare it against at least one other classifier.
4. Evaluate models with metrics suited to an imbalanced target (ROC-AUC, precision, recall, F1), and pick a decision threshold based on business cost.
5. Turn the results into recommendations a lender could act on.

## Dataset

**Default of Credit Card Clients**, UCI Machine Learning Repository (id 350)
Yeh, I.-C. & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.24432/C55S3H

- 30,000 clients of a bank in Taiwan, April–September 2005
- 23 explanatory variables and 1 binary target (`default payment next month`)
- About 22% of clients defaulted
- No missing values; some undocumented category codes and 35 duplicate rows (handled in cleaning)

| Group | Columns | Description |
|---|---|---|
| Credit limit | `LIMIT_BAL` | Amount of credit given (NT dollars), including family/supplementary credit |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | 1 = male, 2 = female; 1 = grad school, 2 = university, 3 = high school, 4 = other; 1 = married, 2 = single, 3 = other; age in years |
| Repayment status | `PAY_1` … `PAY_6` (raw file: `PAY_0`, `PAY_2`–`PAY_6`) | Sep → Apr 2005. -2 = no balance, -1 = paid in full, 0 = revolving credit, 1–9 = months of payment delay |
| Bill amount | `BILL_AMT1` … `BILL_AMT6` | Statement balance, Sep → Apr 2005 |
| Payment amount | `PAY_AMT1` … `PAY_AMT6` | Amount paid, Sep → Apr 2005 |
| Target | `DEFAULT` (raw: `default payment next month`) | 1 = defaulted, 0 = did not |

The raw file is committed in `data/raw/` (the dataset is licensed CC BY 4.0), so the notebooks run even if the UCI link is down. If the file is missing, `src/data_loader.py` downloads it from UCI the first time a notebook runs. The files in `data/processed/` aren't committed; the notebooks rebuild them.

## Repository structure

```
AAI500-credit-default/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # original UCI .xls (committed)
│   └── processed/            # cleaned CSV written by notebook 01 (git-ignored)
├── notebooks/
│   ├── 01_data_prep_eda.ipynb          # load, quality checks, cleaning, first EDA
│   ├── 02_eda_statistical_tests.ipynb  # chi-squared, z test, t tests with CIs, correlation, VIF
│   ├── 03_modeling.ipynb               # logistic regression (odds ratios, LR test) + comparison models
│   └── 04_evaluation.ipynb             # test results, bootstrap CIs, cost-based cutoff, risk deciles
├── src/
│   ├── data_loader.py        # download + load raw data
│   ├── cleaning.py           # cleaning steps used by the notebooks
│   └── features.py           # engineered features + shared train/test split
└── reports/
    ├── final_report.md       # report text (edit this)
    ├── build_report.py       # builds the report PDF with the notebooks as appendix
    ├── figures/              # saved plots for the report and slides
    └── tables/               # odds ratios, CV and test results as CSV
```

## How to run

```bash
git clone https://github.com/asood-usd/AAI500-credit-default.git
cd AAI500-credit-default
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_prep_eda.ipynb
```

Run the notebooks in order (01 → 04). Notebook 01 loads the raw file and writes `data/processed/credit_default_clean.csv`, which the later notebooks read. Notebook 03 writes the model predictions that notebook 04 evaluates.

To rebuild the report PDF after editing `reports/final_report.md` (needs Google Chrome):

```bash
python reports/build_report.py reports/Final-Project-Report-Team-02.pdf
```

## Data cleaning summary

| Issue | Rows | Action |
|---|---|---|
| `ID` column | all | Dropped (row identifier) |
| Exact duplicate rows (ignoring ID) | 35 | Dropped |
| `EDUCATION` = 0, 5, 6 (undocumented) | 345 | Mapped to 4 ("Other") |
| `MARRIAGE` = 0 (undocumented) | 54 | Mapped to 3 ("Other") |
| `PAY_*` = -2, 0 | many | Kept; valid states (no balance / revolving) |
| `PAY_0` name | n/a | Renamed to `PAY_1` to match `BILL_AMT1`, `PAY_AMT1` |

Cleaned dataset: **29,965 rows × 24 columns**, 22.1% default rate.

## Results summary

| | Logistic regression (our features) | Random forest |
|---|---|---|
| CV ROC-AUC (train, 5-fold) | 0.776 | 0.787 |
| Test ROC-AUC (95% bootstrap CI) | 0.764 (0.751–0.776) | 0.773 (0.761–0.785) |
| Cost-based cutoff (missed default = 5 × false alarm) | 0.16 | 0.19 |
| Test recall / precision at that cutoff | 71% / 38% | 70% / 40% |

Train/test split: 70/30, stratified, seed 500 (20,975 / 8,990 clients).

- Repayment history matters most: being 2+ months late in September multiplies the odds of default by about **6** (95% CI 5.19–6.91); each earlier late month multiplies them by **1.29**.
- Demographics add almost nothing: dropping sex, marital status and age changes test AUC from 0.7637 to 0.7636.
- The random forest is better by about 0.01 AUC (paired bootstrap CI 0.003–0.015), which is real but small. We recommend the logistic regression because it can explain every prediction.
- A cost-based cutoff (0.16) instead of 0.5 doubles the share of defaulters caught (34% → 71%) and lowers expected cost by about 25%.

## Collaboration workflow

- `main` is kept runnable. Work happens on feature branches (e.g. `eda-tests`, `modeling-logreg`).
- Open a pull request and have the other teammate review it before merging.
- Code follows [PEP 8](https://peps.python.org/pep-0008/).
- Clear notebook outputs you don't need before committing, so diffs stay readable.

## Deliverables

- `Final-Project-Report-Team-02.pdf`: technical report (Introduction, Data Cleaning/Preparation, EDA, Model Selection, Model Analysis, Conclusion and Recommendations, plus a notebook-output appendix)
- `Final-Project-Presentation-Team-02.mp4`: 8–10 minute presentation for a non-technical audience

## Use of AI tools

In line with the course's Generative AI policy, we disclose that we used Claude (Anthropic) to help scaffold this repository (folder layout, loader and cleaning helpers, the first notebook), and later to help write and debug the code for notebooks 02–04, the report build script, and to polish the report wording. We reviewed, ran and adjusted all code ourselves, checked the report's numbers against the notebook output, and made the analysis decisions (e.g. how to handle undocumented codes, which features to build, the cost assumption, and which model to recommend). The same disclosure appears in the report.
