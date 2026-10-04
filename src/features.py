"""Feature engineering and the train/test split used by notebooks 03 and 04.

Keeping these in one place means every notebook builds the same features
and the same split, so results line up across notebooks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cleaning import TARGET, PAY_STATUS_COLS

RANDOM_STATE = 500
TEST_SIZE = 0.30

# Plain-English names for the engineered features (used in tables and plots).
FEATURE_LABELS = {
    "LIMIT_10K": "Credit limit (per NTD 10,000)",
    "AGE_10Y": "Age (per 10 years)",
    "FEMALE": "Female (vs. male)",
    "EDU_UNIV": "University (vs. graduate school)",
    "EDU_HS": "High school (vs. graduate school)",
    "EDU_OTHER": "Other education (vs. graduate school)",
    "SINGLE": "Single (vs. married)",
    "MAR_OTHER": "Other marital status (vs. married)",
    "SEP_REVOLVING": "Sep: revolving credit (vs. paid / no balance)",
    "SEP_LATE_1M": "Sep: 1 month late (vs. paid / no balance)",
    "SEP_LATE_2PLUS": "Sep: 2+ months late (vs. paid / no balance)",
    "MONTHS_LATE_PRIOR": "Months late, Apr-Aug (count 0-5)",
    "UTILIZATION": "Sep bill / credit limit",
    "LOG_PAY_AMT1": "log(1 + Sep payment)",
    "LOG_PAY_AMT2": "log(1 + Aug payment)",
}


def build_features(df):
    """Turn the cleaned data into the features used by the logistic regression.

    Choices (explained in notebook 03):
    - Credit limit and age are rescaled so odds ratios are per NT$10,000
      and per 10 years, which are easier to read than per dollar / per year.
    - SEX, EDUCATION and MARRIAGE become 0/1 dummy columns. The reference
      groups are male, graduate school and married.
    - September repayment status (PAY_1) is grouped into four levels:
      paid / no balance (-2, -1), revolving (0), 1 month late, 2+ months late.
      Codes 3-8 are rare, so they are folded into "2+ months late".
    - The five older status columns (PAY_2..PAY_6) are summarized as a count
      of months the client was late, instead of five correlated columns.
    - The six BILL_AMT columns are highly correlated (r = 0.80-0.95), so we
      keep only September's bill, as a share of the credit limit.
    - Payment amounts are very right-skewed, so we use log(1 + amount) for
      the two most recent months.
    """
    X = pd.DataFrame(index=df.index)
    X["LIMIT_10K"] = df["LIMIT_BAL"] / 10_000
    X["AGE_10Y"] = df["AGE"] / 10
    X["FEMALE"] = (df["SEX"] == 2).astype(int)
    X["EDU_UNIV"] = (df["EDUCATION"] == 2).astype(int)
    X["EDU_HS"] = (df["EDUCATION"] == 3).astype(int)
    X["EDU_OTHER"] = (df["EDUCATION"] == 4).astype(int)
    X["SINGLE"] = (df["MARRIAGE"] == 2).astype(int)
    X["MAR_OTHER"] = (df["MARRIAGE"] == 3).astype(int)
    X["SEP_REVOLVING"] = (df["PAY_1"] == 0).astype(int)
    X["SEP_LATE_1M"] = (df["PAY_1"] == 1).astype(int)
    X["SEP_LATE_2PLUS"] = (df["PAY_1"] >= 2).astype(int)
    X["MONTHS_LATE_PRIOR"] = (df[PAY_STATUS_COLS[1:]] >= 1).sum(axis=1)
    # A few clients have negative bills (credits) or bills far above their
    # limit; clipping keeps those rare cases from pulling the coefficient.
    X["UTILIZATION"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 3)
    X["LOG_PAY_AMT1"] = np.log1p(df["PAY_AMT1"])
    X["LOG_PAY_AMT2"] = np.log1p(df["PAY_AMT2"])
    return X


def split(df):
    """Stratified 70/30 train/test split with a fixed seed.

    Returns (train_df, test_df) with all original columns, so each notebook
    can build whatever feature set it needs from the same rows.
    """
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
