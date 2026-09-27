"""Cleaning steps for the credit card default dataset.

Each step is a small function so it can be explained in the report's
Data Cleaning/Preparation section and tested on its own.
"""

import pandas as pd

TARGET = "DEFAULT"

PAY_STATUS_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Month labels for the six history columns (1 = most recent).
MONTHS = {1: "Sep", 2: "Aug", 3: "Jul", 4: "Jun", 5: "May", 6: "Apr"}


def rename_columns(df):
    """Standardize column names.

    - PAY_0 -> PAY_1 so repayment status lines up with BILL_AMT1/PAY_AMT1.
    - The long target name -> DEFAULT.
    """
    return df.rename(
        columns={"PAY_0": "PAY_1", "default payment next month": TARGET}
    )


def drop_id(df):
    """ID is just a row number and carries no information."""
    return df.drop(columns=["ID"], errors="ignore")


def drop_duplicates(df):
    """Drop rows that are exact duplicates once ID is removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows")
    return df


def fix_undocumented_codes(df):
    """Fold undocumented category codes into the documented 'other' level.

    Per the UCI documentation:
      EDUCATION: 1=graduate school, 2=university, 3=high school, 4=others
      MARRIAGE:  1=married, 2=single, 3=others
    The data also contains EDUCATION 0, 5, 6 and MARRIAGE 0, which are not
    documented. They are rare, so we map them to 'others'.
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def clean(df):
    """Run the full cleaning pipeline in order."""
    df = rename_columns(df)
    df = drop_id(df)
    df = drop_duplicates(df)
    df = fix_undocumented_codes(df)
    return df


def add_labels(df):
    """Return a copy with readable category labels, for plots and tables."""
    df = df.copy()
    df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"})
    df["EDUCATION"] = df["EDUCATION"].map(
        {1: "Graduate", 2: "University", 3: "High school", 4: "Other"}
    )
    df["MARRIAGE"] = df["MARRIAGE"].map(
        {1: "Married", 2: "Single", 3: "Other"}
    )
    return df
