"""
Stage 1 — Data loading & cleaning.

Run standalone to inspect the cleaned dataset:
    python data_prep.py
"""
import numpy as np
import pandas as pd

import config as C


def load_raw() -> pd.DataFrame:
    """Read the 14 relevant columns (A..N) from the source workbook."""
    raw = pd.read_excel(
        C.DATA_FILE, sheet_name=C.SHEET_NAME, header=None,
        usecols=range(14), skiprows=C.FIRST_DATA_ROW, engine="openpyxl",
    )
    raw.columns = C.COLUMN_NAMES
    return raw


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleaning steps:
      1. coerce everything to numeric, drop rows with any missing value
      2. drop exact duplicate rows
      3. apply physical sanity bounds on the four measured properties
      4. remove extreme statistical outliers on outputs (3 x IQR fence)
    """
    n0 = len(df)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    n_after_na = len(df)

    df = df.drop_duplicates()
    n_after_dup = len(df)

    # physical plausibility for Fe550 TMT tensile results
    df = df[
        df.YS.between(300, 800) & df.UTS.between(400, 900)
        & df.Elong.between(5, 40) & df.Ratio.between(1.0, 1.6)
    ]
    n_after_bounds = len(df)

    # mild outlier trim (3*IQR) on each output
    for col in C.OUTPUTS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)]

    df = df.reset_index(drop=True)

    if verbose:
        print("=== Stage 1: Data cleaning ===")
        print(f"  raw rows                : {n0}")
        print(f"  after drop-NA           : {n_after_na}")
        print(f"  after drop-duplicates   : {n_after_dup}")
        print(f"  after physical bounds   : {n_after_bounds}")
        print(f"  final rows              : {len(df)}")
    return df


def get_clean_data(verbose: bool = True) -> pd.DataFrame:
    return clean(load_raw(), verbose=verbose)


if __name__ == "__main__":
    data = get_clean_data()
    print("\nShape:", data.shape)
    print("\nSummary statistics:")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(data.describe().round(3).T)
    out = C.OUTPUT_DIR / "cleaned_data.csv"
    data.to_csv(out, index=False)
    print(f"\nSaved cleaned dataset -> {out}")
