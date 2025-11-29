# src/data.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def fetch_dataset():
    """
    Load local CSV ONLY. Fix empty rows, bad headers, force numeric conversion.
    """
    csv_path = PROJECT_ROOT / "data" / "ENB2012_data.csv"
    if not csv_path.exists():
        raise RuntimeError("Missing data/ENB2012_data.csv")

    # read raw CSV
    df = pd.read_csv(csv_path)

    # drop empty rows
    df = df.dropna(how="all")

    # force numeric on all X and Y columns
    for c in df.columns:
        if str(c).upper().startswith("X") or str(c).upper().startswith("Y"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows missing Y1 (mandatory label)
    if "Y1" not in df.columns:
        raise RuntimeError("CSV missing required Y1 column.")

    before = len(df)
    df = df.dropna(subset=["Y1"])
    after = len(df)

    if after < before:
        print(f"[WARN] Dropped {before - after} rows with NaN in Y1.")

    return df


def load_prepare_data(random_state=42):
    ensure_dir(DATA_DIR)
    ensure_dir(DATA_DIR / "indices")

    df_all = fetch_dataset()

    # identify features
    feature_cols = [c for c in df_all.columns if str(c).upper().startswith("X")]
    target_col = "Y1"

    X = df_all[feature_cols]
    y = df_all[target_col]

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, shuffle=True
    )

    # threshold for classification
    train_median = float(y_train.median())

    # classification labels
    train_lbl = (y_train <= train_median).astype(int)
    temp_lbl = (y_temp <= train_median).astype(int)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_lbl
    )

    def build_df(Xp, yp):
        dfp = Xp.copy()
        dfp["Y1"] = yp
        dfp["is_efficient"] = (dfp["Y1"] <= train_median).astype(int)
        return dfp

    df_train = build_df(X_train, y_train)
    df_val = build_df(X_val, y_val)
    df_test = build_df(X_test, y_test)

    # final safety check
    if df_train["Y1"].isna().any():
        raise RuntimeError("ERROR: Train Y1 still contains NaN after cleaning.")

    # write files
    df_train.to_csv(DATA_DIR / "train.csv", index=False)
    df_val.to_csv(DATA_DIR / "val.csv", index=False)
    df_test.to_csv(DATA_DIR / "test.csv", index=False)

    print("Saved cleaned splits → data/processed/")
    return df_train, df_val, df_test


if __name__ == "__main__":
    load_prepare_data()
