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

    # must contain Y1 and Y2
    if "Y1" not in df.columns or "Y2" not in df.columns:
        raise RuntimeError("CSV missing required Y1 or Y2 columns.")

    before = len(df)
    df = df.dropna(subset=["Y1", "Y2"])
    after = len(df)

    if after < before:
        print(f"[WARN] Dropped {before - after} rows with NaN in Y1/Y2.")

    # NEW: combined target
    df["Y_sum"] = df["Y1"] + df["Y2"]

    return df


def load_prepare_data(random_state=42):
    ensure_dir(DATA_DIR)
    ensure_dir(DATA_DIR / "indices")

    df_all = fetch_dataset()

    # identify features
    feature_cols = [c for c in df_all.columns if str(c).upper().startswith("X")]
    target_col = "Y_sum"   # NEW target

    X = df_all[feature_cols]
    y = df_all[target_col]

    # split (same ratios as before)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, shuffle=True
    )

    # median for classification threshold (NEW because based on Y_sum)
    train_median = float(y_train.median())

    # labels based on combined target
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
        dfp["Y_sum"] = yp

        # restore original Y1 and Y2 values from df_all
        # merge by index
        orig = df_all.loc[yp.index, ["Y1", "Y2"]]
        dfp["Y1"] = orig["Y1"].values
        dfp["Y2"] = orig["Y2"].values

        # classification label based on Y_sum
        dfp["is_efficient"] = (dfp["Y_sum"] <= train_median).astype(int)
        return dfp

    df_train = build_df(X_train, y_train)
    df_val   = build_df(X_val, y_val)
    df_test  = build_df(X_test, y_test)

    # final safety check
    if df_train["Y_sum"].isna().any():
        raise RuntimeError("ERROR: Train Y_sum still contains NaN.")

    # write files
    df_train.to_csv(DATA_DIR / "train.csv", index=False)
    df_val.to_csv(DATA_DIR / "val.csv", index=False)
    df_test.to_csv(DATA_DIR / "test.csv", index=False)

    print("Saved cleaned splits → data/processed/")
    return df_train, df_val, df_test


if __name__ == "__main__":
    load_prepare_data()
