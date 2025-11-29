# src/features.py
"""
Preprocessing helpers and feature engineering.
Provides functions to build a preprocessor (imputer + scaler) and to apply it.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

def numeric_columns_from_df(df: pd.DataFrame):
    return [c for c in df.columns if str(c).startswith("X")]

def build_preprocessor(numeric_cols):
    """
    Returns a ColumnTransformer with imputer+scaler for numeric columns.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="drop",
        sparse_threshold=0
    )
    return preprocessor

def fit_preprocessor_on_train(train_df):
    numeric_cols = numeric_columns_from_df(train_df)
    preproc = build_preprocessor(numeric_cols)
    preproc.fit(train_df[numeric_cols])
    return preproc, numeric_cols
