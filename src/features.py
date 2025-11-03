import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df, fit_scaler=None):
    numericCols = [c for c in df.columns if c.startswith("X")]
    X = df[numericCols].values
    if fit_scaler is None:
        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(X)
        return Xscaled, scaler
    else:
        Xscaled = fit_scaler.transform(X)
        return Xscaled, fit_scaler
