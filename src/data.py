from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from utils import ensureDir
import pandas as pd
import numpy as np
import os

def loadPrepareData( ):
    ensureDir("data/processed")

    energy = fetch_ucirepo(id=242)
    X = energy.data.features
    y = energy.data.targets["Y1"]  # Heating load

    df = pd.concat([X, y], axis=1)
    df["is_efficient"] = (df["Y1"] <= df["Y1"].median()).astype(int)

    # Split
    Xtrain, Xtemp, ytrain, ytemp = train_test_split(
        X, y, test_size=0.3
    )
    Xval, Xtest, yval, ytest = train_test_split(
        Xtemp, ytemp, test_size=0.5
    )

    dftrain = pd.concat([Xtrain, ytrain], axis=1)
    dfval = pd.concat([Xval, yval], axis=1)
    dftest = pd.concat([Xtest, ytest], axis=1)

    dftrain["is_efficient"] = (dftrain["Y1"] <= df["Y1"].median()).astype(int)
    dfval["is_efficient"] = (dfval["Y1"] <= df["Y1"].median()).astype(int)
    dftest["is_efficient"] = (dftest["Y1"] <= df["Y1"].median()).astype(int)

    dftrain.to_csv("data/processed/train.csv", index=False)
    dfval.to_csv("data/processed/val.csv", index=False)
    dftest.to_csv("data/processed/test.csv", index=False)

    print("Data prepared and saved to data/processed/")
    return dftrain, dfval, dftest

if __name__ == "__main__":
    loadPrepareData()
