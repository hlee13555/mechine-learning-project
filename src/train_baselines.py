# src/train_baselines.py
"""
Train classical ML baselines for regression (Y1) and classification (is_efficient).
Saves artifacts, logs MLflow runs (Windows-safe).
"""
from pathlib import Path
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, confusion_matrix

from features import build_preprocessor, numeric_columns_from_df
from utils import rmse

sns.set()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# dirs
(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow Windows-safe tracking URI ---
tracking_uri = "file:///" + str(MLRUNS_DIR).replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("energy_efficiency_baselines")
# ---------------------------------------

# load splits
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

numericCols = numeric_columns_from_df(train)
preprocessor = build_preprocessor(numericCols)
preprocessor.fit(train[numericCols])

# Classification models
classification_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, class_weight="balanced")
}

for modelName, estimator in classification_models.items():
    with mlflow.start_run(run_name=f"class_{modelName}"):
        pipe = Pipeline([("pre", preprocessor), ("clf", estimator)])
        pipe.fit(train[numericCols], train["is_efficient"])

        for split_name, df in [("val", val), ("test", test)]:
            preds = pipe.predict(df[numericCols])
            probs = pipe.predict_proba(df[numericCols])[:, 1]
            acc = accuracy_score(df["is_efficient"], preds)
            f1 = f1_score(df["is_efficient"], preds)
            try:
                auc = roc_auc_score(df["is_efficient"], probs)
            except Exception:
                auc = float("nan")

            mlflow.log_metric(f"{split_name}_accuracy", float(acc))
            mlflow.log_metric(f"{split_name}_f1", float(f1))
            mlflow.log_metric(f"{split_name}_roc_auc", float(auc))

            # confusion matrix
            cm = confusion_matrix(df["is_efficient"], preds)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
            ax.set_title(f"{modelName} Confusion ({split_name})")
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
            fig_path = REPORTS_DIR / "figures" / f"{modelName}_confusion_{split_name}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(fig_path), artifact_path="figures")

        # save pipeline
        model_path = MODELS_DIR / f"pipeline_{modelName}.pkl"
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(pipe, f"pipeline_{modelName}")
        mlflow.log_artifact(str(model_path), artifact_path="models")

# Regression models
regression_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5)
}

for modelName, estimator in regression_models.items():
    with mlflow.start_run(run_name=f"reg_{modelName}"):
        pipe = Pipeline([("pre", preprocessor), ("reg", estimator)])
        pipe.fit(train[numericCols], train["Y1"])

        for split_name, df in [("val", val), ("test", test)]:
            preds = pipe.predict(df[numericCols])

            mae = mean_absolute_error(df["Y1"], preds)
            mse = mean_squared_error(df["Y1"], preds)
            rmse_val = float(mse ** 0.5)

            mlflow.log_metric(f"{split_name}_mae", float(mae))
            mlflow.log_metric(f"{split_name}_rmse", float(rmse_val))

            # residuals plot
            residuals = df["Y1"].values - preds
            fig, ax = plt.subplots(figsize=(5,3))
            ax.scatter(preds, residuals, alpha=0.6)
            ax.axhline(0, linestyle="--", color="grey")
            ax.set_xlabel("predicted")
            ax.set_ylabel("residual")
            ax.set_title(f"{modelName} Residuals ({split_name})")
            fig_path = REPORTS_DIR / "figures" / f"{modelName}_residuals_{split_name}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(fig_path), artifact_path="figures")

        model_path = MODELS_DIR / f"pipeline_{modelName}.pkl"
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(pipe, f"pipeline_{modelName}")
        mlflow.log_artifact(str(model_path), artifact_path="models")

# Save target counts
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    counts = df["is_efficient"].value_counts().rename_axis("label").reset_index(name="count")
    counts.to_csv(REPORTS_DIR / "tables" / f"target_counts_{split_name}.csv", index=False)
