# src/train_baselines.py
"""
Train classical ML baselines for regression (Y1) and classification (is_efficient).
Produces model artifacts, per-split metrics, confusion matrices, residuals,
and saves metrics tables to reports/tables/.

This version also saves per-model validation metrics so train_nn.py can compare.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, confusion_matrix
)

from features import build_preprocessor, numeric_columns_from_df
from utils import rmse

sns.set()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow Windows-safe tracking URI
tracking_uri = "file:///" + str(MLRUNS_DIR).replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("energy_efficiency_baselines")

# load splits (must be created by src.data)
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

train["YSum"] = train["Y1"] + train["Y2"]
val["YSum"] = val["Y1"] + val["Y2"]
test["YSum"] = test["Y1"] + test["Y2"]

numericCols = numeric_columns_from_df(train)
preprocessor = build_preprocessor(numericCols)
preprocessor.fit(train[numericCols])

# CLASSIFICATION models (simple defaults)
classification_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=6, class_weight="balanced")
}

# REGRESSION models (simple defaults)
regression_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=6)
}

# Containers to hold validation metrics for model selection (class/reg)
class_val_metrics = {}
reg_val_metrics = {}

# Train classification models
for name, estimator in classification_models.items():
    with mlflow.start_run(run_name=f"class_{name}"):
        pipe = Pipeline([("pre", preprocessor), ("clf", estimator)])
        pipe.fit(train[numericCols], train["is_efficient"])

        # Evaluate on validation
        preds_val = pipe.predict(val[numericCols])
        probs_val = None
        try:
            probs_val = pipe.predict_proba(val[numericCols])[:, 1]
        except Exception:
            # some estimators may not support predict_proba
            probs_val = preds_val

        acc_val = accuracy_score(val["is_efficient"], preds_val)
        f1_val = f1_score(val["is_efficient"], preds_val)
        try:
            auc_val = roc_auc_score(val["is_efficient"], probs_val)
        except Exception:
            auc_val = float("nan")

        class_val_metrics[name] = {"accuracy": acc_val, "f1": f1_val, "auc": auc_val}

        # Evaluate on test too (save artifacts)
        preds_test = pipe.predict(test[numericCols])
        probs_test = None
        try:
            probs_test = pipe.predict_proba(test[numericCols])[:, 1]
        except Exception:
            probs_test = preds_test
        acc_test = accuracy_score(test["is_efficient"], preds_test)
        f1_test = f1_score(test["is_efficient"], preds_test)
        try:
            auc_test = roc_auc_score(test["is_efficient"], probs_test)
        except Exception:
            auc_test = float("nan")

        # Save per-split metrics to CSV (append/update)
        df_metrics = pd.DataFrame([{
            "model": name,
            "split": "val",
            "accuracy": acc_val, "f1": f1_val, "auc": auc_val
        }, {
            "model": name,
            "split": "test",
            "accuracy": acc_test, "f1": f1_test, "auc": auc_test
        }])
        df_metrics.to_csv(REPORTS_DIR / "tables" / f"class_metrics_{name}.csv", index=False)

        # Confusion matrix (val)
        cm = confusion_matrix(val["is_efficient"], preds_val)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False)
        ax.set_title(f"{name} Confusion (val)")
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        figpath = REPORTS_DIR / "figures" / f"{name}_confusion_val.png"
        fig.savefig(figpath, bbox_inches="tight"); plt.close(fig)

        # Save pipeline artifact
        ppath = MODELS_DIR / f"pipeline_{name}.pkl"
        joblib.dump(pipe, ppath)
        mlflow.sklearn.log_model(pipe, name)  # mlflow logs model under 'name'
        mlflow.log_artifact(str(ppath), artifact_path="models")

# Train regression models
for name, estimator in regression_models.items():
    with mlflow.start_run(run_name=f"reg_{name}"):
        pipe = Pipeline([("pre", preprocessor), ("reg", estimator)])
        pipe.fit(train[numericCols], train["YSum"])

        # validation
        preds_val = pipe.predict(val[numericCols])
        mae_val = mean_absolute_error(val["YSum"], preds_val)
        mse_val = mean_squared_error(val["YSum"], preds_val)
        rmse_val = float(mse_val ** 0.5)
        reg_val_metrics[name] = {"mae": mae_val, "rmse": rmse_val}

        # test
        preds_test = pipe.predict(test[numericCols])
        mae_test = mean_absolute_error(test["YSum"], preds_test)
        mse_test = mean_squared_error(test["YSum"], preds_test)
        rmse_test = float(mse_test ** 0.5)

        # Save metrics per model
        df_metrics = pd.DataFrame([{
            "model": name, "split": "val", "mae": mae_val, "rmse": rmse_val
        }, {
            "model": name, "split": "test", "mae": mae_test, "rmse": rmse_test
        }])
        df_metrics.to_csv(REPORTS_DIR / "tables" / f"reg_metrics_{name}.csv", index=False)

        # Residuals plot (val)
        residuals = val["YSum"].values - preds_val
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(preds_val, residuals, alpha=0.6)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set_xlabel("predicted"); ax.set_ylabel("residual")
        ax.set_title(f"{name} Residuals (val)")
        figpath = REPORTS_DIR / "figures" / f"{name}_residuals_val.png"
        fig.savefig(figpath, bbox_inches="tight"); plt.close(fig)

        # Save pipeline
        ppath = MODELS_DIR / f"pipeline_{name}.pkl"
        joblib.dump(pipe, ppath)
        mlflow.sklearn.log_model(pipe, name)
        mlflow.log_artifact(str(ppath), artifact_path="models")

# Save aggregated selection info for later comparison
pd.DataFrame.from_dict(class_val_metrics, orient="index").reset_index().rename(columns={"index":"model"}).to_csv(REPORTS_DIR / "tables/class_val_metrics_summary.csv", index=False)
pd.DataFrame.from_dict(reg_val_metrics, orient="index").reset_index().rename(columns={"index":"model"}).to_csv(REPORTS_DIR / "tables/reg_val_metrics_summary.csv", index=False)

print("[train_baselines] Completed training classical baselines. Artifacts in models/ and reports/")
