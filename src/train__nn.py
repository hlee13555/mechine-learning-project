# src/train_nn.py
"""
Train a multi-head neural network for regression (Y1) and classification (is_efficient).
Saves models, preprocessor, metrics, and figures. Logs MLflow safely on Windows.
"""
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

sns.set()

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix

from utils import rmse

# Project paths
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
mlflow.set_experiment("energy_efficiency_nn")

# Deterministic seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    raise RuntimeError("TensorFlow 2.x is required to run train_nn.py") from e

# load data
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

def numeric_columns_from_df(df):
    return [c for c in df.columns if str(c).startswith("X")]

numeric_cols = numeric_columns_from_df(train)

# preprocessor (fit on train)
preproc = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
preproc.fit(train[numeric_cols])
joblib.dump(preproc, MODELS_DIR / "preprocessor_nn.pkl")

X_train = preproc.transform(train[numeric_cols])
X_val = preproc.transform(val[numeric_cols])
X_test = preproc.transform(test[numeric_cols])

y_train_reg = train["Y1"].values
y_val_reg = val["Y1"].values
y_test_reg = test["Y1"].values

y_train_clf = train["is_efficient"].values
y_val_clf = val["is_efficient"].values
y_test_clf = test["is_efficient"].values

INPUT_DIM = X_train.shape[1]

def build_model(hparams):
    inp = keras.Input(shape=(INPUT_DIM,))
    x = inp
    for i in range(hparams["n_layers"]):
        x = layers.Dense(hparams[f"units_{i}"], activation=hparams["activation"])(x)
        if hparams.get("batchnorm", False):
            x = layers.BatchNormalization()(x)
        if hparams.get("dropout", 0.0) > 0:
            x = layers.Dropout(hparams["dropout"])(x)
    reg_out = layers.Dense(1, name="regression_output")(x)
    clf_out = layers.Dense(1, activation="sigmoid", name="classification_output")(x)
    model = keras.Model(inputs=inp, outputs=[reg_out, clf_out])
    opt = keras.optimizers.Adam(learning_rate=hparams["lr"])
    model.compile(
        optimizer=opt,
        loss={"regression_output": "mse", "classification_output": "binary_crossentropy"},
        metrics={"regression_output": [keras.metrics.MeanAbsoluteError()], "classification_output": [keras.metrics.AUC(name="auc")]}
    )
    return model

# default hparams
DEFAULT_HP = {
    "n_layers": 2,
    "units_0": 64,
    "units_1": 32,
    "activation": "relu",
    "batchnorm": True,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 100
}

# For speed, we skip Optuna here but the code supports it in earlier versions.
H = DEFAULT_HP.copy()
H["epochs"] = 120

model = build_model(H)
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

history = model.fit(
    X_train, {"regression_output": y_train_reg, "classification_output": y_train_clf},
    validation_data=(X_val, {"regression_output": y_val_reg, "classification_output": y_val_clf}),
    batch_size=H["batch_size"],
    epochs=H["epochs"],
    callbacks=[es],
    verbose=1
)

# loss curve
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("NN combined loss")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "nn_loss_curve.png")
plt.close()

def eval_split(X, y_reg, y_clf, split_name):
    preds = model.predict(X)
    y_reg_pred = preds[0].squeeze()
    y_clf_prob = preds[1].squeeze()
    y_clf_pred = (y_clf_prob >= 0.5).astype(int)

    mse_val = mean_squared_error(y_reg, y_reg_pred)
    rmse_val = float(mse_val ** 0.5)
    mae_val = float(mean_absolute_error(y_reg, y_reg_pred))

    acc = float(accuracy_score(y_clf, y_clf_pred))
    f1 = float(f1_score(y_clf, y_clf_pred))
    try:
        auc = float(roc_auc_score(y_clf, y_clf_prob))
    except Exception:
        auc = float("nan")

    # confusion
    cm = confusion_matrix(y_clf, y_clf_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(f"NN Confusion ({split_name})")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.savefig(REPORTS_DIR / "figures" / f"nn_confusion_{split_name}.png")
    plt.close()

    # residuals
    residuals = y_reg - y_reg_pred
    plt.figure(figsize=(5,3))
    plt.scatter(y_reg_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="grey")
    plt.xlabel("predicted")
    plt.ylabel("residual")
    plt.title(f"NN Residuals ({split_name})")
    plt.savefig(REPORTS_DIR / "figures" / f"nn_residuals_{split_name}.png")
    plt.close()

    metrics = {
        "rmse": rmse_val,
        "mae": mae_val,
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "tables" / f"nn_metrics_{split_name}.csv", index=False)
    return metrics

val_metrics = eval_split(X_val, y_val_reg, y_val_clf, "val")
test_metrics = eval_split(X_test, y_test_reg, y_test_clf, "test")

# save model & params
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
model_dir = MODELS_DIR / f"nn_model_{ts}"
model.save(str(model_dir))
joblib.dump(H, MODELS_DIR / "nn_best_params.pkl")

summary = {"best_params": H, "val_metrics": val_metrics, "test_metrics": test_metrics}
with open(REPORTS_DIR / "tables" / "nn_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("[nn] Training finished. Artifacts in models/ and reports/")
