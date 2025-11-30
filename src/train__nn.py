# src/train_nn.py
"""
Train a multi-head neural network for both regression (Y1) and classification (is_efficient).
- Fits a simple imputer+scaler preprocessor on TRAIN only and reuses it.
- Optionally uses Optuna if available; otherwise uses a small random-search fallback.
- Saves model, preprocessor, and metrics/figures to models/ and reports/.
"""
import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import mlflow  # added mlflow to ensure consistent tracking on Windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

os.makedirs(REPORTS_DIR / "figures", exist_ok=True)
os.makedirs(REPORTS_DIR / "tables", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# --- MLflow Windows-safe tracking URI setup ---
tracking_uri = "file:///" + str(MLRUNS_DIR).replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("energy_efficiency_nn")
# -----------------------------------------------

# Deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    raise RuntimeError("TensorFlow 2.x is required to run train_nn.py") from e

# Optuna optional
try:
    import optuna
    OPTUNA = True
except Exception:
    OPTUNA = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# load data
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

train["YSum"] = train["Y1"]
val["YSum"] = val["Y1"]
test["YSum"] = test["Y1"]

def numeric_columns_from_df(df):
    return [c for c in df.columns if str(c).startswith("X")]

numeric_cols = numeric_columns_from_df(train)

# fit preprocessor on train only
preproc = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
preproc.fit(train[numeric_cols])
joblib.dump(preproc, MODELS_DIR / "preprocessor_nn.pkl")

X_train = preproc.transform(train[numeric_cols])
X_val = preproc.transform(val[numeric_cols])
X_test = preproc.transform(test[numeric_cols])

y_train_reg = train["YSum"].values
y_val_reg = val["YSum"].values
y_test_reg = test["YSum"].values

y_train_clf = train["is_efficient"].values
y_val_clf = val["is_efficient"].values
y_test_clf = test["is_efficient"].values

INPUT_DIM = X_train.shape[1]

def build_model(hparams):
    inp = keras.Input(shape=(INPUT_DIM,))
    x = inp
    for i in range(hparams["n_layers"]):
        x = layers.Dense(hparams[f"units_{i}"], activation=hparams["activation"])(x)
        if hparams["batchnorm"]:
            x = layers.BatchNormalization()(x)
        if hparams["dropout"] > 0:
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

# default params
default_hparams = {
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

def objective_optuna(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hparams = {
        "n_layers": n_layers,
        "activation": trial.suggest_categorical("activation", ["relu", "elu", "tanh"]),
        "batchnorm": trial.suggest_categorical("batchnorm", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 60
    }
    for i in range(n_layers):
        hparams[f"units_{i}"] = trial.suggest_categorical(f"units_{i}", [16, 32, 64, 128])
    model = build_model(hparams)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X_train, {"regression_output": y_train_reg, "classification_output": y_train_clf},
        validation_data=(X_val, {"regression_output": y_val_reg, "classification_output": y_val_clf}),
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        callbacks=[es],
        verbose=0
    )
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val_reg, preds_val[0].squeeze()))
    try:
        auc = roc_auc_score(y_val_clf, preds_val[1].squeeze())
    except Exception:
        auc = 0.5
    return rmse + (1.0 - auc)

# tuning
BEST_PARAMS = default_hparams.copy()
if OPTUNA:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_optuna, n_trials=20)
    best = study.best_trial.params
    BEST_PARAMS = {
        "n_layers": best.get("n_layers", 2),
        "activation": best.get("activation", "relu"),
        "batchnorm": best.get("batchnorm", True),
        "dropout": best.get("dropout", 0.2),
        "lr": best.get("lr", 1e-3),
        "batch_size": best.get("batch_size", 32),
        "epochs": 100
    }
    for i in range(BEST_PARAMS["n_layers"]):
        BEST_PARAMS[f"units_{i}"] = best.get(f"units_{i}", 32)
    with open(REPORTS_DIR / "tables" / "optuna_best_params.txt", "w") as f:
        f.write(str(study.best_trial.params))
else:
    # fallback simple candidates
    candidates = [
        {"n_layers": 2, "units_0": 64, "units_1": 32, "activation": "relu", "batchnorm": True, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
        {"n_layers": 1, "units_0": 64, "activation": "relu", "batchnorm": True, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
        {"n_layers": 3, "units_0": 128, "units_1": 64, "units_2": 32, "activation": "elu", "batchnorm": True, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
    ]
    best_score = None
    for cand in candidates:
        cand["epochs"] = 80
        model = build_model(cand)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(
            X_train, {"regression_output": y_train_reg, "classification_output": y_train_clf},
            validation_data=(X_val, {"regression_output": y_val_reg, "classification_output": y_val_clf}),
            batch_size=cand["batch_size"],
            epochs=cand["epochs"],
            callbacks=[es],
            verbose=0
        )
        preds_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val_reg, preds_val[0].squeeze()))
        try:
            auc = roc_auc_score(y_val_clf, preds_val[1].squeeze())
        except Exception:
            auc = 0.5
        score = rmse + (1.0 - auc)
        if best_score is None or score < best_score:
            best_score = score
            BEST_PARAMS = cand.copy()
    with open(REPORTS_DIR / "tables" / "random_search_best_params.txt", "w") as f:
        f.write(str(BEST_PARAMS))

# final train
BEST_PARAMS["epochs"] = 200
model = build_model(BEST_PARAMS)
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
history = model.fit(
    X_train, {"regression_output": y_train_reg, "classification_output": y_train_clf},
    validation_data=(X_val, {"regression_output": y_val_reg, "classification_output": y_val_clf}),
    batch_size=BEST_PARAMS.get("batch_size", 32),
    epochs=BEST_PARAMS["epochs"],
    callbacks=[es],
    verbose=1
)

# save loss curve
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

    rmse = np.sqrt(mean_squared_error(y_reg, y_reg_pred))
    mae = mean_absolute_error(y_reg, y_reg_pred)
    acc = accuracy_score(y_clf, y_clf_pred)
    f1 = f1_score(y_clf, y_clf_pred)
    try:
        auc = roc_auc_score(y_clf, y_clf_prob)
    except Exception:
        auc = float("nan")

    # confusion matrix
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

    metrics = {"rmse": float(rmse), "mae": float(mae), "accuracy": float(acc), "f1": float(f1), "auc": float(auc)}
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "tables" / f"nn_metrics_{split_name}.csv", index=False)
    return metrics

val_metrics = eval_split(X_val, y_val_reg, y_val_clf, "val")
test_metrics = eval_split(X_test, y_test_reg, y_test_clf, "test")

# save model and params
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = MODELS_DIR / f"nn_model_{ts}"
model_path = MODELS_DIR / f"nn_model_{ts}.keras"
model.save(str(model_path))

joblib.dump(BEST_PARAMS, MODELS_DIR / "nn_best_params.pkl")

summary = {"best_params": BEST_PARAMS, "val_metrics": val_metrics, "test_metrics": test_metrics}
import json
with open(REPORTS_DIR / "tables" / "nn_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("NN training complete. Artifacts in models/ and reports/")
