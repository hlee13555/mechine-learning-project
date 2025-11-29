# src/utils.py
"""
Small shared helpers (directories, metrics).
"""
from pathlib import Path
import os
import joblib
from sklearn.metrics import mean_squared_error
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_obj(obj: Any, path: Path):
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_obj(path: Path):
    return joblib.load(path)

def rmse(y_true, y_pred):
    """Robust RMSE calculation compatible with sklearn >=1.6"""
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)

def save_dict_as_json(d: Dict, path: Path):
    import json
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
