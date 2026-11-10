"""
src/models/lgbm_model.py

LightGBM tabular forecasting / classification wrapper.
Fit strictly on training data; apply unchanged to val/test.
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config_path: str = _DEFAULT_CONFIG,
    task: str = "regression",        # "regression" | "binary" | "multiclass"
    artifact_name: str = "lgbm",
) -> dict:
    """
    Train a LightGBM model with early stopping on validation set.

    Parameters
    ----------
    X_train, y_train : training features and target
    X_val, y_val     : validation features and target
    task : str
        'regression' for return/vol forecasting; 'binary'/'multiclass' for regime classification.
    artifact_name : str
        Base filename for the saved model artifact.

    Returns
    -------
    dict with 'model', 'scaler', 'artifact_path'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lgbm_cfg = cfg["models"]["lightgbm"]
    art_dir   = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    objective_map = {
        "regression": "regression",
        "binary": "binary",
        "multiclass": "multiclass",
    }
    params = {
        "objective":        objective_map[task],
        "n_estimators":     lgbm_cfg["n_estimators"],
        "learning_rate":    lgbm_cfg["learning_rate"],
        "num_leaves":       lgbm_cfg["num_leaves"],
        "verbosity":        -1,
    }
    if task == "regression":
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        callbacks=[
            lgb.early_stopping(lgbm_cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump({"model": model, "scaler": scaler}, artifact_path)
    print(f"  LightGBM artifact saved → {artifact_path}")

    return {"model": model, "scaler": scaler, "artifact_path": str(artifact_path)}
