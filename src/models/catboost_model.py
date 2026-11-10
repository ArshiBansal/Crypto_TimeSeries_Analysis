"""
src/models/catboost_model.py

CatBoost tabular forecasting / classification wrapper.
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
import joblib
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config_path: str = _DEFAULT_CONFIG,
    task: str = "regression",
    artifact_name: str = "catboost",
) -> dict:
    """
    Train a CatBoost model with early stopping on the validation set.

    Parameters
    ----------
    task : str
        'regression' | 'binary' | 'multiclass'
    artifact_name : str
        Base filename for the saved model artifact.

    Returns
    -------
    dict with 'model', 'scaler', 'artifact_path'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cb_cfg  = cfg["models"]["catboost"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    train_pool = Pool(X_train_s, y_train)
    val_pool   = Pool(X_val_s, y_val)

    common_params = dict(
        iterations=cb_cfg["iterations"],
        learning_rate=cb_cfg["learning_rate"],
        depth=cb_cfg["depth"],
        early_stopping_rounds=cb_cfg["early_stopping_rounds"],
        verbose=0,
    )

    if task == "regression":
        model = CatBoostRegressor(**common_params)
    elif task == "binary":
        model = CatBoostClassifier(loss_function="Logloss", **common_params)
    else:
        model = CatBoostClassifier(loss_function="MultiClass", **common_params)

    model.fit(train_pool, eval_set=val_pool)

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump({"model": model, "scaler": scaler}, artifact_path)
    print(f"  CatBoost artifact saved → {artifact_path}")

    return {"model": model, "scaler": scaler, "artifact_path": str(artifact_path)}
