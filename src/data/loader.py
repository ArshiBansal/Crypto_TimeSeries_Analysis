"""
src/data/loader.py

Loads Bitcoin market datasets via the HuggingFace `datasets` library.

IMPORTANT
---------
Datasets are fetched IN-MEMORY via the library API (streaming=True by default).
No files are downloaded to disk; the HuggingFace cache (~/.cache/huggingface/)
is used only for library-managed metadata.

Sources
-------
Primary  : WinkingFace/CryptoLM-Bitcoin-BTC-USDT
Secondary: lanasx3/crypto-market-datasets
"""
from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
from datasets import load_dataset, IterableDataset, Dataset

# Resolve project root from this file's location: src/data/ → project root
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


def _load_cfg(config_path: str = _DEFAULT_CONFIG) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["datasets"]


# ── Primary Dataset ───────────────────────────────────────────────────────────

def load_primary_dataset(
    config_path: str = _DEFAULT_CONFIG,
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """
    Load the primary CryptoLM BTC-USDT dataset via the HuggingFace
    `datasets` library.

    Parameters
    ----------
    config_path : str
        Path to pipeline.yaml.
    streaming : bool
        If True (default), returns an IterableDataset — rows are fetched
        lazily; nothing is written to disk beyond library-managed metadata.
        If False, returns a fully materialised Dataset (held in memory /
        Arrow format in the HF cache).

    Returns
    -------
    datasets.IterableDataset | datasets.Dataset
    """
    cfg = _load_cfg(config_path)["primary"]
    dataset = load_dataset(
        cfg["repo_id"],
        split=cfg.get("split", "train"),
        streaming=streaming,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    return dataset


def primary_to_pandas(
    config_path: str = _DEFAULT_CONFIG,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Materialise primary dataset as a pandas DataFrame.

    Strategy
    --------
    - max_rows is set  → streaming=True + .take(n)   (fast sample for audit/EDA)
    - max_rows is None → streaming=False + .to_pandas() (Arrow batch load, much faster
                         than row-by-row streaming for a full dataset)

    Parameters
    ----------
    max_rows : int | None
        Cap on rows to load.  Pass e.g. 50_000 for audit.
        Pass None only when you need the full dataset for training.

    Returns
    -------
    pd.DataFrame
    """
    cfg = _load_cfg(config_path)["primary"]

    if max_rows is not None:
        # Streaming sample — fast, low memory
        ds = load_dataset(
            cfg["repo_id"],
            split=cfg.get("split", "train"),
            streaming=True,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        return pd.DataFrame(list(ds.take(max_rows)))
    else:
        # Full load via Arrow (non-streaming) — far faster than row-by-row
        print("Loading full dataset via Arrow (non-streaming)…")
        ds = load_dataset(
            cfg["repo_id"],
            split=cfg.get("split", "train"),
            streaming=False,
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
        return ds.to_pandas()


# ── Secondary Dataset ─────────────────────────────────────────────────────────

def load_secondary_dataset(
    config_path: str = _DEFAULT_CONFIG,
    name: str | None = None,
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """
    Load a subset from the Crypto Market Data Lake dataset.

    Parameters
    ----------
    name : str | None
        Config / subset name within the dataset repo.  Defaults to the
        ``name`` key in pipeline.yaml (``klines_spot``).  Pass explicitly
        to override (e.g. ``'funding_rates'``).
    streaming : bool
        Same semantics as ``load_primary_dataset``.

    Returns
    -------
    datasets.IterableDataset | datasets.Dataset
    """
    cfg = _load_cfg(config_path)["secondary"]
    # Resolve name: caller arg → yaml key → error
    resolved_name = name or cfg.get("name")
    if not resolved_name:
        raise ValueError(
            "Secondary dataset requires a config name. "
            "Set 'name' under datasets.secondary in pipeline.yaml, "
            "e.g.  name: klines_spot"
        )
    dataset = load_dataset(
        cfg["repo_id"],
        name=resolved_name,
        split=cfg.get("split", "train"),
        streaming=streaming,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    return dataset


def secondary_to_pandas(
    config_path: str = _DEFAULT_CONFIG,
    name: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Materialise secondary dataset (or a subset) as a pandas DataFrame.

    Uses streaming sample when max_rows is set; Arrow batch load otherwise.
    The ``name`` config is read from pipeline.yaml if not supplied.
    """
    cfg = _load_cfg(config_path)["secondary"]
    resolved_name = name or cfg.get("name")
    if not resolved_name:
        raise ValueError(
            "Secondary dataset requires a config name. "
            "Set 'name' under datasets.secondary in pipeline.yaml."
        )

    common_kwargs: dict = dict(
        split=cfg.get("split", "train"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    if max_rows is not None:
        ds = load_dataset(cfg["repo_id"], name=resolved_name, streaming=True, **common_kwargs)
        return pd.DataFrame(list(ds.take(max_rows)))
    else:
        print("Loading full secondary dataset via Arrow (non-streaming)…")
        ds = load_dataset(cfg["repo_id"], name=resolved_name, streaming=False, **common_kwargs)
        return ds.to_pandas()
