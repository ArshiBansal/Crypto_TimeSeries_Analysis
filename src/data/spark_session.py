"""
src/data/spark_session.py
Centralised PySpark session factory.
Reads Spark config from configs/pipeline.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path as _Path

_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")

import yaml
from pyspark.sql import SparkSession


def get_spark(config_path: str = _DEFAULT_CONFIG) -> SparkSession:
    """
    Build (or retrieve) a SparkSession using settings in pipeline.yaml.

    On Windows it is common for multiple Java versions to coexist and for the
    system PATH (or VS Code / Jupyter's inherited environment) to resolve a
    newer, PySpark-incompatible Java instead of Java 17.  Set ``java_home``
    in the ``spark:`` section of pipeline.yaml to pin the correct JDK:

        spark:
          java_home: "C:\\\\Program Files\\\\Java\\\\jdk-17"

    PySpark 4.x supports Java 17 and Java 21 only.

    Parameters
    ----------
    config_path : str
        Path to the pipeline config YAML.

    Returns
    -------
    SparkSession
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["spark"]

    # ── Pin JAVA_HOME if specified in config ──────────────────────────────
    java_home = cfg.get("java_home")
    if java_home:
        java_home = str(java_home).strip()
        os.environ["JAVA_HOME"] = java_home
        # Prepend the bin dir so this java.exe wins over any PATH entry
        bin_dir = str(_Path(java_home) / "bin")
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[spark_session] JAVA_HOME pinned → {java_home}")
    else:
        # Warn if the active Java is not 17 or 21
        import subprocess, re
        try:
            result = subprocess.run(
                ["java", "-version"], capture_output=True, text=True, timeout=10
            )
            version_output = result.stderr or result.stdout
            match = re.search(r'version "(\d+)', version_output)
            if match:
                major = int(match.group(1))
                if major not in (17, 21):
                    print(
                        f"[spark_session] ⚠️  Java {major} detected — PySpark 4.x "
                        f"requires Java 17 or 21.  Set 'java_home' in configs/pipeline.yaml "
                        f"to point at your Java 17 / 21 installation."
                    )
        except Exception:
            pass  # java not on PATH or timeout — Spark will surface its own error

    builder = (
        SparkSession.builder.appName(cfg["app_name"])
        .master(cfg["master"])
    )
    for key, value in cfg.get("config", {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
