"""
src/data/__init__.py
Data ingestion sub-package.
"""
from .loader import load_primary_dataset, load_secondary_dataset
from .spark_session import get_spark
from .profiler import profile_dataframe
from .cleaner import clean_and_align

__all__ = [
    "load_primary_dataset",
    "load_secondary_dataset",
    "get_spark",
    "profile_dataframe",
    "clean_and_align",
]
