"""
src/features/__init__.py
Feature engineering sub-package.
"""
from .returns           import add_return_features
from .volatility        import add_volatility_features
from .volume            import add_volume_features
from .market_structure  import add_market_structure_features
from .pipeline          import build_feature_pipeline
from .microstructure    import add_ofi, add_vpin, add_arrival_features, fit_hawkes

__all__ = [
    "add_return_features",
    "add_volatility_features",
    "add_volume_features",
    "add_market_structure_features",
    "build_feature_pipeline",
    # Conditional (requires tick / order-book data):
    "add_ofi", "add_vpin", "add_arrival_features", "fit_hawkes",
]
