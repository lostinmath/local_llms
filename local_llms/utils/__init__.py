"""Utility modules for security, data handling, and optimization."""

# Lazy imports
def __getattr__(name):
    if name == "DataEncryption":
        from .security import DataEncryption
        return DataEncryption
    elif name == "DataProcessor":
        from .data_processor import DataProcessor
        return DataProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DataEncryption", "DataProcessor"]
