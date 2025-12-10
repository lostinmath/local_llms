"""
Local LLMs - Privacy-focused, efficient custom language models for personal use.

This package provides tools for:
- Loading and running local LLM models
- Fine-tuning models on custom data with privacy
- Secure data handling and encryption
- Efficient inference with optimization techniques
"""

__version__ = "0.1.0"
__author__ = "lostinmath"

# Lazy imports to avoid importing heavy dependencies on package import
def __getattr__(name):
    if name == "ModelManager":
        from .models.model_manager import ModelManager
        return ModelManager
    elif name == "InferenceEngine":
        from .models.inference import InferenceEngine
        return InferenceEngine
    elif name == "DataEncryption":
        from .utils.security import DataEncryption
        return DataEncryption
    elif name == "Config":
        from .config.settings import Config
        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModelManager",
    "InferenceEngine", 
    "DataEncryption",
    "Config",
]
