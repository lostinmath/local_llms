"""Tests for configuration management."""

import os
import tempfile
import pytest
from pathlib import Path
from local_llms.config.settings import Config


def test_default_config():
    """Test default configuration initialization."""
    config = Config()
    
    assert config.get("model.default_model") == "gpt2"
    assert config.get("inference.temperature") == 0.7
    assert config.get("security.encrypt_data") is True
    assert config.get("fine_tuning.use_lora") is True


def test_config_get_set():
    """Test getting and setting config values."""
    config = Config()
    
    # Test get
    assert config.get("model.device") == "auto"
    assert config.get("nonexistent.key", "default") == "default"
    
    # Test set
    config.set("model.device", "cuda")
    assert config.get("model.device") == "cuda"
    
    # Test nested set
    config.set("new.nested.value", 42)
    assert config.get("new.nested.value") == 42


def test_config_dict_access():
    """Test dictionary-style access."""
    config = Config()
    
    assert config["model"]["default_model"] == "gpt2"
    
    config["model"]["custom_key"] = "custom_value"
    assert config["model"]["custom_key"] == "custom_value"


def test_config_save_load():
    """Test saving and loading configuration."""
    config = Config()
    config.set("model.device", "cuda")
    config.set("inference.temperature", 0.9)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save config
        config.save(temp_path)
        assert os.path.exists(temp_path)
        
        # Load config
        loaded_config = Config(temp_path)
        assert loaded_config.get("model.device") == "cuda"
        assert loaded_config.get("inference.temperature") == 0.9
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
