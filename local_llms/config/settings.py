"""Configuration management for Local LLMs."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Manages configuration for the local LLM system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default settings.
        """
        load_dotenv()
        
        self.config_path = config_path
        self.settings = self._load_default_settings()
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            "model": {
                "default_model": "gpt2",
                "cache_dir": os.getenv("MODEL_CACHE_DIR", "./models_cache"),
                "device": os.getenv("DEVICE", "auto"),  # auto, cpu, cuda, mps
                "max_memory": None,
                "load_in_8bit": False,
                "load_in_4bit": False,
            },
            "inference": {
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
            },
            "security": {
                "encrypt_data": True,
                "encryption_key_env": "ENCRYPTION_KEY",
                "secure_delete": True,
            },
            "fine_tuning": {
                "output_dir": "./fine_tuned_models",
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 100,
                "use_lora": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
            },
            "privacy": {
                "local_only": True,
                "no_telemetry": True,
                "clear_cache_on_exit": False,
            }
        }
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Merge with defaults
        self._deep_update(self.settings, file_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "model.device")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.settings
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        target = self.settings
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.settings, f, default_flow_style=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.settings[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.settings[key] = value
