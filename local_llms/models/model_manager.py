"""Model management for loading and managing local LLMs."""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ModelManager:
    """Manages loading and caching of LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model manager.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        self.cache_dir = self.config.get("cache_dir", "./models_cache")
        self.device = self._get_device()
        
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_device(self) -> str:
        """Determine the best device to use.
        
        Returns:
            Device string (cuda, mps, or cpu)
        """
        device_config = self.config.get("device", "auto")
        
        if device_config != "auto":
            return device_config
        
        # Auto-detect best device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(
        self,
        model_name: str,
        load_in_8bit: Optional[bool] = None,
        load_in_4bit: Optional[bool] = None,
        trust_remote_code: bool = False,
        **kwargs
    ) -> None:
        """Load a model and its tokenizer.
        
        Args:
            model_name: HuggingFace model identifier or local path
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        
        # Get quantization settings
        if load_in_8bit is None:
            load_in_8bit = self.config.get("load_in_8bit", False)
        if load_in_4bit is None:
            load_in_4bit = self.config.get("load_in_4bit", False)
        
        # Prepare quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": trust_remote_code,
            **kwargs
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not quantization_config and self.device != "auto":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully!")
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model unloaded.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"error": "No model loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        return info
    
    def save_model(self, output_path: Union[str, Path]) -> None:
        """Save the current model and tokenizer.
        
        Args:
            output_path: Directory to save the model
        """
        if not self.model or not self.tokenizer:
            raise ValueError("No model loaded to save")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.tokenizer is not None
