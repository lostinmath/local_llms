"""Inference engine for running LLM predictions."""

import torch
from typing import Optional, Dict, Any, List, Union
from transformers import GenerationConfig


class InferenceEngine:
    """Handles inference with loaded LLM models."""
    
    def __init__(self, model_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize inference engine.
        
        Args:
            model_manager: ModelManager instance with loaded model
            config: Configuration dictionary with inference settings
        """
        self.model_manager = model_manager
        self.config = config or {}
        
        # Default generation parameters
        self.default_params = {
            "max_length": self.config.get("max_length", 512),
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 0.9),
            "top_k": self.config.get("top_k", 50),
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
            "do_sample": self.config.get("do_sample", True),
        }
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text string, or list of strings if num_return_sequences > 1
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer
        device = self.model_manager.device
        
        # Use provided params or defaults
        gen_params = {
            "max_length": max_length or self.default_params["max_length"],
            "temperature": temperature or self.default_params["temperature"],
            "top_p": top_p or self.default_params["top_p"],
            "top_k": top_k or self.default_params["top_k"],
            "repetition_penalty": repetition_penalty or self.default_params["repetition_penalty"],
            "do_sample": do_sample if do_sample is not None else self.default_params["do_sample"],
            "num_return_sequences": num_return_sequences,
            **kwargs
        }
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        if device != "auto":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the output
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generated_texts.append(text)
        
        return generated_texts[0] if num_return_sequences == 1 else generated_texts
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in a batch.
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated text strings
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer
        device = self.model_manager.device
        
        # Merge with defaults
        gen_params = {**self.default_params, **generation_kwargs}
        gen_params.pop("num_return_sequences", None)  # Not supported in batch mode
        
        # Tokenize inputs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        if device != "auto":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            # Try to remove the prompt from the output
            if text.startswith(prompts[i]):
                text = text[len(prompts[i]):].strip()
            generated_texts.append(text)
        
        return generated_texts
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs
    ) -> str:
        """Generate response in chat format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **generation_kwargs: Generation parameters
            
        Returns:
            Generated response text
        """
        # Format messages into a prompt
        # This is a simple implementation; can be customized per model
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        return self.generate(prompt, **generation_kwargs)
    
    def update_generation_config(self, **kwargs) -> None:
        """Update default generation parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.default_params.update(kwargs)
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration.
        
        Returns:
            Dictionary of generation parameters
        """
        return self.default_params.copy()
