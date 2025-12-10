"""Fine-tuning capabilities for customizing models on personal data."""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


class FineTuner:
    """Handles fine-tuning of LLM models on custom data."""
    
    def __init__(self, model_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize fine-tuner.
        
        Args:
            model_manager: ModelManager instance with loaded model
            config: Configuration dictionary with fine-tuning settings
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.trainer = None
    
    def prepare_model_for_lora(
        self,
        r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Prepare model for LoRA (Low-Rank Adaptation) training.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of module names to apply LoRA to
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        # Get parameters from config or use defaults
        r = r or self.config.get("lora_r", 8)
        lora_alpha = lora_alpha or self.config.get("lora_alpha", 16)
        lora_dropout = lora_dropout or self.config.get("lora_dropout", 0.05)
        
        # Common target modules for different architectures
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Prepare model for training if using quantization
        model = self.model_manager.model
        if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
            model = prepare_model_for_kbit_training(model)
        elif hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model_manager.model = get_peft_model(model, lora_config)
        
        print("Model prepared for LoRA training")
        print(f"Trainable parameters: {self._count_trainable_parameters()}")
    
    def _count_trainable_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        model = self.model_manager.model
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": 100 * trainable_params / all_param
        }
    
    def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        text_column: str = "text",
        max_length: Optional[int] = None,
    ) -> Dataset:
        """Prepare dataset for training.
        
        Args:
            data: List of training examples
            text_column: Name of the text column
            max_length: Maximum sequence length
            
        Returns:
            Prepared HuggingFace Dataset
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        tokenizer = self.model_manager.tokenizer
        max_length = max_length or self.config.get("max_length", 512)
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[str] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        **training_kwargs
    ) -> None:
        """Train the model on prepared dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            **training_kwargs: Additional training arguments
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        # Get parameters from config or use defaults
        output_dir = output_dir or self.config.get("output_dir", "./fine_tuned_models")
        num_epochs = num_epochs or self.config.get("num_epochs", 3)
        learning_rate = learning_rate or self.config.get("learning_rate", 2e-5)
        batch_size = batch_size or self.config.get("batch_size", 4)
        
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        warmup_steps = self.config.get("warmup_steps", 100)
        logging_steps = self.config.get("logging_steps", 10)
        save_steps = self.config.get("save_steps", 100)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="none",  # Disable telemetry for privacy
            **training_kwargs
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model")
        self.save_model(final_model_path)
    
    def save_model(self, output_path: Union[str, Path]) -> None:
        """Save the fine-tuned model.
        
        Args:
            output_path: Directory to save the model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {output_path}")
        self.model_manager.model.save_pretrained(output_path)
        self.model_manager.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trainer:
            raise ValueError("No trainer initialized. Please run train() first.")
        
        print("Evaluating model...")
        metrics = self.trainer.evaluate(eval_dataset)
        print("Evaluation completed!")
        
        return metrics
