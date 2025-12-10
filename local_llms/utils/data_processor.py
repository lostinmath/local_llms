"""Data processing utilities for preparing training and inference data."""

import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
import pandas as pd


class DataProcessor:
    """Handles data processing for LLM fine-tuning and inference."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data processor.
        
        Args:
            data_dir: Directory for data storage. Defaults to ./data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of dictionaries
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
        """Save data to JSONL file.
        
        Args:
            data: List of dictionaries to save
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def prepare_instruction_data(
        self,
        data: List[Dict[str, str]],
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output"
    ) -> List[Dict[str, str]]:
        """Prepare data in instruction-following format.
        
        Args:
            data: List of data dictionaries
            instruction_key: Key for instruction field
            input_key: Key for input field
            output_key: Key for output field
            
        Returns:
            Formatted data for instruction tuning
        """
        formatted_data = []
        
        for item in data:
            instruction = item.get(instruction_key, "")
            input_text = item.get(input_key, "")
            output = item.get(output_key, "")
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            formatted_data.append({
                "prompt": prompt,
                "completion": output
            })
        
        return formatted_data
    
    def prepare_chat_data(
        self,
        data: List[Dict[str, Any]],
        role_key: str = "role",
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """Prepare data in chat format.
        
        Args:
            data: List of message dictionaries
            role_key: Key for role field (user, assistant, system)
            content_key: Key for content field
            
        Returns:
            Formatted chat data
        """
        formatted_data = []
        
        for item in data:
            if isinstance(item, dict) and role_key in item and content_key in item:
                formatted_data.append({
                    "role": item[role_key],
                    "content": item[content_key]
                })
        
        return formatted_data
    
    def split_data(
        self,
        data: List[Any],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, List[Any]]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: List of data items
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Dictionary with 'train', 'val', and 'test' splits
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        if shuffle:
            import random
            data = data.copy()
            random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def anonymize_data(self, text: str, patterns: Optional[Dict[str, str]] = None) -> str:
        """Anonymize sensitive information in text.
        
        Args:
            text: Input text with potential sensitive info
            patterns: Dictionary of regex patterns to replacements
            
        Returns:
            Anonymized text
        """
        import re
        
        if patterns is None:
            patterns = {
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[PHONE]',
                r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': '[IP]',
            }
        
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def export_to_csv(self, data: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """Export data to CSV file.
        
        Args:
            data: List of dictionaries
            output_path: Output CSV file path
        """
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def import_from_csv(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Import data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of dictionaries
        """
        df = pd.read_csv(file_path)
        return df.to_dict('records')
