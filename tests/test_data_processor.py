"""Tests for data processing utilities."""

import os
import tempfile
import pytest
from local_llms.utils.data_processor import DataProcessor


def test_data_processor_init():
    """Test data processor initialization."""
    processor = DataProcessor()
    assert processor.data_dir.exists()


def test_prepare_instruction_data():
    """Test instruction data preparation."""
    processor = DataProcessor()
    
    data = [
        {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour"
        },
        {
            "instruction": "What is 2+2?",
            "input": "",
            "output": "4"
        }
    ]
    
    formatted = processor.prepare_instruction_data(data)
    
    assert len(formatted) == 2
    assert "Translate to French" in formatted[0]["prompt"]
    assert "Hello" in formatted[0]["prompt"]
    assert formatted[0]["completion"] == "Bonjour"
    assert "What is 2+2?" in formatted[1]["prompt"]


def test_prepare_chat_data():
    """Test chat data preparation."""
    processor = DataProcessor()
    
    data = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    formatted = processor.prepare_chat_data(data)
    
    assert len(formatted) == 3
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


def test_split_data():
    """Test data splitting."""
    processor = DataProcessor()
    
    data = list(range(100))
    splits = processor.split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_clean_text():
    """Test text cleaning."""
    processor = DataProcessor()
    
    dirty_text = "  This   has   extra    spaces  \n\n  and newlines  "
    clean = processor.clean_text(dirty_text)
    
    assert clean == "This has extra spaces and newlines"


def test_anonymize_data():
    """Test data anonymization."""
    processor = DataProcessor()
    
    text = "Contact me at john.doe@example.com or call 555-123-4567"
    anonymized = processor.anonymize_data(text)
    
    assert "[EMAIL]" in anonymized
    assert "[PHONE]" in anonymized
    assert "john.doe@example.com" not in anonymized
    assert "555-123-4567" not in anonymized


def test_jsonl_save_load():
    """Test JSONL save and load."""
    processor = DataProcessor()
    
    data = [
        {"id": 1, "text": "First item"},
        {"id": 2, "text": "Second item"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        processor.save_jsonl(data, temp_path)
        assert os.path.exists(temp_path)
        
        # Load
        loaded = processor.load_jsonl(temp_path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["text"] == "Second item"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
