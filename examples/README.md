# Examples

This directory contains example scripts demonstrating the key features of Local LLMs.

## Setup

Before running the examples, install the package:

```bash
# From the root directory
pip install -e .
```

Or install just the dependencies:

```bash
pip install -r requirements.txt
```

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates:
- Loading a model
- Text generation
- Chat interface
- Model information

**Run:**
```bash
python basic_usage.py
```

**Note**: This downloads GPT-2 (~500MB) on first run.

### 2. Security Features (`security_example.py`)

Demonstrates:
- Text and file encryption/decryption
- Data anonymization (removing PII)
- Secure file deletion
- Data hashing

**Run:**
```bash
python security_example.py
```

**Key Takeaways**:
- Always encrypt sensitive data before storage
- Use anonymization for training data with PII
- Secure deletion overwrites files before removal

### 3. Fine-tuning (`fine_tuning_example.py`)

Demonstrates:
- Preparing custom training data
- LoRA-based efficient fine-tuning
- Instruction-following format
- Model saving

**Run:**
```bash
python fine_tuning_example.py
```

**Note**: This example uses minimal data for demonstration. For real fine-tuning:
- Use at least 100-1000 examples
- Train for multiple epochs
- Use larger models (7B+ parameters)
- Validate on a held-out test set

## Customizing Examples

### Change the Model

```python
# Instead of GPT-2
model_manager.load_model("gpt2")

# Try other models
model_manager.load_model("meta-llama/Llama-2-7b-hf", load_in_4bit=True)
model_manager.load_model("mistralai/Mistral-7B-v0.1", load_in_8bit=True)
```

### Adjust Generation Parameters

```python
response = inference.generate(
    prompt,
    max_length=200,        # Longer responses
    temperature=0.8,       # More creative
    top_p=0.95,           # Nucleus sampling
    repetition_penalty=1.2 # Reduce repetition
)
```

### Use Your Own Data

For fine-tuning, replace the sample data with your own:

```python
# Load from file
import json
with open('my_data.json', 'r') as f:
    data = json.load(f)

# Format as instruction data
formatted = data_processor.prepare_instruction_data(data)
```

## Tips for Success

### Memory Management

If you run out of memory:

1. **Use Quantization**:
   ```python
   model_manager.load_model("model_name", load_in_4bit=True)
   ```

2. **Reduce Batch Size**:
   ```python
   fine_tuner.train(..., batch_size=1)
   ```

3. **Use Smaller Models**:
   - GPT-2 (small, ~500MB)
   - GPT-2 Medium (~1.5GB)
   - GPT-2 Large (~3GB)

### Privacy Best Practices

1. **Keep Data Local**: Never upload to cloud
2. **Encrypt Everything**: Use `DataEncryption` for sensitive files
3. **Anonymize Training Data**: Remove PII before fine-tuning
4. **Secure Environment**: Use `.env` for keys, not hardcode

### Performance Optimization

1. **Use GPU**: Set `device: "cuda"` in config
2. **Batch Processing**: Use `generate_batch()` for multiple prompts
3. **Cache Models**: Models are cached after first download
4. **LoRA for Training**: Reduces memory by 10-100x

## Troubleshooting

**Model Download Fails**:
- Check internet connection
- Verify model name on HuggingFace
- Try setting `cache_dir` to a location with more space

**CUDA Out of Memory**:
- Enable 4-bit quantization
- Use smaller model
- Reduce batch size or max_length

**Import Errors**:
- Ensure package is installed: `pip install -e .`
- Install all dependencies: `pip install -r requirements.txt`

## Creating Your Own Examples

Feel free to create and share your own examples! Good example characteristics:

- Clear, focused demonstration
- Well-commented code
- Realistic use cases
- Privacy-conscious
- Handle errors gracefully

## Contributing

Found a bug or want to improve an example? See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Questions?

Open an issue on GitHub or check the main [README.md](../README.md).

---

**Happy experimenting with Local LLMs!** 🔒🤖
