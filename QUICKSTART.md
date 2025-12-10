# Quick Start Guide

Get started with Local LLMs in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/lostinmath/local_llms.git
cd local_llms

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## First Steps

### 1. Basic Text Generation

```python
from local_llms import ModelManager, InferenceEngine, Config

# Initialize
config = Config()
model_manager = ModelManager(config["model"])

# Load a model (starts with a small one)
model_manager.load_model("gpt2")

# Generate text
inference = InferenceEngine(model_manager, config["inference"])
response = inference.generate("The future of AI is", max_length=100)
print(response)
```

### 2. Secure Your Data

```python
from local_llms.utils.security import DataEncryption

# Create encryption handler
encryption = DataEncryption()

# Encrypt sensitive text
encrypted = encryption.encrypt_text("My private data")
decrypted = encryption.decrypt_text(encrypted)

# Encrypt files
encryption.encrypt_file("sensitive.txt")
```

### 3. Fine-tune on Your Data

```python
from local_llms.models.fine_tuning import FineTuner
from local_llms.utils.data_processor import DataProcessor

# Prepare your data
data = [
    {
        "instruction": "What is your task?",
        "input": "",
        "output": "I help users with their specific needs."
    },
    # Add more examples...
]

# Process and format data
processor = DataProcessor()
formatted = processor.prepare_instruction_data(data)
train_data = [{"text": f"{item['prompt']}{item['completion']}"} 
              for item in formatted]

# Fine-tune with LoRA (efficient)
fine_tuner = FineTuner(model_manager, config["fine_tuning"])
fine_tuner.prepare_model_for_lora()
dataset = fine_tuner.prepare_dataset(train_data)
fine_tuner.train(dataset, output_dir="./my_custom_model")
```

## Configuration

Create a `config.yaml` file:

```yaml
model:
  default_model: "gpt2"
  device: "auto"  # or "cuda", "cpu", "mps"
  load_in_4bit: true  # Enable for memory efficiency

inference:
  temperature: 0.7
  max_length: 512

fine_tuning:
  use_lora: true
  num_epochs: 3
```

Load it:

```python
config = Config("config.yaml")
```

## Environment Variables

Create a `.env` file:

```bash
ENCRYPTION_KEY=your_secure_key_here
MODEL_CACHE_DIR=./models_cache
DEVICE=cuda
```

## Tips

1. **Start Small**: Test with GPT-2 before using larger models
2. **Use Quantization**: Enable 4-bit or 8-bit for memory efficiency
3. **Encrypt Sensitive Data**: Always encrypt before storing
4. **Local Only**: Keep all processing on your machine
5. **LoRA for Fine-tuning**: Use LoRA to reduce training parameters by 99%

## Common Commands

```bash
# Run tests
pytest tests/

# Run examples
cd examples
python basic_usage.py
python security_example.py
python fine_tuning_example.py

# Install with dev dependencies
pip install -r requirements-dev.txt
```

## Next Steps

- Read the full [README.md](README.md)
- Check out [examples/](examples/)
- Review [CONTRIBUTING.md](CONTRIBUTING.md)
- Explore the API documentation

## Troubleshooting

**Out of Memory?**
- Enable quantization: `load_in_4bit=True`
- Use a smaller model
- Reduce batch size

**Model Download Issues?**
- Check internet connection
- Verify HuggingFace model name
- Try setting `cache_dir` in config

**Import Errors?**
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)

## Support

- GitHub Issues: [Report bugs](https://github.com/lostinmath/local_llms/issues)
- Discussions: [Ask questions](https://github.com/lostinmath/local_llms/discussions)

---

**Ready to build your private AI assistant? Start coding!** 🚀
