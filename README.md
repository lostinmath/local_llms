# Local LLMs 🔒🤖

**Privacy-focused, efficient custom language models for personal use**

Local LLMs is a Python framework for running and fine-tuning large language models on your own hardware. It prioritizes **data privacy**, **efficiency**, and **customization** to help you create AI assistants that are truly yours.

## ✨ Features

### 🔐 Privacy & Security
- **100% Local Processing**: All data stays on your machine
- **Data Encryption**: Built-in encryption for sensitive data
- **Secure Deletion**: Overwrite files before deletion
- **Data Anonymization**: Remove PII from training data
- **No Telemetry**: Zero data sent to external servers

### ⚡ Efficiency
- **Quantization Support**: Run models in 4-bit or 8-bit precision
- **LoRA Fine-tuning**: Train with <1% of normal parameters
- **Optimized Inference**: Fast text generation with configurable parameters
- **Memory Management**: Automatic cache cleanup and efficient model loading
- **Batch Processing**: Handle multiple prompts efficiently

### 🎯 Customization
- **Fine-tuning on Custom Data**: Adapt models to your specific needs
- **Multiple Model Formats**: Support for HuggingFace models and GGUF
- **Flexible Configuration**: YAML-based config with environment variables
- **Instruction Tuning**: Built-in support for instruction-following models
- **Chat Interface**: Easy-to-use chat functionality

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

### Basic Installation

```bash
git clone https://github.com/lostinmath/local_llms.git
cd local_llms
pip install -r requirements.txt
```

### With GPU Support (Recommended)

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

## 📖 Quick Start

### Basic Usage

```python
from local_llms import ModelManager, InferenceEngine, Config

# Initialize
config = Config()
model_manager = ModelManager(config["model"])
model_manager.load_model("gpt2")

# Generate text
inference = InferenceEngine(model_manager, config["inference"])
response = inference.generate("The future of AI is", max_length=100)
print(response)
```

### Chat Interface

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are local LLMs?"},
]

response = inference.chat(messages)
print(response)
```

### Fine-tuning on Custom Data

```python
from local_llms.models.fine_tuning import FineTuner
from local_llms.utils.data_processor import DataProcessor

# Prepare your data
data_processor = DataProcessor()
training_data = [
    {
        "instruction": "Your instruction here",
        "input": "Optional input",
        "output": "Expected output"
    },
    # ... more examples
]

formatted_data = data_processor.prepare_instruction_data(training_data)
train_data = [{"text": f"{item['prompt']}{item['completion']}"} 
              for item in formatted_data]

# Fine-tune with LoRA
fine_tuner = FineTuner(model_manager, config["fine_tuning"])
fine_tuner.prepare_model_for_lora()
dataset = fine_tuner.prepare_dataset(train_data)
fine_tuner.train(dataset, output_dir="./my_model")
```

### Security Features

```python
from local_llms.utils.security import DataEncryption

# Encrypt sensitive data
encryption = DataEncryption()
encrypted = encryption.encrypt_text("sensitive information")
decrypted = encryption.decrypt_text(encrypted)

# Anonymize PII
from local_llms.utils.data_processor import DataProcessor
processor = DataProcessor()
clean_text = processor.anonymize_data("Email: john@example.com Phone: 555-1234")
```

## 📁 Project Structure

```
local_llms/
├── local_llms/           # Main package
│   ├── models/           # Model management and training
│   │   ├── model_manager.py    # Load and manage models
│   │   ├── inference.py        # Text generation
│   │   └── fine_tuning.py      # Fine-tuning with LoRA
│   ├── utils/            # Utilities
│   │   ├── security.py         # Encryption and privacy
│   │   └── data_processor.py   # Data preparation
│   └── config/           # Configuration
│       └── settings.py         # Config management
├── examples/             # Example scripts
│   ├── basic_usage.py          # Basic model usage
│   ├── fine_tuning_example.py  # Fine-tuning demo
│   └── security_example.py     # Security features
├── data/                 # Data directory (gitignored)
├── tests/                # Test suite
└── requirements.txt      # Dependencies
```

## ⚙️ Configuration

Create a `config.yaml` file to customize settings:

```yaml
model:
  default_model: "gpt2"
  cache_dir: "./models_cache"
  device: "auto"  # auto, cpu, cuda, mps
  load_in_8bit: false
  load_in_4bit: false

inference:
  max_length: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1

fine_tuning:
  output_dir: "./fine_tuned_models"
  num_epochs: 3
  learning_rate: 2e-5
  use_lora: true
  lora_r: 8

security:
  encrypt_data: true
  secure_delete: true

privacy:
  local_only: true
  no_telemetry: true
```

Load the config:

```python
config = Config("config.yaml")
```

## 🔧 Advanced Usage

### Load Different Model Sizes

```python
# Small model (for testing)
model_manager.load_model("gpt2")

# Medium model with quantization
model_manager.load_model("meta-llama/Llama-2-7b-hf", load_in_4bit=True)

# Large model with 8-bit quantization
model_manager.load_model("meta-llama/Llama-2-13b-hf", load_in_8bit=True)
```

### Batch Processing

```python
prompts = [
    "Translate to French: Hello",
    "Summarize: AI is...",
    "Complete: The sky is"
]

responses = inference.generate_batch(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}\nA: {response}\n")
```

### Custom Generation Parameters

```python
response = inference.generate(
    "Write a story about",
    max_length=500,
    temperature=0.8,      # Higher = more creative
    top_p=0.95,          # Nucleus sampling
    top_k=50,            # Top-k sampling
    repetition_penalty=1.2  # Reduce repetition
)
```

## 🛡️ Privacy Best Practices

1. **Keep Data Local**: Never upload sensitive data to cloud services
2. **Encrypt Everything**: Use encryption for any stored sensitive data
3. **Anonymize Training Data**: Remove PII before fine-tuning
4. **Secure Deletion**: Use secure_delete() for sensitive files
5. **Environment Variables**: Store keys in `.env` files (never commit these!)
6. **Audit Data Flow**: Review what data your models process

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- [Transformers](https://huggingface.co/transformers/) by HuggingFace
- [PyTorch](https://pytorch.org/)
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization

## 📚 Resources

- [HuggingFace Models](https://huggingface.co/models)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Quantization Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

## 🆘 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ for privacy-conscious AI enthusiasts**