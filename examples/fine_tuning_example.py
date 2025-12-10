"""Fine-tuning example for Local LLMs.

To run this example:
1. Install the package: pip install -e ..
2. Run: python fine_tuning_example.py
"""

from local_llms import ModelManager, Config
from local_llms.models.fine_tuning import FineTuner
from local_llms.utils.data_processor import DataProcessor


def main():
    """Demonstrate fine-tuning on custom data."""
    
    print("="*50)
    print("FINE-TUNING EXAMPLE")
    print("="*50)
    
    # Initialize configuration
    print("\n1. Initializing configuration...")
    config = Config()
    
    # Prepare sample training data
    print("\n2. Preparing training data...")
    data_processor = DataProcessor()
    
    # Example: Instruction-following data
    sample_data = [
        {
            "instruction": "Explain what a local LLM is",
            "input": "",
            "output": "A local LLM is a language model that runs on your own hardware, ensuring data privacy and independence from cloud services."
        },
        {
            "instruction": "What are the benefits of fine-tuning?",
            "input": "",
            "output": "Fine-tuning allows you to adapt a pre-trained model to your specific use case, improving performance on domain-specific tasks."
        },
        {
            "instruction": "How can I ensure data privacy?",
            "input": "",
            "output": "Use local models, encrypt sensitive data, and avoid sending information to external servers."
        },
    ]
    
    # Format data for training
    formatted_data = data_processor.prepare_instruction_data(sample_data)
    
    # Convert to text format for language modeling
    train_data = [{"text": f"{item['prompt']}{item['completion']}"} for item in formatted_data]
    
    print(f"   Prepared {len(train_data)} training examples")
    
    # Initialize model manager
    print("\n3. Loading base model...")
    model_manager = ModelManager(config["model"])
    
    # Load a small model (for demonstration)
    # In practice, you might use a larger model like "meta-llama/Llama-2-7b-hf"
    model_manager.load_model("gpt2", load_in_8bit=False)
    
    # Initialize fine-tuner
    print("\n4. Setting up fine-tuning...")
    fine_tuner = FineTuner(model_manager, config["fine_tuning"])
    
    # Prepare model for LoRA (efficient fine-tuning)
    if config["fine_tuning"].get("use_lora", True):
        print("   Applying LoRA for efficient training...")
        fine_tuner.prepare_model_for_lora()
        
        # Show parameter count
        params = fine_tuner._count_trainable_parameters()
        print(f"   Trainable parameters: {params['trainable_params']:,}")
        print(f"   Total parameters: {params['all_params']:,}")
        print(f"   Trainable percentage: {params['trainable_percentage']:.2f}%")
    
    # Prepare dataset
    print("\n5. Preparing dataset...")
    train_dataset = fine_tuner.prepare_dataset(train_data)
    print(f"   Dataset size: {len(train_dataset)}")
    
    # Train the model
    print("\n6. Training model...")
    print("   Note: This is a demonstration with minimal data.")
    print("   For real fine-tuning, use more data and longer training.")
    
    # Use minimal settings for demonstration
    fine_tuner.train(
        train_dataset,
        output_dir="./demo_fine_tuned",
        num_epochs=1,
        batch_size=1,
        learning_rate=3e-4,
        logging_steps=1,
        save_steps=10,
    )
    
    print("\n7. Training completed!")
    print("   Model saved to: ./demo_fine_tuned/final_model")
    
    # Cleanup
    print("\n8. Cleaning up...")
    model_manager.unload_model()
    print("\nDone!")


if __name__ == "__main__":
    main()
