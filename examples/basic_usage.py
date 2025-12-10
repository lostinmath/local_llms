"""Basic usage example for Local LLMs."""

import sys
sys.path.insert(0, '../')

from local_llms import ModelManager, InferenceEngine, Config


def main():
    """Demonstrate basic model loading and inference."""
    
    # Initialize configuration
    print("Initializing configuration...")
    config = Config()
    
    # Initialize model manager
    print("\nInitializing model manager...")
    model_manager = ModelManager(config["model"])
    
    # Load a small model for demonstration (GPT-2)
    print("\nLoading model...")
    model_manager.load_model("gpt2")
    
    # Display model information
    print("\nModel Information:")
    info = model_manager.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    inference = InferenceEngine(model_manager, config["inference"])
    
    # Generate text
    print("\n" + "="*50)
    print("TEXT GENERATION EXAMPLE")
    print("="*50)
    
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    response = inference.generate(
        prompt,
        max_length=100,
        temperature=0.7,
    )
    
    print(f"\nResponse: {response}")
    
    # Chat example
    print("\n" + "="*50)
    print("CHAT EXAMPLE")
    print("="*50)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the benefits of using local LLMs?"},
    ]
    
    print("\nMessages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    print("\nGenerating response...")
    chat_response = inference.chat(messages, max_length=150)
    print(f"\nAssistant: {chat_response}")
    
    # Cleanup
    print("\n" + "="*50)
    print("Cleaning up...")
    model_manager.unload_model()
    print("Done!")


if __name__ == "__main__":
    main()
