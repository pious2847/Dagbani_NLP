import os
import sys
import torch
from transformers import pipeline

# --- CONFIGURATION ---
MODEL_PATH = "../models/dagbani_translator_local"

def check_setup():
    """Checks if the model folder exists and if PyTorch can detect a GPU."""
    print("--- System & Model Check ---")
    
    # 1. Check if the model directory exists
    if not os.path.isdir(MODEL_PATH):
        print(f"❌ Error: Model directory not found at '{MODEL_PATH}'")
        print("Please make sure you have run the training script successfully and the model folder exists.")
        sys.exit(1) # Exit the script if the model isn't found
    
    print(f"✅ Model directory found at: '{MODEL_PATH}'")

    # 2. Check PyTorch and GPU status
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {device_name}. Using GPU for faster inference.")
    else:
        print("⚠️ Warning: No GPU detected. Model will run on the CPU, which will be slower.")
    
    print("-" * 30)
    return is_cuda_available

def main():
    """Main function to load the model and translate text from the command line."""
    
    use_gpu = check_setup()
    
    # Determine which device to use: GPU (0) or CPU (-1) for the pipeline
    device_index = 0 if use_gpu else -1

    try:
        print("Loading your fine-tuned translation model...")
        # Create the translation pipeline, loading your local model
        translator = pipeline("translation", model=MODEL_PATH, device=device_index)
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")
        print("Please ensure all required libraries are installed (`pip install torch transformers sentencepiece`)")
        sys.exit(1)

    # Check if a sentence was provided as a command-line argument
    if len(sys.argv) > 1:
        # Join all arguments after the script name (e.g., "python translate.py") into a single sentence
        text_to_translate = " ".join(sys.argv[1:])
        print(f"\nTranslating: '{text_to_translate}'")
        
        # The pipeline returns a list with a dictionary inside
        result = translator(text_to_translate)
        
        print("\n--- Translation Result ---")
        print(f"  English: {text_to_translate}")
        print(f"  Dagbani: {result[0]['translation_text']}")
        print("--------------------------")
    else:
        # If no sentence is provided, show instructions on how to use the script
        print("\n--- How to Use This Script ---")
        print("Run the script from your terminal followed by the English sentence you want to translate.")
        print("\nExample:")
        print("  python translate.py \"what is your name?\"")
        print("  python translate.py \"1?\"")
        print("  python translate.py \"the weather is good today\"")

# This ensures the main() function runs only when the script is executed directly
if __name__ == "__main__":
    main()