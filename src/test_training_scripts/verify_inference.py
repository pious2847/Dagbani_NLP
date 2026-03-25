import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the saved model (relative to this script in src/test_training_scripts/)
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../models/final_dagbani_nllb")

def verify_model():
    print(f"--- Verifying Model at {MODEL_PATH} ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model path not found: {MODEL_PATH}")
        return

    try:
        print("Loading tokenizer and model...")
        # Load the fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Test Sentences
        test_sentences = [
            ("I am eating rice.", "eng_Latn", "dag_Latn"),
            ("Good morning.", "eng_Latn", "dag_Latn"),
        ]
        
        print("\n--- Running Test Translations ---")
        for text, src, tgt in test_sentences:
            print(f"\nInput ({src}): {text}")
            
            inputs = tokenizer(text, return_tensors="pt")
            
            # NLLB translation requires forcing the target language token
            # Use convert_tokens_to_ids for robustness
            forced_bos_id = tokenizer.convert_tokens_to_ids(tgt)
            
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_id, 
                max_length=30
            )
            
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"Output ({tgt}): {translation}")
            
        print("\n✅ Verification Complete!")
        
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")

if __name__ == "__main__":
    verify_model()
