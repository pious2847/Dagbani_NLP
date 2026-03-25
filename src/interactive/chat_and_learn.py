import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../models/final_dagbani_nllb")
FEEDBACK_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/user_feedback.tsv")

def load_model():
    print("--- Waking up the Baby AI... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang="eng_Latn", tgt_lang="dag_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"✅ AI is ready! (Brain: {device})")
        return tokenizer, model, device
    except Exception as e:
        print(f"❌ Could not load model from {MODEL_PATH}")
        print("   (Have you run the training script yet?)")
        print(f"   Error: {e}")
        return None, None, None

def translate(text, tokenizer, model, device, direction="en2dag"):
    if direction == "en2dag":
        tokenizer.src_lang = "eng_Latn"
        target_lang_code = "dag_Latn" # NLLB code for Dagbani
    else:
        tokenizer.src_lang = "dag_Latn"
        target_lang_code = "eng_Latn"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Force the target language token
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    
    outputs = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id, 
        max_length=128
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def save_feedback(english, dagbani):
    """Saves the corrected pair to a file."""
    new_data = pd.DataFrame({'english': [english], 'dagbani': [dagbani]})
    
    # Append to feedback file
    if not os.path.exists(FEEDBACK_FILE):
        new_data.to_csv(FEEDBACK_FILE, sep='\t', index=False)
    else:
        new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, sep='\t', index=False)
    
    print("✅ Learned! (Saved to memory)")

def main():
    tokenizer, model, device = load_model()
    if not tokenizer:
        return

    print("\n👶 Baby AI: Hello! I am learning Dagbani. Talk to me!")
    print("   Type 'exit' to stop.")
    print("   Type 'switch' to change direction (default: English -> Dagbani)")
    
    direction = "en2dag"
    
    while True:
        prompt = "English: " if direction == "en2dag" else "Dagbani: "
        user_input = input(f"\n{prompt}").strip()
        
        if user_input.lower() == 'exit':
            print("Baby AI: Bye bye! 👋")
            break
        
        if user_input.lower() == 'switch':
            direction = "dag2en" if direction == "en2dag" else "en2dag"
            print(f"🔄 Switched to {direction}")
            continue
            
        # Translate
        translation = translate(user_input, tokenizer, model, device, direction)
        print(f"Baby AI: {translation}")
        
        # Feedback Loop
        feedback = input("   Is this correct? (y/n/correction): ").strip()
        
        if feedback.lower() == 'y':
            print("   Baby AI: Yay! 😊")
        elif feedback.lower() == 'n':
            print("   Baby AI: Oh no... 😢")
        else:
            # Assume it's a correction
            correction = feedback
            print(f"   Baby AI: Oh, I see! '{user_input}' means '{correction}'.")
            
            if direction == "en2dag":
                save_feedback(user_input, correction)
            else:
                save_feedback(correction, user_input)

if __name__ == "__main__":
    main()
