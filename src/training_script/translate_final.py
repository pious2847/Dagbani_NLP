import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIGURATION ---
# Path to the FINAL, most powerful model you just trained
MODEL_DIR = "../../models/gentle_finetuned_translator"

# --- Load Tokenizer and Model ---
print("--- Loading Final Translator Model ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the path in MODEL_DIR is correct.")
    exit()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is running on: {device.type.upper()}")

# --- List of English sentences to translate ---
english_sentences = [
    # --- Simple phrases from your data ---
    "how are you?",
    "thank you",
    "i love you",
    "what is your name?",
    # --- Full sentences from your data ---
    "we are united",
    "love your father and mother.",
    "the car is blue.",
    # --- NEW sentences to test generalization ---
    "where is the market?",
    "the sun is hot",
    "my friend is coming today",
    "this is my child",
    "we want to eat food"
]

# --- Translation ---
print("\n--- Translating Sentences (English -> Dagbani) ---")

for sentence in english_sentences:
    # 1. Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # 2. Generate the translation (output token IDs)
    output_ids = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )

    # 3. Decode the token IDs back to a string
    translated_sentence = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 4. Print the result
    print(f"English: {sentence}")
    print(f"Dagbani: {translated_sentence}")
    print("-" * 20)