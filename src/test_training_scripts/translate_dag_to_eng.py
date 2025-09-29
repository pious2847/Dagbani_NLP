import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIGURATION ---
MODEL_DIR = "../models/dagbani_to_english_model"

# --- Load Tokenizer and Model ---
print("--- Loading Model & Tokenizer ---")
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

# --- List of Dagbani sentences to translate ---
dagbani_sentences = [
    "dasiba",
    "ka di be wula?",
    "n yura",
    "naawuni su jema",
    "ti mali nangbanyini",
    "fara deei zaa",
    "Guuimi ti boli karimba maa",
    "O chagya zugo",
    "bia",
    "bihi"
]

# --- Translation ---
print("\n--- Translating Sentences ---")

for sentence in dagbani_sentences:
    # 1. Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # 2. Generate the translation (output token IDs)
    output_ids = model.generate(
        **inputs,
        max_length=50,       # Max length of the generated sentence
        num_beams=5,         # Beam search for better quality
        early_stopping=True
    )

    # 3. Decode the token IDs back to a string
    translated_sentence = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 4. Print the result
    print(f"Dagbani: {sentence}")
    print(f"English: {translated_sentence}")
    print("-" * 20)