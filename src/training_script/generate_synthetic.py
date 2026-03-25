import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_DIR = "../../models/dagbani_to_english_model"
MONOLINGUAL_CORPUS = "../../data/raw/dagbani_corpus.txt"
OUTPUT_TSV_FILE = "../../data/raw/synthetic_dataset.tsv"
# How many sentences to process at once
BATCH_SIZE = 16

# --- Load Model and Tokenizer ---
print("--- Loading Dagbani-to-English Model ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is running on: {device.type.upper()}")

# --- Load Monolingual Data ---
print(f"\n--- Reading monolingual data from '{MONOLINGUAL_CORPUS}' ---")
try:
    with open(MONOLINGUAL_CORPUS, 'r', encoding='utf-8') as f:
        dagbani_sentences = [line.strip() for line in f if line.strip()]
    print(f"Found {len(dagbani_sentences)} sentences to translate.")
except FileNotFoundError:
    print(f"❌ Error: The file '{MONOLINGUAL_CORPUS}' was not found.")
    print("Please create this file and add your Dagbani sentences to it.")
    exit()

# --- Generate Synthetic English Translations ---
print("\n--- Generating synthetic English translations (this may take a while) ---")
synthetic_english_sentences = []

# Process in batches for efficiency
for i in tqdm(range(0, len(dagbani_sentences), BATCH_SIZE), desc="Translating batches"):
    batch = dagbani_sentences[i:i + BATCH_SIZE]
    
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    translated_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    synthetic_english_sentences.extend(translated_batch)

# --- Save the Results to a TSV file ---
print(f"\n--- Saving results to '{OUTPUT_TSV_FILE}' ---")
df = pd.DataFrame({
    'english': synthetic_english_sentences, # Synthetic English is the SOURCE
    'dagbani': dagbani_sentences          # Original Dagbani is the TARGET
})

# Make sure the columns are in the correct order for our final training script
df = df[['english', 'dagbani']]
df.to_csv(OUTPUT_TSV_FILE, sep='\t', index=False)

print("\n--- All Done! ---")
print(f"✅ Successfully created '{OUTPUT_TSV_FILE}' with {len(df)} translation pairs.")
print("You can now add this file to your main training dataset.")