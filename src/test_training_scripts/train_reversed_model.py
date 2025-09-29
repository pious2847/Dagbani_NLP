import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# NEW: Import the 'evaluate' library
import evaluate

# --- 1. CONFIGURATION ---
# All your settings are here for easy access

# --- File Paths ---
DATA_FOLDER = "../data/datasets_reversed"
MODEL_OUTPUT_DIR = "../models/dagbani_to_english_model"
PLOT_FILENAME = "../experiments/training_graph_reversed.png"

# --- Model & Tokenizer ---
MODEL_CHECKPOINT = "../models/model"

# --- Training Parameters (TUNED FOR LOW VRAM GPUS) ---
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# --- Helper Functions ---

def check_gpu():
    """Checks for GPU availability and prints status."""
    print("--- Hardware Check ---")
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print("Training will run on the GPU.")
        return True
    else:
        print("⚠️ Warning: No GPU detected. Training will run on the CPU (very slow).")
        return False

def load_and_prepare_data(data_folder):
    """Loads, cleans, and prepares the dataset from multiple files."""
    print("\n--- 1. Loading & Preparing Data ---")
    
    if not os.path.exists(data_folder):
        print(f"❌ Error: Data folder not found at '{data_folder}'")
        sys.exit()

    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.tsv')]
    if not all_files:
        print(f"❌ Error: No .tsv files found in '{data_folder}'")
        sys.exit()

    print(f"Found {len(all_files)} dataset files.")
    
    # Use tqdm for a progress bar during file reading
    df_list = [pd.read_csv(file, sep='\t', on_bad_lines='warn') for file in tqdm(all_files, desc="Reading files")]
    full_df = pd.concat(df_list, ignore_index=True)

    # Ensure column names are consistent ('english', 'dagbani')
    full_df.columns = ['dagbani', 'english']
    
    # Cleaning
    full_df['english'] = full_df['english'].astype(str).str.lower().str.strip()
    full_df['dagbani'] = full_df['dagbani'].astype(str).str.lower().str.strip()

    original_rows = len(full_df)
    full_df.dropna(inplace=True)
    full_df.drop_duplicates(inplace=True)
    final_rows = len(full_df)

    print(f"Loaded {original_rows} rows. After cleaning, {final_rows} unique examples remain.")
    
    # Convert pandas DataFrame to Hugging Face Dataset object
    hf_dataset = Dataset.from_pandas(full_df)
    return hf_dataset.train_test_split(test_size=0.1, seed=42)

def plot_training_history(history, filename):
    """Plots and saves the training and validation loss."""
    print("\n--- 5. Generating Training Graph ---")
    
    train_loss = [item['loss'] for item in history if 'loss' in item]
    eval_loss = [item['eval_loss'] for item in history if 'eval_loss' in item]
    epochs = [item['epoch'] for item in history if 'eval_loss' in item]
    
    if not epochs or not eval_loss:
        print("⚠️ Warning: Not enough data in history to plot a graph.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss[:len(epochs)], 'b-o', label='Training Loss')
    plt.plot(epochs, eval_loss, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)
    print(f"✅ Training graph saved as '{filename}'")


def main():
    """Main function to run the entire training pipeline."""
    
    check_gpu()
    
    # --- Step 1: Load Data ---
    split_datasets = load_and_prepare_data(DATA_FOLDER)
    
    # --- Step 2: Initialize Tokenizer and Model ---
    print("\n--- 2. Initializing Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # ** CRITICAL CHANGE FOR REVERSE MODEL **
    # The source language is now 'dagbani', and the target is 'english'
    def preprocess_function(examples):
        inputs = examples["dagbani"]   # Source is Dagbani
        targets = examples["english"] # Target is English
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_datasets = split_datasets.map(preprocess_function, batched=True, desc="Tokenizing datasets")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --- Step 3: Define Metrics for Analytics ---
    bleu_metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        return {"bleu": result["score"]}

    # --- Step 4: Configure and Run Training ---
    print("\n--- 3. Configuring Training ---")
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        fp16=True,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n--- 4. Starting Training ---")
    trainer.train()

    # --- Step 5: Save Final Model and Generate Plot ---
    trainer.save_model()
    print(f"\n✅ Best model saved to '{MODEL_OUTPUT_DIR}'")
    
    plot_training_history(trainer.state.log_history, PLOT_FILENAME)
    
    print("\n--- All Done! ---")
    print(f"You now have a Dagbani-to-English model saved in the '{MODEL_OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()