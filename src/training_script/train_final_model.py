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

import evaluate

# --- 1. CONFIGURATION ---

# --- File Paths ---
DATA_FOLDER = "../../data/datasets_combined"
MODEL_OUTPUT_DIR = "../../models/final_dagbani_translator"
PLOT_FILENAME = "../../experiments/training_graph_final.png"

# --- Model & Tokenizer ---
MODEL_CHECKPOINT = "../../models/model"

# --- Training Parameters ---
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
        return True
    else:
        print("⚠️ Warning: No GPU detected. Training will run on the CPU (very slow).")
        return False

# **** NEW AND IMPROVED DATA LOADING FUNCTION ****
def load_and_prepare_data(data_folder):
    """
    Loads data, intelligently combining high-quality and synthetic datasets
    to avoid data collision and maximize unique examples.
    """
    print("\n--- 1. Loading & Preparing All Combined Data (Smart Method) ---")
    
    if not os.path.exists(data_folder):
        print(f"❌ Error: Data folder not found at '{data_folder}'")
        sys.exit()

    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.tsv')]
    if not all_files:
        print(f"❌ Error: No .tsv files found in '{data_folder}'")
        sys.exit()

    # Separate high-quality (HQ) files from the synthetic file
    hq_files = [f for f in all_files if 'synthetic_dataset' not in f]
    synthetic_file = [f for f in all_files if 'synthetic_dataset' in f]

    if not hq_files:
        print("❌ Error: No high-quality dataset files found.")
        sys.exit()
    if not synthetic_file:
        print("⚠️ Warning: No synthetic dataset file found. Proceeding with HQ data only.")
        synthetic_file = None
    else:
        synthetic_file = synthetic_file[0]

    # --- Step 1: Load and clean the High-Quality (HQ) data ---
    print(f"Found {len(hq_files)} high-quality dataset files.")
    hq_df_list = [pd.read_csv(file, sep='\t', on_bad_lines='warn', header=0) for file in tqdm(hq_files, desc="Reading HQ files")]
    hq_df = pd.concat(hq_df_list, ignore_index=True)
    hq_df.rename(columns={'English': 'english', 'Dagbani': 'dagbani'}, inplace=True, errors='ignore')
    hq_df['english'] = hq_df['english'].astype(str).str.lower().str.strip()
    hq_df['dagbani'] = hq_df['dagbani'].astype(str).str.lower().str.strip()
    hq_df.dropna(inplace=True)
    hq_df.drop_duplicates(inplace=True)
    print(f"Found {len(hq_df)} unique examples in high-quality datasets.")

    # --- Step 2: Load and filter the synthetic data ---
    if synthetic_file:
        synth_df = pd.read_csv(synthetic_file, sep='\t', on_bad_lines='warn', header=0)
        synth_df.rename(columns={'english': 'english', 'dagbani': 'dagbani'}, inplace=True, errors='ignore')
        synth_df['english'] = synth_df['english'].astype(str).str.lower().str.strip()
        synth_df['dagbani'] = synth_df['dagbani'].astype(str).str.lower().str.strip()
        synth_df.dropna(inplace=True)
        synth_df.drop_duplicates(inplace=True)
        print(f"Loaded {len(synth_df)} unique examples from synthetic dataset.")

        # This is the CRUCIAL step: keep only synthetic examples for NEW Dagbani sentences
        existing_dagbani = set(hq_df['dagbani'])
        synth_df = synth_df[~synth_df['dagbani'].isin(existing_dagbani)]
        print(f"Kept {len(synth_df)} synthetic examples for novel Dagbani sentences.")

        # --- Step 3: Combine the HQ data and the filtered synthetic data ---
        full_df = pd.concat([hq_df, synth_df], ignore_index=True)
    else:
        full_df = hq_df

    print(f"\nTotal unique examples for training: {len(full_df)}")

    hf_dataset = Dataset.from_pandas(full_df)
    return hf_dataset.train_test_split(test_size=0.1, seed=42)


def plot_training_history(history, filename):
    """Plots and saves the training and validation loss."""
    print("\n--- 5. Generating Training Graph ---")
    
    train_loss = [item['loss'] for item in history if 'loss' in item]
    eval_loss = [item['eval_loss'] for item in history if 'eval_loss' in item]
    epochs = [item['epoch'] for item in history if 'eval_loss' in item]
    
    if not epochs or not eval_loss:
        print("⚠️ Warning: Not enough data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss[:len(epochs)], 'b-o', label='Training Loss')
    plt.plot(epochs, eval_loss, 'r-o', label='Validation Loss')
    plt.title('Final Model - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)
    print(f"✅ Training graph saved as '{filename}'")


def main():
    """Main function to run the entire training pipeline."""
    
    check_gpu()
    split_datasets = load_and_prepare_data(DATA_FOLDER)
    
    print("\n--- 2. Initializing Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["dagbani"]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_datasets = split_datasets.map(preprocess_function, batched=True, desc="Tokenizing datasets")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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

    print("\n--- 3. Configuring Final Training ---")
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
    
    print("\n--- 4. Starting Final Training ---")
    trainer.train()

    trainer.save_model()
    print(f"\n✅ Best model saved to '{MODEL_OUTPUT_DIR}'")
    
    plot_training_history(trainer.state.log_history, PLOT_FILENAME)
    
    print("\n--- FINAL MODEL TRAINING COMPLETE! ---")

if __name__ == "__main__":
    main()