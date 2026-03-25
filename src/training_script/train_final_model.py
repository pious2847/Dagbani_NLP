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

# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- File Paths ---
# Resolve paths relative to the script location
DATA_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/final_training_set.tsv")
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../../models/final_dagbani_nllb")
PLOT_FILENAME = os.path.join(SCRIPT_DIR, "../../experiments/training_graph_nllb.png")

# --- Model & Tokenizer ---
# Switch to NLLB-200 (Distilled 600M is a good balance of speed/performance)
MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"

# --- Training Parameters ---
LEARNING_RATE = 2e-5 # Slightly lower for fine-tuning a larger model
NUM_EPOCHS = 10
# Reduced batch size for 4GB GPU
PER_DEVICE_BATCH_SIZE = 1 
# Increased accumulation to maintain effective batch size
GRADIENT_ACCUMULATION_STEPS = 16 

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

def load_and_prepare_data(data_file):
    """
    Loads the pre-built curriculum dataset.
    """
    print(f"\n--- 1. Loading Curriculum Data from {data_file} ---")
    
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found at '{data_file}'")
        sys.exit()

    df = pd.read_csv(data_file, sep='\t', on_bad_lines='skip')
    
    # Ensure strings
    df['english'] = df['english'].astype(str).str.strip()
    df['dagbani'] = df['dagbani'].astype(str).str.strip()
    
    print(f"Loaded {len(df)} training examples.")
    
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset.train_test_split(test_size=0.1, seed=42)


def plot_training_history(history, filename):
    """Plots and saves the training and validation loss."""
    print("\n--- 5. Generating Training Graph ---")
    
    train_loss = [item['loss'] for item in history if 'loss' in item]
    # Filter eval_loss to match training steps roughly or just plot available points
    eval_loss = [item['eval_loss'] for item in history if 'eval_loss' in item]
    epochs = [item['epoch'] for item in history if 'eval_loss' in item]
    
    if not epochs or not eval_loss:
        print("⚠️ Warning: Not enough data to plot.")
        return

    plt.figure(figsize=(10, 6))
    # Note: train_loss might be more frequent than eval_loss, so we plot what we have
    plt.plot(train_loss, label='Training Loss (Steps)')
    # We can't easily align epochs on x-axis if lengths differ without more logic, 
    # so simple plot is safer for now or just plot eval
    # plt.plot(epochs, eval_loss, 'r-o', label='Validation Loss')
    
    plt.title('NLLB Model - Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)
    print(f"✅ Training graph saved as '{filename}'")


def main():
    """Main function to run the entire training pipeline."""
    
    check_gpu()
    split_datasets = load_and_prepare_data(DATA_FILE)
    
    print("\n--- 2. Initializing NLLB Model & Tokenizer ---")
    # NLLB requires specifying source and target languages
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang="eng_Latn", tgt_lang="dag_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["dagbani"]
        
        # Tokenize inputs (English) - Reduced max_length to 64 to save memory
        model_inputs = tokenizer(inputs, max_length=64, truncation=True)
        
        # Tokenize targets (Dagbani)
        # We must set the tokenizer to target language mode for labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=64, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
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
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
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
        gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        optim="adafactor", # Use Adafactor to save memory
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