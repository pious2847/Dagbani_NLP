# This file is named train_intermediate_model.py

import os
import sys
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# --- CONFIGURATION ---
DATA_FOLDER = "../data/datasets_synthetic_only"
MODEL_OUTPUT_DIR = "../models/intermediate_dagbani_translator"
MODEL_CHECKPOINT = "../models/model" # Start from the original base model

# --- Main Training Logic (simplified as helper functions are the same) ---
def load_data(data_folder):
    if not os.path.exists(data_folder):
        print(f"❌ Error: Data folder '{data_folder}' not found.")
        print("Please create it and place 'synthetic_dataset.tsv' inside.")
        sys.exit()
    
    file_path = os.path.join(data_folder, 'synthetic_dataset.tsv')
    if not os.path.exists(file_path):
        print(f"❌ Error: 'synthetic_dataset.tsv' not found in '{data_folder}'.")
        sys.exit()

    df = pd.read_csv(file_path, sep='\t', on_bad_lines='warn')
    df.rename(columns={'english': 'english', 'dagbani': 'dagbani'}, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    print(f"Loaded {len(df)} unique synthetic examples for intermediate training.")
    return Dataset.from_pandas(df).train_test_split(test_size=0.05, seed=42) # Use a small test set

def main():
    # --- Setup ---
    print("--- STAGE 1: INTERMEDIATE TRAINING ON SYNTHETIC DATA ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # --- Data Loading and Processing ---
    split_datasets = load_data(DATA_FOLDER)
    
    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["dagbani"]
        return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)

    tokenized_datasets = split_datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --- Training ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5, # 5 epochs is usually enough for this stage
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
    )
    
    trainer.train()
    trainer.save_model()
    print(f"\n✅ Intermediate model saved to '{MODEL_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()