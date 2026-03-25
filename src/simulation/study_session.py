import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset
import evaluate

# Add src to path to import pdf_processor
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_engineering.pdf_processor import PDFProcessor

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../models/final_dagbani_nllb")
PDF_DIR = os.path.join(SCRIPT_DIR, "../../data/pdf")
LESSON_SIZE = 50 # Number of words to learn per "lesson"

def load_model():
    print("--- Loading Brain ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang="eng_Latn", tgt_lang="dag_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

def run_lesson(lesson_id, df_lesson, tokenizer, model):
    print(f"\n📚 Starting Lesson {lesson_id}: {len(df_lesson)} new words.")
    
    # 1. Prepare Data
    hf_dataset = Dataset.from_pandas(df_lesson)
    # Split for training (no validation set for this small batch simulation, we just want to learn)
    # Actually, let's just train on all of it to "memorize" the dictionary
    
    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["dagbani"]
        model_inputs = tokenizer(inputs, max_length=64, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 2. Configure Training (Lightweight)
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODEL_PATH, f"lesson_{lesson_id}"),
        learning_rate=2e-5,
        per_device_train_batch_size=1, # Keep it small for safety
        gradient_accumulation_steps=4,
        num_train_epochs=3, # Quick study
        fp16=True,
        logging_steps=10,
        save_strategy="no", # Don't save every checkpoint
        optim="adafactor",
        gradient_checkpointing=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 3. Study!
    print("   🧠 Studying...")
    trainer.train()
    
    # 4. Save Knowledge
    print("   💾 Consolidating memory...")
    trainer.save_model(MODEL_PATH) # Overwrite main model with new knowledge
    
    # Cleanup
    del trainer
    torch.cuda.empty_cache()
    print(f"✅ Lesson {lesson_id} Complete!")

def main():
    # 1. Load Cleaned Data
    data_file = os.path.join(SCRIPT_DIR, "../../data/processed/pdf_vocab_clean.tsv")
    
    if not os.path.exists(data_file):
        print(f"❌ Cleaned data file not found: {data_file}")
        print("   Please run src/data_engineering/clean_pdf_vocab.py first.")
        return
    
    print(f"📖 Reading textbook data from: {data_file}")
    df = pd.read_csv(data_file, sep='\t')
    print(f"Total words found: {len(df)}")
    
    # 3. Start Simulation Loop
    tokenizer, model = load_model()
    
    # Split into lessons
    num_lessons = (len(df) // LESSON_SIZE) + 1
    
    for i in range(num_lessons):
        start_idx = i * LESSON_SIZE
        end_idx = min((i + 1) * LESSON_SIZE, len(df))
        
        if start_idx >= len(df):
            break
            
        lesson_df = df.iloc[start_idx:end_idx]
        run_lesson(i + 1, lesson_df, tokenizer, model)
        
    print("\n🎓 Graduation! The AI has finished the textbook.")

if __name__ == "__main__":
    main()
