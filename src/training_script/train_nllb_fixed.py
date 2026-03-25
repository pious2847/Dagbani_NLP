"""
train_nllb_fixed.py
-------------------
RESEARCH FIX: Properly fine-tunes NLLB-200-distilled-600M for Dagbani.

CRITICAL BUGS FIXED vs. original train_final_model.py:
  1. dag_Latn is NOT a native NLLB language code — this script injects it
     properly and initialises its embedding from mos_Latn (Mooré), which is
     the closest language in NLLB-200's Moore-Dagbani subgroup.
  2. Deprecated `tokenizer.as_target_tokenizer()` API replaced with the
     modern `text_target` parameter.
  3. ChrF++ added as PRIMARY evaluation metric (better than BLEU for
     morphologically rich low-resource languages with tonal diacritics).
  4. Cosine LR schedule + warmup to prevent early overfitting on small data.
  5. forced_bos_token_id correctly set during generation for Dagbani output.

Research basis:
  - David Dale / cointegrated — NLLB new-language tutorial
  - "Unlocking PEFT for Low-Resource Language Translation" (arxiv 2404.04212)
  - "Evaluating Extremely Low-Resource MT" (arxiv 2602.17425)
  - NLLB-200 HuggingFace discussion thread on dag_Latn absence

Usage:
  python train_nllb_fixed.py
  python train_nllb_fixed.py --data path/to/master.tsv --epochs 15
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_cosine_schedule_with_warmup,
)
import evaluate

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "../..")

DEFAULT_DATA   = os.path.join(ROOT, "data/processed/master_training_set.tsv")
OUTPUT_DIR     = os.path.join(ROOT, "models/nllb_dagbani_fixed")
PLOT_FILE      = os.path.join(ROOT, "experiments/training_graph_fixed.png")

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"

# dag_Latn is NOT in NLLB natively; we inject it from mos_Latn (Mooré).
# Mooré is in the same Moore-Dagbani subgroup (Niger-Congo > Gur > Oti-Volta)
# and is natively supported by NLLB-200 (mos_Latn token exists).
SRC_LANG   = "eng_Latn"
TGT_LANG   = "dag_Latn"   # will be injected below
PIVOT_LANG = "mos_Latn"   # Mooré — used to initialise dag_Latn embedding

# ── Training hyper-parameters ──────────────────────────────────────────────
LEARNING_RATE              = 2e-5
NUM_EPOCHS                 = 12
PER_DEVICE_BATCH_SIZE      = 1
GRADIENT_ACCUMULATION      = 16   # effective batch = 16
MAX_SEQ_LEN                = 80   # slightly longer than original 64
WARMUP_RATIO               = 0.1  # 10 % of total steps


# ══════════════════════════════════════════════════════════════════════════
#  TOKENIZER SURGERY — inject dag_Latn
# ══════════════════════════════════════════════════════════════════════════

def inject_dagbani_language_token(tokenizer, model):
    """
    dag_Latn does not exist in NLLB-200's vocabulary.  Naively passing
    tgt_lang='dag_Latn' silently fails (HuggingFace converts unknown lang
    codes to <unk>), meaning the model never learns to target Dagbani.

    Fix:
      1. Add dag_Latn as a new special token (preserve all existing tokens).
      2. Resize model embeddings to cover the new vocabulary.
      3. Initialise dag_Latn's embedding vector from mos_Latn (Mooré) —
         the structurally closest language that IS natively in NLLB-200.
         This gives the embedding a sensible starting point instead of a
         random init, which dramatically accelerates convergence on <2k pairs.

    Reference: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865
    """
    dag_token = "dag_Latn"

    # Check if already present (e.g. if someone ran the script twice)
    existing = tokenizer.additional_special_tokens
    if dag_token in existing:
        print(f"  ℹ️  {dag_token} already in tokenizer — skipping injection.")
        dag_id = tokenizer.convert_tokens_to_ids(dag_token)
        return dag_id

    print(f"  🔧 Injecting '{dag_token}' into NLLB tokenizer …")

    # IMPORTANT: replace_additional_special_tokens=False preserves the
    # existing 200 language tokens; setting it to True would wipe them.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [dag_token]},
        replace_additional_special_tokens=False,
    )

    # Resize model embedding table to match new vocab size
    old_vocab_size = model.model.shared.weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.model.shared.weight.shape[0]
    print(f"  Vocabulary: {old_vocab_size} → {new_vocab_size} tokens")

    dag_id = tokenizer.convert_tokens_to_ids(dag_token)
    mos_id = tokenizer.convert_tokens_to_ids(PIVOT_LANG)

    if mos_id == tokenizer.unk_token_id:
        print(f"  ⚠️  {PIVOT_LANG} not found — using random init for {dag_token}")
    else:
        print(f"  📐 Copying embedding: {PIVOT_LANG} (id={mos_id}) → {dag_token} (id={dag_id})")
        with torch.no_grad():
            # Copy Mooré embedding → Dagbani embedding (warm start)
            model.model.shared.weight[dag_id] = model.model.shared.weight[mos_id].clone()
            # Also update the lm_head if it has its own weight matrix
            if model.lm_head.weight.data_ptr() != model.model.shared.weight.data_ptr():
                model.lm_head.weight[dag_id] = model.lm_head.weight[mos_id].clone()

    print(f"  ✅ dag_Latn injected — token id = {dag_id}")
    return dag_id


# ══════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════

def load_data(data_file):
    print(f"\n── Loading data: {data_file}")
    if not os.path.exists(data_file):
        sys.exit(f"❌ Data file not found: {data_file}\n"
                 "   Run src/data_engineering/build_master_dataset.py first.")

    df = pd.read_csv(data_file, sep="\t", on_bad_lines="skip", comment="#",
                     encoding="utf-8")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["english", "dagbani"]].dropna()
    df["english"] = df["english"].astype(str).str.strip()
    df["dagbani"]  = df["dagbani"].astype(str).str.strip()
    df = df[(df["english"].str.len() >= 2) & (df["dagbani"].str.len() >= 2)]
    df = df.drop_duplicates(subset="english").reset_index(drop=True)

    print(f"   Pairs loaded: {len(df):,}")
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.1, seed=42)


# ══════════════════════════════════════════════════════════════════════════
#  TOKENISATION
# ══════════════════════════════════════════════════════════════════════════

def make_preprocess_fn(tokenizer, dag_id, max_len):
    """
    Returns a batched preprocessing function.

    FIX: Uses 'text_target' keyword argument instead of the deprecated
    'as_target_tokenizer()' context manager (removed in transformers ≥4.36).
    The forced target language token is set via tokenizer.lang_code_to_id
    or directly as forced_bos_token_id during generation.
    """
    def preprocess(examples):
        # Tokenise English source
        model_inputs = tokenizer(
            examples["english"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        # Tokenise Dagbani target using the modern API
        labels = tokenizer(
            text_target=examples["dagbani"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


# ══════════════════════════════════════════════════════════════════════════
#  METRICS — ChrF++ (primary) + BLEU (secondary)
# ══════════════════════════════════════════════════════════════════════════

def make_compute_metrics(tokenizer):
    """
    ChrF++ is the primary metric for Dagbani because:
      - Character n-gram matching handles tonal diacritics (ŋ, ɣ, ɛ, ɔ)
        correctly, where BLEU's word-level match would fail on near-matches.
      - Better correlation with human judgements for morphologically rich
        low-resource languages (arxiv 2602.17425).
    BLEU is retained as a secondary metric for comparison with prior work.
    """
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 padding in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # ChrF++ (word_order=2 → ChrF++, better than ChrF for this task)
        chrf_result = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            word_order=2,
        )

        # BLEU
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )

        return {
            "chrf++": round(chrf_result["score"], 4),
            "bleu":   round(bleu_result["score"],  4),
        }

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_history(log_history, filename):
    train_steps, train_loss = [], []
    eval_epochs, eval_chrf, eval_bleu = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_loss.append(entry["loss"])
        if "eval_chrf++" in entry:
            eval_epochs.append(entry.get("epoch", len(eval_epochs) + 1))
            eval_chrf.append(entry["eval_chrf++"])
            eval_bleu.append(entry.get("eval_bleu", 0))

    if not eval_epochs:
        print("⚠️  Not enough eval data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(train_steps, train_loss, color="steelblue", label="Train Loss")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss"); axes[0].grid(True); axes[0].legend()

    # ChrF++ and BLEU
    axes[1].plot(eval_epochs, eval_chrf, "o-", color="green",  label="ChrF++ (primary)")
    axes[1].plot(eval_epochs, eval_bleu, "s--", color="orange", label="BLEU (secondary)")
    axes[1].set_title("Eval Metrics vs Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score"); axes[1].grid(True); axes[1].legend()

    plt.suptitle("NLLB-Fixed: Dagbani Translation Training", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=120)
    print(f"✅ Training graph saved → {filename}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train fixed NLLB-200 for Dagbani")
    p.add_argument("--data",   default=DEFAULT_DATA, help="TSV training file")
    p.add_argument("--output", default=OUTPUT_DIR,   help="Model output directory")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs")
    p.add_argument("--lr",     type=float, default=LEARNING_RATE)
    p.add_argument("--max_len",type=int, default=MAX_SEQ_LEN)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  NLLB-Fixed Dagbani Trainer")
    print("=" * 60)

    # GPU check
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU — training will be very slow on CPU")

    # ── Load tokenizer and model ───────────────────────────────────────────
    print(f"\n── Loading {MODEL_CHECKPOINT} …")
    # Initialise tokenizer in English source mode (we manually manage target)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang=SRC_LANG)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # ── CRITICAL: inject dag_Latn ──────────────────────────────────────────
    dag_token_id = inject_dagbani_language_token(tokenizer, model)

    # Set forced_bos_token_id so generation always targets Dagbani
    model.config.forced_bos_token_id = dag_token_id

    # ── Data ──────────────────────────────────────────────────────────────
    splits = load_data(args.data)

    preprocess_fn = make_preprocess_fn(tokenizer, dag_token_id, args.max_len)
    tokenized = splits.map(
        preprocess_fn, batched=True,
        remove_columns=splits["train"].column_names,
        desc="Tokenising",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=-100)

    # ── Metrics ───────────────────────────────────────────────────────────
    compute_metrics = make_compute_metrics(tokenizer)

    # ── Training arguments ────────────────────────────────────────────────
    total_steps = (
        len(tokenized["train"]) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION)
    ) * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,

        # Evaluation & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_chrf++",   # PRIMARY metric
        greater_is_better=True,

        # Batch / gradient
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        # Learning rate with cosine decay + warmup
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,

        # Memory optimisations (4 GB GPU friendly)
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adafactor",

        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_len,

        # Logging
        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n── Training for {args.epochs} epochs …")
    print(f"   Warmup steps : {warmup_steps}")
    print(f"   Total steps  : {total_steps}")
    torch.cuda.empty_cache()
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n✅ Model saved → {args.output}")

    # Save the dag_Latn token id in the config for inference
    import json
    config_extra = {"dag_latn_token_id": dag_token_id, "src_lang": SRC_LANG, "tgt_lang": TGT_LANG}
    with open(os.path.join(args.output, "dagbani_config.json"), "w") as f:
        json.dump(config_extra, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_history(trainer.state.log_history, PLOT_FILE)

    # ── Final eval summary ────────────────────────────────────────────────
    print("\n── Final evaluation metrics:")
    for entry in reversed(trainer.state.log_history):
        if "eval_chrf++" in entry:
            print(f"   ChrF++ : {entry['eval_chrf++']:.2f}")
            print(f"   BLEU   : {entry.get('eval_bleu', 'N/A')}")
            break

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("  Next step: run src/training_script/translate_nllb_fixed.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
