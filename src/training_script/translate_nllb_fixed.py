"""
translate_nllb_fixed.py
-----------------------
Inference and evaluation script for the fixed NLLB-200 Dagbani model.

FIXES vs original translate_final.py:
  1. Loads the dag_Latn token id from dagbani_config.json saved during training.
  2. Sets forced_bos_token_id correctly so the decoder targets Dagbani.
  3. Reports ChrF++ (primary) and BLEU (secondary) on a held-out test set.
  4. Interactive translation mode for live testing.

USAGE:
  # Evaluate on built-in test sentences
  python translate_nllb_fixed.py

  # Evaluate on a TSV file (english\tdagbani)
  python translate_nllb_fixed.py --eval data/processed/master_training_set.tsv

  # Interactive translation mode
  python translate_nllb_fixed.py --interactive

  # Use a different model directory
  python translate_nllb_fixed.py --model models/nllb_dagbani_fixed
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "../..")

DEFAULT_MODEL_DIR = os.path.join(ROOT, "models/nllb_dagbani_fixed")

# ── Built-in test sentences covering different complexity levels ───────────
BUILTIN_TEST = [
    # (english, reference_dagbani)
    ("how are you?",                    "ali bɛ?"),
    ("thank you",                       "naa"),
    ("my name is John",                 "n yuli n nyela John"),
    ("good morning",                    "antire"),
    ("we are eating",                   "ti dirila"),
    ("sit down please",                 "guŋ fu"),
    ("where is the water?",             "nii bɛ ka?"),
    ("the car is blue",                 "loori ba bluu."),
    ("we are united",                   "ti pala mala"),
    ("love your father and mother",     "di yira, di malim a baa ne a mali."),
    ("this is my child",                "wɔ n bia"),
    ("I am going to the market",        "n bɛ wuli luŋa"),
    ("where is the market?",            "luŋa bɛ ka?"),
    ("the sun is hot",                  "cham ba zaɣim"),
    ("my friend is coming today",       "n doo bɛ waa timiya"),
]

SRC_LANG = "eng_Latn"
TGT_LANG = "dag_Latn"


def load_model_and_tokenizer(model_dir: str):
    """Load the fixed model, tokenizer, and recover the dag_Latn token id."""
    print(f"── Loading model from: {model_dir}")

    if not os.path.exists(model_dir):
        sys.exit(f"❌ Model directory not found: {model_dir}\n"
                 "   Train first with: python train_nllb_fixed.py")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=SRC_LANG)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"   Device: {device.type.upper()}")

    # Recover dag_Latn token id from saved config
    config_path = os.path.join(model_dir, "dagbani_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        dag_id = cfg["dag_latn_token_id"]
        print(f"   dag_Latn token id: {dag_id}")
    else:
        # Fallback: look up from tokenizer
        dag_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
        if dag_id == tokenizer.unk_token_id:
            sys.exit(
                "❌ dag_Latn token not found in tokenizer.\n"
                "   This model was probably trained with the OLD broken script.\n"
                "   Re-train using train_nllb_fixed.py."
            )
        print(f"   dag_Latn token id (recovered): {dag_id}")

    model.config.forced_bos_token_id = dag_id
    return tokenizer, model, device, dag_id


def translate(text: str, tokenizer, model, device, dag_id: int,
              num_beams: int = 5, max_length: int = 80) -> str:
    """Translate a single English string to Dagbani."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=80).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=dag_id,   # ← critical: target Dagbani
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_on_pairs(pairs: list[tuple], tokenizer, model, device, dag_id):
    """
    Compute ChrF++ and BLEU on (english, reference_dagbani) pairs.

    ChrF++ is reported as the primary metric because:
      - It matches at character level → handles tonal diacritics (ŋ, ɣ, ɛ, ɔ)
      - Better human correlation for morphologically rich low-resource languages
    BLEU is secondary for cross-paper comparisons.
    """
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    predictions, references = [], []
    print(f"\n{'English':<40} {'Reference Dagbani':<30} {'Model Output':<30}")
    print("-" * 100)

    for eng, ref in pairs:
        pred = translate(eng, tokenizer, model, device, dag_id)
        predictions.append(pred)
        references.append(ref)
        print(f"{eng:<40} {ref:<30} {pred:<30}")

    chrf_score = chrf_metric.compute(
        predictions=predictions,
        references=references,
        word_order=2,
    )["score"]

    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )["score"]

    print("\n" + "=" * 60)
    print(f"  ChrF++ (primary) : {chrf_score:.2f}")
    print(f"  BLEU   (secondary): {bleu_score:.2f}")
    print("=" * 60)

    # Diagnostic: warn if metrics diverge significantly
    if abs(chrf_score - bleu_score) > 10:
        print("\n⚠️  WARNING: ChrF++ and BLEU diverge by >10 points.")
        if chrf_score > bleu_score + 10:
            print("   → Possible source copying or word-order issues.")
        else:
            print("   → Possible character-level hallucination.")
        print("   Review outputs manually.")

    return chrf_score, bleu_score


def interactive_mode(tokenizer, model, device, dag_id):
    """REPL for live translation testing."""
    print("\n── Interactive Translation Mode (English → Dagbani)")
    print("   Type an English sentence and press Enter.")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("English: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        dagbani = translate(text, tokenizer, model, device, dag_id)
        print(f"Dagbani: {dagbani}\n")


def main():
    parser = argparse.ArgumentParser(description="Translate/evaluate fixed NLLB Dagbani model")
    parser.add_argument("--model",       default=DEFAULT_MODEL_DIR, help="Model directory")
    parser.add_argument("--eval",        default=None,
                        help="TSV file (english\\tdagbani) to evaluate on")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive translation REPL")
    parser.add_argument("--num_beams",   type=int, default=5)
    args = parser.parse_args()

    tokenizer, model, device, dag_id = load_model_and_tokenizer(args.model)

    if args.interactive:
        interactive_mode(tokenizer, model, device, dag_id)
        return

    if args.eval:
        # Evaluate on a TSV file
        print(f"\n── Evaluating on: {args.eval}")
        df = pd.read_csv(args.eval, sep="\t", on_bad_lines="skip", comment="#",
                         encoding="utf-8")
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[["english", "dagbani"]].dropna().head(200)  # cap at 200 for speed
        pairs = list(zip(df["english"].astype(str), df["dagbani"].astype(str)))
        evaluate_on_pairs(pairs, tokenizer, model, device, dag_id)
    else:
        # Use built-in test set
        print("\n── Evaluating on built-in test sentences")
        evaluate_on_pairs(BUILTIN_TEST, tokenizer, model, device, dag_id)


if __name__ == "__main__":
    main()
