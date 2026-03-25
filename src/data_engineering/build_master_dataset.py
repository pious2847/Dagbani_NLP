"""
build_master_dataset.py
-----------------------
Merges ALL available Dagbani-English data sources, cleans noise, deduplicates,
and writes a single master TSV used by the fixed training pipeline.

Output: data/processed/master_training_set.tsv  (english \t dagbani)
"""

import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "../..")

# ── Source files to merge ──────────────────────────────────────────────────
# Each entry is (path, english_col, dagbani_col)
# Paths relative to ROOT
SOURCES = [
    # Core processed datasets
    ("data/processed/final_training_set.tsv",        "english", "dagbani"),
    ("data/processed/final_training_set_refined.tsv","english", "dagbani"),
    ("data/processed/dagbani_english_dataset_final.tsv","english","dagbani"),
    ("data/processed/dagbani_general_vocabs.tsv",    "english", "dagbani"),
    ("data/processed/pdf_vocab_clean.tsv",           "english", "dagbani"),
    ("data/processed/dagbani_enaglish_numbers.tsv",  "english", "dagbani"),
    ("data/processed/dagbani_enaglish_colors.tsv",   "english", "dagbani"),
    ("data/processed/dagbani_english_idoms.tsv",     "english", "dagbani"),
    ("data/processed/stage_1_vocab.tsv",             "english", "dagbani"),
    ("data/processed/stage_2_baby.tsv",              "english", "dagbani"),
    ("data/processed/stage_3_corpus.tsv",            "english", "dagbani"),
    ("data/processed/user_feedback.tsv",             "english", "dagbani"),
    # Root-level duplicates (may have unique rows)
    ("data/dagbani_english_dataset_final.tsv",       "english", "dagbani"),
    ("data/dagbani_english_idoms.tsv",               "english", "dagbani"),
    ("data/dagbani_enaglish_numbers.tsv",             "english", "dagbani"),
    ("data/dagbani_enaglish_colors.tsv",             "english", "dagbani"),
    # New Bryn Mawr scrape (336 high-quality pairs)
    ("data/raw/scraped_brynmawr_data.tsv",           "english", "dagbani"),
    # High-quality combined master (if exists)
    ("data/datasets_combined/hq_dataset_master.tsv", "english", "dagbani"),
]

OUTPUT_FILE = os.path.join(ROOT, "data/processed/master_training_set.tsv")

# ── Noise patterns to strip ────────────────────────────────────────────────
# Metadata markers found in several files
NOISE_PATTERNS = [
    r"\bKO\b",                    # "KO" annotation marker
    r"\[.*?\]",                   # anything in brackets  e.g. [note]
    r"\(Hausa\)",                 # language-source annotations
    r"\(lit\..*?\)",              # literal-meaning notes
]

# Rows with these English-side values are metadata, not translation pairs
BLACKLIST_ENGLISH = {
    "english", "source", "translation", "note", "nan", "none", "",
}


def clean_text(text: str) -> str:
    """Strip noise patterns and normalise whitespace."""
    text = str(text).strip()
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # Normalise multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Normalise curly / smart quotes to straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def is_valid_row(eng: str, dag: str) -> bool:
    """Return True if this row is a genuine translation pair."""
    eng_l = eng.lower().strip()
    dag_l = dag.lower().strip()

    # Skip metadata / header rows
    if eng_l in BLACKLIST_ENGLISH or dag_l in BLACKLIST_ENGLISH:
        return False

    # Skip if either side is too short (single char noise)
    if len(eng_l) < 2 or len(dag_l) < 2:
        return False

    # Skip if both sides are identical (untranslated)
    if eng_l == dag_l:
        return False

    # Skip if Dagbani side is purely ASCII digits (bad parse)
    if re.fullmatch(r"[\d\s]+", dag_l):
        return False

    return True


def load_source(path: str, eng_col: str, dag_col: str) -> pd.DataFrame:
    """Load a single TSV source, normalise columns, skip comment lines."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["english", "dagbani"])

    # Skip lines starting with '#' (comment headers in brynmawr file)
    try:
        df = pd.read_csv(
            path, sep="\t", on_bad_lines="skip",
            comment="#", encoding="utf-8"
        )
    except Exception as e:
        print(f"  ⚠️  Could not read {path}: {e}")
        return pd.DataFrame(columns=["english", "dagbani"])

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    missing = [c for c in [eng_col, dag_col] if c not in df.columns]
    if missing:
        print(f"  ⚠️  Skipping {path} — missing columns: {missing}")
        return pd.DataFrame(columns=["english", "dagbani"])

    df = df[[eng_col, dag_col]].copy()
    df.columns = ["english", "dagbani"]
    return df


def main():
    print("=" * 60)
    print("  Dagbani Master Dataset Builder")
    print("=" * 60)

    frames = []
    for rel_path, eng_col, dag_col in SOURCES:
        full_path = os.path.join(ROOT, rel_path)
        df = load_source(full_path, eng_col, dag_col)
        if len(df):
            print(f"  ✅  {rel_path:<55} {len(df):>5} rows")
            frames.append(df)
        else:
            print(f"  ⬜  {rel_path:<55} (skipped/missing)")

    if not frames:
        print("\n❌ No data loaded. Check file paths.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nRaw combined rows : {len(combined):,}")

    # ── Clean ──────────────────────────────────────────────────────────────
    combined["english"] = combined["english"].apply(clean_text)
    combined["dagbani"] = combined["dagbani"].apply(clean_text)

    # ── Filter noise ───────────────────────────────────────────────────────
    mask = combined.apply(
        lambda r: is_valid_row(r["english"], r["dagbani"]), axis=1
    )
    combined = combined[mask].reset_index(drop=True)
    print(f"After noise filter: {len(combined):,}")

    # ── Deduplicate (case-insensitive on English side) ─────────────────────
    combined["_eng_key"] = combined["english"].str.lower().str.strip()
    combined.drop_duplicates(subset="_eng_key", keep="first", inplace=True)
    combined.drop(columns=["_eng_key"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"After dedup        : {len(combined):,}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.to_csv(OUTPUT_FILE, sep="\t", index=False, encoding="utf-8")
    print(f"\n✅ Master dataset saved → {OUTPUT_FILE}")
    print(f"   Total pairs: {len(combined):,}")

    # ── Quick sample ───────────────────────────────────────────────────────
    print("\nSample rows:")
    print(combined.sample(min(5, len(combined)), random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()
