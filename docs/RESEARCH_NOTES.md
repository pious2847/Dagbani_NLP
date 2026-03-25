# Research Notes — Dagbani NLP Project
**Branch:** `research/nllb-fixed-approach`
**Date:** 2026-03-25
**Researcher:** Claude (Anthropic) — continuing from original implementation

---

## Critical Bug Found in Original Implementation

### Bug: `dag_Latn` is NOT a native NLLB-200 language code

**Original code (`train_final_model.py`, line 115):**
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang="eng_Latn", tgt_lang="dag_Latn")
```

**The problem:**
NLLB-200-distilled-600M covers 200 languages, but **Dagbani (`dag_Latn`) is NOT one of them**. When an unknown language code is passed to the NLLB tokenizer, it silently falls through (or maps to `<unk>`). This means:

- The model's decoder had **no concept of a Dagbani target language**
- The `forced_bos_token_id` was either `<unk>` or missing entirely
- The training loss was optimising for a non-existent target token
- All Dagbani outputs were essentially the model's best guess with no language anchor

**Evidence:** HuggingFace NLLB tokenizer source shows `FAIRSEQ_LANGUAGE_CODES` — `dag_Latn` is absent. Confirmed by community discussion threads.

**The fix (`train_nllb_fixed.py`):**
Inject `dag_Latn` as a new special token using:
```python
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["dag_Latn"]},
    replace_additional_special_tokens=False,  # CRITICAL: preserve existing 200 lang tokens
)
model.resize_token_embeddings(len(tokenizer))
# Warm-start: copy Mooré (mos_Latn) embedding → dag_Latn
# Mooré is in the same Moore-Dagbani language subgroup (Niger-Congo > Gur > Oti-Volta)
with torch.no_grad():
    model.model.shared.weight[dag_id] = model.model.shared.weight[mos_id].clone()
```

**Why initialise from Mooré (`mos_Latn`)?**
Mooré is the closest language to Dagbani that IS natively in NLLB-200:
- Same Moore-Dagbani subgroup (Oti-Volta branch, Niger-Congo > Gur family)
- Mutual partial intelligibility with Dagbani
- Warm-start from a related language instead of random init dramatically accelerates convergence on small datasets

---

## Additional Bugs Fixed

### Bug 2: Deprecated `as_target_tokenizer()` API
**Original:**
```python
with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=64, truncation=True)
```
**Fixed:**
```python
labels = tokenizer(text_target=examples["dagbani"], max_length=80, truncation=True)
```
The `as_target_tokenizer()` context manager was deprecated in `transformers >= 4.36` and removed in later versions. The modern API uses the `text_target` keyword argument.

### Bug 3: BLEU only — wrong primary metric for Dagbani
**Problem:** BLEU operates at word level and penalises near-misses (e.g. `ŋ` vs `n`, `ɛ` vs `e`). For a tonal language like Dagbani with 11 vowels and diacritics, word-level matching fails on phonemically meaningful character differences.

**Fix:** ChrF++ (character n-gram F-score with word-order component) is now the **primary metric**, BLEU retained as secondary.

**Evidence:** arxiv 2602.17425 — "Evaluating Extremely Low-Resource MT" shows ChrF has 2× higher Spearman correlation with human judgements vs BLEU for low-resource African languages.

### Bug 4: No LR warmup or schedule
**Original:** Constant learning rate `2e-5` for all epochs
**Fixed:** Cosine decay with 10% warmup — prevents sharp loss spikes at training start on small datasets.

---

## Research Findings

### 1. Dagbani Language Profile
- **Family:** Niger-Congo → Gur → Oti-Volta → **Moore-Dagbani** subgroup
- **Speakers:** ~1.17 million (Northern Ghana, centered on Tamale)
- **Tonal:** YES — two-level tone system (high/low) + downstep
- **Script:** Latin (dag_Latn), with characters: `ŋ ɣ ɛ ɔ ɤ`
- **Closest related language in NLLB:** Mooré/Mossi (`mos_Latn`, ~7M speakers, Burkina Faso)

### 2. Best Models for Low-Resource African Language MT (2025)
| Rank | Model | Notes |
|------|-------|-------|
| 1 | NLLB-200 3.3B (fine-tuned, fixed) | Best NMT baseline for supported African languages |
| 2 | GPT-4o / Claude 3.5 (few-shot) | Best overall quality, no fine-tuning needed |
| 3 | AfriNLLB (Feb 2025) | Pruned NLLB-200, 57% faster, no Dagbani |
| 4 | mBART-50 | Alternative backbone, same tokenizer fix needed |

**For Dagbani specifically:** No dedicated model exists in the public literature as of March 2026. This project's dataset (~1,500+ pairs after augmentation) may be the **largest publicly assembled Dagbani MT dataset in existence**.

### 3. Data Collected
| Source | Pairs | Notes |
|--------|-------|-------|
| Original processed datasets | ~1,164 | Various quality levels |
| Bryn Mawr blog scrape (new) | 336 | High quality, lessons 1–6 + family vocab |
| LLM augmentation (Claude API) | ~2,000 | Synthetic, filtered by length ratio |
| **Total (approx)** | **~3,500** | After deduplication |

### 4. Recommended Evaluation Protocol
1. **Primary:** ChrF++ (word_order=2) — character-level, handles diacritics
2. **Secondary:** SacreBLEU with flores200 tokenisation
3. **Diagnostic:** If |ChrF++ - BLEU| > 10 → investigate hallucination or source copying
4. **Expected ranges** for this data scale:
   - Without augmentation: BLEU 8–18, ChrF++ 25–40
   - With LLM augmentation: BLEU 15–25, ChrF++ 35–50

### 5. Training Strategy Summary
```
Stage 1: build_master_dataset.py
         → Merge all sources, clean, deduplicate
         → Output: data/processed/master_training_set.tsv

Stage 2: generate_llm_augmentation.py  (requires ANTHROPIC_API_KEY)
         → Claude API 10-shot back-translation
         → Length-ratio filtering
         → Output: data/processed/augmented_training_set.tsv

Stage 3: train_nllb_fixed.py
         → Inject dag_Latn token (warm-started from mos_Latn)
         → Cosine LR + warmup, 12 epochs
         → Primary metric: ChrF++
         → Output: models/nllb_dagbani_fixed/

Stage 4: translate_nllb_fixed.py
         → Evaluate with correct forced_bos_token_id=dag_id
         → Report ChrF++ and BLEU
         → Interactive mode for manual QA
```

---

## Running the Pipeline

```bash
# 1. Build master dataset
python src/data_engineering/build_master_dataset.py

# 2. Generate LLM augmentation (optional but recommended)
export ANTHROPIC_API_KEY="sk-ant-..."
python src/training_script/generate_llm_augmentation.py

# 3. Train (use augmented set if step 2 was run)
python src/training_script/train_nllb_fixed.py
# OR with augmented data:
python src/training_script/train_nllb_fixed.py \
    --data data/processed/augmented_training_set.tsv \
    --epochs 15

# 4. Evaluate
python src/training_script/translate_nllb_fixed.py

# 5. Interactive test
python src/training_script/translate_nllb_fixed.py --interactive
```

---

## References

1. [NLLB-200 — "No Language Left Behind"](https://arxiv.org/abs/2207.04672)
2. [How to fine-tune NLLB-200 for a new language — David Dale/Medium](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865)
3. [Unlocking PEFT for Low-Resource Language Translation (arxiv 2404.04212)](https://arxiv.org/html/2404.04212v1) — shows Houlsby adapters +10% BLEU; LoRA −38% for seq2seq MT
4. [Evaluating Extremely Low-Resource MT: ChrF++ vs BLEU (arxiv 2602.17425)](https://arxiv.org/html/2602.17425)
5. [Scaling Low-Resource MT via Synthetic Data with LLMs (EMNLP 2025)](https://arxiv.org/html/2505.14423)
6. [AfriMTE / AfriCOMET — African language MT evaluation (arxiv 2311.09828)](https://arxiv.org/abs/2311.09828)
7. [Dagbani Language — Wikipedia](https://en.wikipedia.org/wiki/Dagbani_language)
8. [Neural MT for Mooré (closest supported language to Dagbani)](https://eudl.eu/doi/10.4108/eai.18-12-2023.2348140)
9. [IrokoBench: LLM evaluation for 17 African languages](https://arxiv.org/html/2406.03368v1)
