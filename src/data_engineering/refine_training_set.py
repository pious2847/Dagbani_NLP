import pandas as pd
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove "KO" (seems to be a source annotation)
    text = re.sub(r"\bKO\b", "", text)
    
    # 2. Remove language origin notes in brackets/parentheses
    # e.g., "<Hausa", "(Arabic via Hausa)", "[vide ...]"
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    
    # 3. Remove specific phrases found in the file
    text = text.replace("Arabic via Hausa", "")
    text = text.replace("English", "")
    text = text.replace("Hausa", "")
    
    # 4. Clean up punctuation and whitespace
    text = text.replace(";", "")
    text = text.replace("’", "'")
    text = text.replace("â€˜", "'")
    text = text.replace("â€™", "'")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def is_valid_row(row):
    eng = str(row['english'])
    dag = str(row['dagbani'])
    
    # Filter out empty or too short
    if len(eng) < 2 or len(dag) < 2:
        return False
        
    # Filter out rows that are just definitions or metadata
    if "DATE TITLE" in eng:
        return False
    if "AUTHOR" in eng:
        return False
        
    # Filter out rows where English is just a description of a letter or symbol
    if eng.startswith("c ") and len(eng) < 5:
        return False
        
    return True

def refine_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data", "processed", "final_training_set.tsv")
    output_file = os.path.join(base_dir, "data", "processed", "final_training_set_refined.tsv")
    
    print(f"--- Refining {input_file} ---")
    
    try:
        df = pd.read_csv(input_file, sep='\t', on_bad_lines='skip')
        original_len = len(df)
        
        # Apply cleaning
        df['english'] = df['english'].apply(clean_text)
        df['dagbani'] = df['dagbani'].apply(clean_text)
        
        # Filter
        df = df[df.apply(is_valid_row, axis=1)]
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Save
        df.to_csv(output_file, sep='\t', index=False)
        
        print(f"✅ Refined Dataset Saved to {output_file}")
        print(f"   Original Rows: {original_len}")
        print(f"   Refined Rows:  {len(df)}")
        print(f"   Removed:       {original_len - len(df)}")
        
        # Overwrite the original final set if successful, so training script picks it up
        df.to_csv(input_file, sep='\t', index=False)
        print(f"   (Overwrote original file for convenience)")
        
    except Exception as e:
        print(f"❌ Error refining dataset: {e}")

if __name__ == "__main__":
    refine_dataset()
