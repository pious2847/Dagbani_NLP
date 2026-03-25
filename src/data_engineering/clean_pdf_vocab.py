import pandas as pd
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Fix encoding issues (common mojibake)
    text = text.replace("â€˜", "'").replace("â€™", "'")
    
    # 2. Remove specific noise tokens found in this PDF
    # "KO" appears frequently, likely a source annotation
    text = re.sub(r"\bKO\b", "", text)
    
    # 3. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def is_valid_entry(row):
    eng = str(row['english'])
    dag = str(row['dagbani'])
    
    # 1. Filter out headers
    if "DATE" in eng and "TITLE" in eng:
        return False
    if "AUTHOR" in eng:
        return False
        
    # 2. Filter out very short or very long entries
    if len(eng) < 2 or len(dag) < 2:
        return False
    if len(eng) > 100 or len(dag) > 50: # Definitions shouldn't be massive paragraphs usually
        return False
        
    # 3. Filter out rows that look like page numbers or artifacts
    if eng.isdigit() or dag.isdigit():
        return False
        
    return True

def clean_vocab_file():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "data", "processed", "pdf_extracted_vocab.tsv")
    output_file = os.path.join(base_dir, "data", "processed", "pdf_vocab_clean.tsv")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    print(f"--- Cleaning {input_file} ---")
    
    try:
        df = pd.read_csv(input_file, sep='\t')
        original_count = len(df)
        
        # Apply cleaning
        df['english'] = df['english'].apply(clean_text)
        df['dagbani'] = df['dagbani'].apply(clean_text)
        
        # Filter rows
        df = df[df.apply(is_valid_entry, axis=1)]
        
        # Drop duplicates again after cleaning
        df = df.drop_duplicates()
        
        # Save
        df.to_csv(output_file, sep='\t', index=False)
        
        print(f"✅ Cleaning Complete!")
        print(f"   Original Rows: {original_count}")
        print(f"   Cleaned Rows:  {len(df)}")
        print(f"   Removed:       {original_count - len(df)}")
        print(f"   Saved to:      {output_file}")
        
        print("\n--- Sample Cleaned Entries ---")
        print(df.head(10))
        
    except Exception as e:
        print(f"❌ Error cleaning file: {e}")

if __name__ == "__main__":
    clean_vocab_file()
