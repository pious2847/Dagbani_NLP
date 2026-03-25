import os
import re
import pandas as pd
from pypdf import PdfReader

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.extracted_data = []

    def extract_text(self, start_page=0, end_page=None):
        """Extracts text from the PDF page by page."""
        print(f"--- Processing PDF: {os.path.basename(self.pdf_path)} ---")
        
        try:
            reader = PdfReader(self.pdf_path)
            total_pages = len(reader.pages)
            print(f"   Total Pages: {total_pages}")
            
            if end_page is None:
                end_page = total_pages

            raw_text = ""
            # Limit page range for safety if needed, or process all
            for i in range(start_page, min(end_page, total_pages)):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
            
            return raw_text
            
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""

    def parse_dictionary_entries(self, raw_text):
        """
        Attempts to parse dictionary entries from raw text.
        Dictionary formats vary, so this uses a heuristic approach.
        Assumes lines might look like: "EnglishWord - DagbaniWord" or similar.
        """
        print("--- Parsing Text ---")
        lines = raw_text.split('\n')
        parsed_pairs = []
        
        # Heuristic 1: Look for lines with a clear separator
        # Common separators: " - ", " : ", " means ", or just tab/large spaces
        # For now, let's try to find lines that have at least two words separated by some delimiter
        
        # Regex 1: Word [POS] Definition
        # Matches: "word  n.  definition"
        # Captures: Group 1 (Word), Group 2 (Definition)
        pattern_pos = re.compile(r"^([a-zA-Zɛɣŋɔz]+)\s+(?:n\.|adv\.|v\.|adj\.|v\.p\.|num\.|prep\.|conj\.|int\.)\s+(.+)$", re.IGNORECASE)
        
        # Regex 2: Word - Definition (Backup)
        pattern_sep = re.compile(r"^([a-zA-Zɛɣŋɔz\s]+?)\s*[-–:]\s*([a-zA-Z\s\(\)]+)$")

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Try POS pattern first
            match = pattern_pos.match(line)
            if match:
                eng = match.group(2).strip() # Definition is usually English in this direction?
                dag = match.group(1).strip() # Headword is Dagbani
                
                # Heuristic check: If definition contains "1.", it might be a list. Take first part.
                if "1." in eng:
                    eng = eng.split("1.")[1].strip()
                
                parsed_pairs.append({'english': eng, 'dagbani': dag})
                continue

            # Try Separator pattern
            match = pattern_sep.match(line)
            if match:
                # Assume Left = Dagbani, Right = English for this specific PDF based on snippet
                # "baaji n. bag" -> Left is Dagbani, Right is English
                dag = match.group(1).strip()
                eng = match.group(2).strip()
                parsed_pairs.append({'english': eng, 'dagbani': dag})

        print(f"   Found {len(parsed_pairs)} potential entries.")
        self.extracted_data = parsed_pairs
        return parsed_pairs

    def save_to_tsv(self, output_path):
        if not self.extracted_data:
            print("⚠️ No data to save.")
            return
            
        df = pd.DataFrame(self.extracted_data)
        # Clean duplicates
        df = df.drop_duplicates()
        df.to_csv(output_path, sep='\t', index=False)
        print(f"✅ Saved {len(df)} entries to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_dir = os.path.join(base_dir, "data", "pdf")
    output_file = os.path.join(base_dir, "data", "processed", "pdf_extracted_vocab.tsv")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    all_data = []
    
    if pdf_files:
        for pdf_file in pdf_files:
            print(f"\n{'='*30}\nProcessing: {pdf_file}\n{'='*30}")
            target_pdf = os.path.join(pdf_dir, pdf_file)
            processor = PDFProcessor(target_pdf)
            
            # Process entire PDF
            text = processor.extract_text() # No page limit
            
            # Parse
            entries = processor.parse_dictionary_entries(text)
            all_data.extend(entries)
            
        # Save combined
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.drop_duplicates()
            df.to_csv(output_file, sep='\t', index=False)
            print(f"\n✅ Saved {len(df)} unique entries to {output_file}")
            print("\n--- Sample Entries ---")
            print(df.head(10))
            print("----------------------")
        else:
            print("❌ No entries found in any PDF.")
            
    else:
        print("No PDFs found in data/pdf")
