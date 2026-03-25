import pandas as pd
import os
import random

class CurriculumBuilder:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vocab = {} # English -> Dagbani
        self.templates = []
        
    def load_data(self):
        """Loads all TSV files and builds a raw dictionary."""
        print("--- Loading Data ---")
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tsv')]
        
        full_df_list = []
        for f in all_files:
            try:
                df = pd.read_csv(f, sep='\t', on_bad_lines='skip', header=0)
                # Normalize columns
                df.columns = [c.lower().strip() for c in df.columns]
                if 'english' in df.columns and 'dagbani' in df.columns:
                    full_df_list.append(df[['english', 'dagbani']])
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        self.full_df = pd.concat(full_df_list, ignore_index=True).dropna().drop_duplicates()
        
        # Build basic vocabulary from single words
        for _, row in self.full_df.iterrows():
            eng = str(row['english']).lower().strip()
            dag = str(row['dagbani']).lower().strip()
            
            # If it's a single word (no spaces), add to vocab
            if ' ' not in eng:
                self.vocab[eng] = dag
                
        print(f"Loaded {len(self.full_df)} pairs. Found {len(self.vocab)} vocabulary terms.")

    def generate_baby_talk(self):
        """Generates simple sentences using substitution templates."""
        print("--- Generating 'Baby Talk' Synthetic Data ---")
        
        # Define some manual templates based on observed patterns
        # Format: (English Template, Dagbani Template, [Variables])
        
        # Pronouns
        pronouns = {
            "i": "n",
            "you": "a",
            "he": "o",
            "she": "o",
            "we": "ti",
            "they": "be"
        }
        
        # Foods (inferred from data + common ones)
        foods = {
            "fufu": "sakoro",
            "banku": "banku",
            "tz": "sagam",
            "rice": "shinkaafa", # Added common word
            "yam": "nyuli"       # Added common word
        }
        
        # Verbs
        verbs_eating = {
            "eating": "dirila",
            "eat": "di"
        }
        
        synthetic_data = []
        
        # Template 1: [PRONOUN] is/am/are eating [FOOD]
        for eng_pro, dag_pro in pronouns.items():
            for eng_food, dag_food in foods.items():
                # English grammar adjustment
                be_verb = "am" if eng_pro == "i" else "is" if eng_pro in ["he", "she"] else "are"
                
                eng_sent = f"{eng_pro} {be_verb} eating {eng_food}"
                dag_sent = f"{dag_pro} dirila {dag_food}"
                synthetic_data.append((eng_sent, dag_sent))
                
                # Variation: Simple "eat"
                # "I eat fufu" -> "N di sakoro" (Simplified grammar for baby stage)
                eng_sent_simple = f"{eng_pro} eat {eng_food}"
                dag_sent_simple = f"{dag_pro} di {dag_food}"
                synthetic_data.append((eng_sent_simple, dag_sent_simple))

        # Template 2: [PRONOUN] have/has [NUMBER] [OBJECT]
        # "I have one goat" -> "N mali bua yini"
        numbers = {
            "one": "yini",
            "two": "ayi",
            "three": "ata",
            "ten": "pia"
        }
        objects = {
            "goat": "bua",
            "child": "bia",
            "bag": "bagi",
            "car": "loori"
        }
        
        for eng_pro, dag_pro in pronouns.items():
            for eng_num, dag_num in numbers.items():
                for eng_obj, dag_obj in objects.items():
                    # Pluralization (simplified)
                    eng_obj_plural = eng_obj + "s" if eng_num != "one" else eng_obj
                    have_verb = "have" if eng_pro != "he" and eng_pro != "she" else "has"
                    
                    eng_sent = f"{eng_pro} {have_verb} {eng_num} {eng_obj_plural}"
                    dag_sent = f"{dag_pro} mali {dag_obj} {dag_num}" # Dagbani structure: Subject + Have + Object + Number
                    synthetic_data.append((eng_sent, dag_sent))

        print(f"Generated {len(synthetic_data)} synthetic 'Baby Talk' sentences.")
        return pd.DataFrame(synthetic_data, columns=['english', 'dagbani'])

    def build_curriculum(self):
        """Exports data in curriculum stages."""
        self.load_data()
        
        # Stage 1: Vocabulary (Single words)
        vocab_df = pd.DataFrame(list(self.vocab.items()), columns=['english', 'dagbani'])
        
        # Stage 2: Baby Talk (Synthetic Simple Sentences)
        baby_df = self.generate_baby_talk()
        
        # Stage 3: Real Corpus (Existing sentences)
        # Filter out single words to keep this "sentences only"
        corpus_df = self.full_df[self.full_df['english'].str.contains(' ')]
        
        # Combine all for final training
        # Combine all for final training
        # Removed intentional oversampling to avoid duplicates as requested
        final_df = pd.concat([vocab_df, baby_df, corpus_df], ignore_index=True).drop_duplicates()
        
        # Save
        output_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        vocab_df.to_csv(os.path.join(output_dir, "stage_1_vocab.tsv"), sep='\t', index=False)
        baby_df.to_csv(os.path.join(output_dir, "stage_2_baby.tsv"), sep='\t', index=False)
        corpus_df.to_csv(os.path.join(output_dir, "stage_3_corpus.tsv"), sep='\t', index=False)
        final_df.to_csv(os.path.join(output_dir, "curriculum_complete.tsv"), sep='\t', index=False)
        
        print(f"✅ Curriculum built! Saved to {output_dir}")
        print(f"   - Vocab: {len(vocab_df)}")
        print(f"   - Baby Talk: {len(baby_df)}")
        print(f"   - Corpus: {len(corpus_df)}")
        print(f"   - Total Combined: {len(final_df)}")

if __name__ == "__main__":
    # Assuming script is run from src/data_engineering or similar, adjust path
    # But we are running from root usually or src.
    # Let's use absolute path based on user workspace
    DATA_DIR = r"c:\Users\CODE-D\OneDrive\Desktop\DEV\Dagbani_NLP_Project\data"
    
    builder = CurriculumBuilder(DATA_DIR)
    builder.build_curriculum()
