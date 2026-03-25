import os
import pandas as pd
import glob

def clean_and_merge():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    output_file = os.path.join(processed_dir, "final_training_set.tsv")
    
    print(f"--- Cleaning and Merging Data in {processed_dir} ---")
    
    all_files = glob.glob(os.path.join(processed_dir, "*.tsv"))
    combined_data = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Skip the output file itself if it exists to avoid recursion loop
        if filename == "final_training_set.tsv":
            continue
            
        try:
            # Read file
            df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
            
            # Normalize columns
            df.columns = [c.lower().strip() for c in df.columns]
            
            if 'english' not in df.columns or 'dagbani' not in df.columns:
                print(f"⚠️ Skipping {filename}: Missing 'english' or 'dagbani' columns.")
                continue
                
            # Select only relevant columns
            df = df[['english', 'dagbani']]
            
            # Drop duplicates in this file
            original_len = len(df)
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)
            new_len = len(df)
            
            if original_len != new_len:
                print(f"🧹 Cleaned {filename}: Removed {original_len - new_len} duplicates.")
                # Save back to file (User requested to clean each file)
                df.to_csv(file_path, sep='\t', index=False)
            else:
                print(f"✅ {filename} is already clean.")
                
            combined_data.append(df)
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            
    # Merge all
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Drop duplicates across the entire combined dataset
        total_before = len(final_df)
        final_df.drop_duplicates(inplace=True)
        total_after = len(final_df)
        
        # Save final set
        final_df.to_csv(output_file, sep='\t', index=False)
        
        print(f"\n🎉 Merging Complete!")
        print(f"   Total Raw Rows: {total_before}")
        print(f"   Unique Training Pairs: {total_after}")
        print(f"   Saved to: {output_file}")
    else:
        print("❌ No data found to merge.")

if __name__ == "__main__":
    clean_and_merge()
