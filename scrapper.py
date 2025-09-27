# To run this script, you need to install requests and beautifulsoup4
# pip install requests
# pip install beautifulsoup4

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_dagbani_blog():
    """
    Scrapes Dagbani-English translation pairs from the Bryn Mawr blog.
    """
    base_url = "https://lagim.blogs.brynmawr.edu"
    category_url = f"{base_url}/category/dagbani/"
    
    print(f"Fetching posts from: {category_url}")

    # Get the main category page to find all post links
    try:
        main_page = requests.get(category_url)
        main_page.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching main category page: {e}")
        return []

    main_soup = BeautifulSoup(main_page.content, "html.parser")
    
    post_links = []
    # Find all links to individual posts
    for a in main_soup.select('h2.entry-title a'):
        if a.get('href'):
            post_links.append(a['href'])
            
    print(f"Found {len(post_links)} post links to scrape.")
    
    dataset = []

    # Loop through each post link and scrape its content
    for link in post_links:
        print(f"  -> Scraping: {link}")
        try:
            post_page = requests.get(link)
            post_page.raise_for_status()
            post_soup = BeautifulSoup(post_page.content, "html.parser")
            
            # --- Strategy 1: Extract from paragraphs (<strong>Dagbani</strong> - English) ---
            for p in post_soup.select('div.entry-content p'):
                # Check for strong tags which often contain the Dagbani word
                strong_tags = p.find_all('strong')
                if strong_tags:
                    # Get the text content of the paragraph
                    p_text = p.get_text(separator=' ', strip=True)
                    # Use regex to find patterns like "Dagbani word – English translation"
                    # This handles different separators like '–', '-', ':'
                    pairs = re.findall(r'([\w\sŋɣɛɔ]+)\s*[–-:]\s*(.+)', p_text)
                    for dagbani, english in pairs:
                        dataset.append({
                            "dagbani_text": dagbani.strip().lower(),
                            "english_text": english.strip().lower()
                        })

            # --- Strategy 2: Extract from tables ---
            for table in post_soup.select('table'):
                rows = table.find_all('tr')
                # Skip header row if it exists
                start_row = 1 if rows[0].find('th') else 0
                
                for row in rows[start_row:]:
                    cells = row.find_all('td')
                    if len(cells) >= 2: # Ensure there are at least two columns
                        dagbani = cells[0].get_text(strip=True)
                        english = cells[1].get_text(strip=True)
                        
                        # Check for empty cells
                        if dagbani and english:
                            dataset.append({
                                "dagbani_text": dagbani.lower(),
                                "english_text": english.lower()
                            })

        except requests.exceptions.RequestException as e:
            print(f"    Error scraping post {link}: {e}")
            continue
            
    return dataset

# --- Main execution ---
if __name__ == "__main__":
    scraped_data = scrape_dagbani_blog()
    
    if scraped_data:
        # Create a DataFrame and remove duplicates
        df = pd.DataFrame(scraped_data)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print("\n--- Scraping Complete ---")
        print(f"Successfully extracted {len(df)} unique translation pairs.")
        
        # Display a sample of the data
        print("\nSample of Scraped Data:")
        print(df.head(10))
        
        # Save the dataset to a CSV file
        output_filename = "dagbani_scraped_dataset.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nFull dataset saved to '{output_filename}'")
    else:
        print("\nNo data was scraped.")