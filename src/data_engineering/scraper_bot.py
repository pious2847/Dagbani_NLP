import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class DagbaniScraper:
    def __init__(self):
        print("--- Initializing Scraper Bot 🤖 ---")
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless') # Keep visible for debugging
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # Start at the first lesson to see the sidebar
        self.start_url = "https://learndagbani.org/courses/learn-dagbanli/?unit=unit-1&lesson=01-what-is-the-dagbanli-language"
        self.data = []
        self.visited_urls = set()

    def expand_sidebar(self):
        """Clicks all Unit buttons to ensure links are visible."""
        print("Expanding sidebar units...")
        try:
            # Find buttons that look like Unit headers
            # They contain "Unit" text
            buttons = self.driver.find_elements(By.XPATH, "//button[contains(., 'Unit')]")
            for btn in buttons:
                try:
                    btn.click()
                    time.sleep(0.5)
                except:
                    pass
            time.sleep(2)
        except Exception as e:
            print(f"Error expanding sidebar: {e}")

    def get_lesson_links(self):
        """Extracts lesson links from the sidebar."""
        print(f"Navigating to start page: {self.start_url}")
        self.driver.get(self.start_url)
        time.sleep(5) # Wait for sidebar to load
        
        self.expand_sidebar()

        # Sidebar selector based on user provided HTML: div.w-80 -> div -> a
        try:
            sidebar = self.driver.find_element(By.CSS_SELECTOR, "div.w-80.fixed.left-0")
            links = sidebar.find_elements(By.TAG_NAME, "a")
            
            lesson_urls = []
            for link in links:
                href = link.get_attribute("href")
                if href and "lesson=" in href:
                    lesson_urls.append(href)
            
            # Remove duplicates and preserve order
            unique_urls = []
            for url in lesson_urls:
                if url not in unique_urls:
                    unique_urls.append(url)
                    
            print(f"Found {len(unique_urls)} unique lessons in sidebar.")
            return unique_urls
            
        except Exception as e:
            print(f"❌ Error finding sidebar: {e}")
            return []

    def scrape_lesson(self, url):
        """Visits a lesson and extracts text content, specifically looking for tables."""
        if url in self.visited_urls:
            print(f"Skipping already visited: {url}")
            return
            
        print(f"Scraping: {url}")
        self.driver.get(url)
        self.visited_urls.add(url)
        time.sleep(3) 
        
        try:
            # 1. Try to find Tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if tables:
                print(f"   Found {len(tables)} tables.")
                for table in tables:
                    # Try to get headers
                    headers = []
                    header_row = table.find_elements(By.TAG_NAME, "th")
                    if header_row:
                        headers = [h.text.strip() for h in header_row]
                    
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        cells = [c.text.strip() for c in cols]
                        
                        if not cells:
                            continue
                        
                        # Create a generic entry
                        entry = {'source': url, 'type': 'table_row'}
                        
                        # Map cells to headers if possible, else use indices
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                col_name = headers[i]
                            else:
                                col_name = f"col_{i+1}"
                            entry[col_name] = cell
                            
                        self.data.append(entry)

            # 2. Capture ALL Raw Text (for non-table layouts like Numbers)
            try:
                content_div = self.driver.find_element(By.CSS_SELECTOR, "div.flex-1.overflow-y-auto")
                # Get text and split by lines
                raw_text = content_div.text
                lines = raw_text.split('\n')
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if len(line) > 1:
                        self.data.append({
                            'source': url, 
                            'type': 'raw_text_line', 
                            'line_index': i, 
                            'content': line
                        })
            except:
                pass

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    def run(self):
        try:
            urls = self.get_lesson_links()
            
            for i, url in enumerate(urls):
                print(f"[{i+1}/{len(urls)}] Processing...")
                self.scrape_lesson(url)
                
            # Save raw data
            df = pd.DataFrame(self.data)
            output_path = os.path.join(os.path.dirname(__file__), "../../data/raw/scraped_web_data.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Scraping Complete! Saved {len(df)} entries to {output_path}")
            
        finally:
            self.driver.quit()

if __name__ == "__main__":
    bot = DagbaniScraper()
    bot.run()
