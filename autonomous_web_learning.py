from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the Chrome driver with options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Function to perform web search and return HTML content
def perform_web_search(query):
    try:
        driver.get("https://www.google.com")
        search_box = driver.find_element("name", "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        driver.implicitly_wait(5)
        return driver.page_source
    except Exception as e:
        logging.error(f"Error performing web search: {e}")
        return None

# Function to parse HTML content and extract search results
def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    for item in soup.find_all('h3'):
        title = item.get_text()
        parent_a = item.find_parent('a')
        if parent_a and 'href' in parent_a.attrs:
            link = parent_a['href']
            results.append({'title': title, 'link': link})
        else:
            logging.warning(f"No parent anchor or href found for item: {title}")
    return results

# Function to store search results in a DataFrame
def store_results(results, query):
    df = pd.DataFrame(results)
    df['Query'] = query
    df.to_csv(f'results_{query}.csv', index=False)

# Function to automate multiple web searches and store results
def automate_searches(queries):
    all_results = []
    for query in queries:
        logging.info(f"Performing web search for query: {query}")
        html_content = perform_web_search(query)
        if html_content:
            results = parse_html(html_content)
            store_results(results, query)
            all_results.extend(results)
        time.sleep(2)  # Sleep to avoid being blocked by Google
    return all_results

# Main function to automate web search, parse results, and store data
def main():
    queries = [
        "Massive Multitask Language Understanding",
        "MMLU benchmark",
        "advanced pre-trained models for MMLU",
        "MMLU research papers"
    ]
    all_results = automate_searches(queries)
    driver.quit()
    logging.info("Web search automation completed.")

if __name__ == "__main__":
    main()