import os
import sys
import urllib.robotparser
import urllib.parse
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from ng_client import NewsGuardClient

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from log import Logger
os.chdir(current_dir)

class Scraper:
    def __init__(self):
        """
        Initializes the Scraper class, setting up the logger and DuckDuckGo search client.
        
        Attributes:
            logger (Logger): A logger instance for logging messages.
            ddg (DDGS): An instance of DuckDuckGo search client.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.ddg = DDGS()
        self.ng_client = NewsGuardClient()

    def extract_context(self, url):
        """
        Extracts the title and body of a web page from the given URL.
        
        Args:
            url (str): The URL of the web page to extract content from.
        
        Returns:
            dict: A dictionary containing:
                - 'title' (str): The title of the page, or 'Title not found' if not available.
                - 'site' (str): The name of the site
                - 'url' (str): The URL of the web page.
                - 'body' (str): The text content of the page, with spaces separating content blocks.
        """
        self.logger.info(f"Start body extraction: {url} ...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title and body
            title = soup.title.string if soup.title else 'Title not found'
            body = soup.get_text(separator=' ', strip=True)
            parsed_url = urlparse(url)
            site = parsed_url.netloc
        
            return {'title': title, 'site': site,'url': url, 'body': body}

        except requests.RequestException as e:
            self.logger.error(f"Request error for URL '{url}': {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while extracting body from URL '{url}': {e}")

        return {'title': 'Title not found', 'url': url, 'body': ''}
    
    def can_scrape(self, url):
        """
        Check if web scraping is allowed by the website's robots.txt.

        Args:
            url (str): The URL of the website to check.

        Returns:
            bool: True if scraping is allowed, False otherwise.
        """
        parsed_url = urllib.parse.urlparse(url)
        robot_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robot_url)
            rp.read()
            
            # Check if the robots.txt allows scraping for all user-agents ('*')
            if rp.can_fetch('*', url):
                return True
            else:
                return False
        except Exception as e:
            # If there is an error accessing the robots.txt, assume scraping is allowed.
            # You could also choose to log this error if necessary.
            return True

    def search_and_extract(self, query, num_results=10):
        """
        Performs a search using the provided query, and extracts the title, body, and site of the resulting pages.
        
        Args:
            query (str): The search query to send to DuckDuckGo.
            num_results (int): The number of search results to retrieve. Default is 5.
        
        Returns:
            list: A list of dictionaries, each containing:
                - 'title' (str): The title extracted from the search result pages.
                - 'url' (str): The URL extracted from the search result pages.
                - 'body' (str): The body content extracted from the search result pages.
                - 'site' (str): The domain name of the site.
        """
        self.logger.info("Start searching and extracting query...")
        search_results = []

        try:
            results = self.ddg.text(query, max_results=num_results)

            self.logger.info("Scraped websites: %i sites", len(results))

            results = self.filter_sites(results)

            for result in results:
                url = result['href']
                
                # Check if scraping is allowed for the site
                if not self.can_scrape(url):
                    self.logger.info(f"Skipping {url} due to scraping restrictions.")
                    continue
            
                extracted_data = self.extract_context(url)

                if extracted_data['title'] and extracted_data['body']:
                    self.logger.info(f'{extracted_data["title"]} - {extracted_data["url"]} - {extracted_data["site"]}')
                    self.logger.info(f"{extracted_data['body'][:200]}...")  # Preview body text
                    search_results.append(extracted_data)

        except Exception as e:
            self.logger.error(f"Error during search and extract for query '{query}': {e}")

        return search_results
    
    def filter_sites(self, sites_list, score_threshold=70):
        """
        Filters a list of sites by their rating from the NewsGuard API.

        :param sites_list: List of dictionaries containing site information, each with an 'href' key.
        :param score_threshold: Minimum score threshold to filter sites (default is 70).
        :return: List of sites that meet the criteria (rank == 'T' and score >= score_threshold).
        """
        filtered_sites = []

        for site in sites_list:
            # Estrazione dell'URL dalla posizione href nella lista
            href = site.get('href')
            if not href:
                continue  # Se non c'è href, salta questa iterazione
            
            # Parsing dell'URL per avere un URL "pulito"
            parsed_url = urlparse(href)
            cleared_site = parsed_url.netloc

            # Ottieni il rating del sito
            rating = self.ng_client.get_rating(cleared_site)
            
            # Se il rating è valido, controlla il rank e lo score
            if rating:
                rank = rating.get('rank')
                score = rating.get('score')
                
                # Se il sito ha rank 'T' e score >= score_threshold, includilo
                if rank == 'T' and score >= score_threshold:
                    filtered_sites.append(site)
                else:
                    # Log per i siti che vengono esclusi
                    self.logger.info("Excluded site %s with rank: %s, score: %s", cleared_site, rank, score)

        self.logger.info("Filtered websites: %s sites", len(filtered_sites))

        return filtered_sites
    
def main():
    # Create an instance of the Scraper class
    scraper = Scraper()

    # Define the search query and the number of results
    query = "Trump federal workers resignation program deferred resignation COVID-19 pandemic US Office Personnel Management OPM"

    # Call the search_and_extract method
    results = scraper.search_and_extract(query)

    # Print the extracted data
    if results:
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Site: {result['site']}")
            print(f"Body: {result['body'][:200]}...")  # Preview of the body (first 200 characters)
            print("-" * 50)
    else:
        print("No results found or an error occurred during extraction.")

if __name__ == "__main__":
    main()
