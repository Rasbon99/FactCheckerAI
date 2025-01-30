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
        
        Args:
            None

        Returns:
            None
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
                - 'title' (str or None): The title of the page, or None if not available.
                - 'site' (str or None): The name of the site, or None if unavailable.
                - 'url' (str): The URL of the web page.
                - 'body' (str or None): The text content of the page, or None if access is restricted.
        
        Raises:
            requests.RequestException: If there is an error during the HTTP request.
            Exception: For unexpected errors during content extraction.
        """
        self.logger.info(f"Starting body extraction: {url} ...")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code in [401, 403, 402]:
                self.logger.warning(f"Access denied for URL '{url}' with status {response.status_code}.")
                return {'title': None, 'site': None, 'url': url, 'body': None}
            
            soup = BeautifulSoup(response.content, 'html.parser')

            # Check if the content indicates a restriction message
            blocked_keywords = ["subscribe", "log in", "sign in", "register", "access denied", "are you a robot"]
            page_text = soup.get_text(separator=' ', strip=True).lower()
            if any(keyword in page_text[:100] for keyword in blocked_keywords):
                self.logger.warning(f"Content appears restricted for URL '{url}'.")
                return {'title': None, 'site': None, 'url': url, 'body': None}

            # Extract title and body
            title = soup.title.string if soup.title else None
            body = soup.get_text(separator=' ', strip=True)
            parsed_url = urlparse(url)
            site = parsed_url.netloc
        
            return {'title': title, 'site': site, 'url': url, 'body': body}

        except requests.Timeout:
            self.logger.error(f"Timeout error for URL '{url}'")
            return {'title': None, 'site': None, 'url': url, 'body': None}
        
        except requests.RequestException as e:
            self.logger.error(f"Request error for URL '{url}': {e}")
            return {'title': None, 'site': None, 'url': url, 'body': None}

        except Exception as e:
            self.logger.error(f"Unexpected error while extracting body from URL '{url}': {e}")
            return {'title': None, 'site': None, 'url': url, 'body': None}


    def can_scrape(self, url):
        """
        Check if web scraping is allowed by the website's robots.txt.
        
        Args:
            url (str): The URL of the website to check.
        
        Returns:
            bool: True if scraping is allowed, False otherwise.

        Raises:
            Exception: If there is an error accessing the robots.txt or if parsing fails.
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
            num_results (int): The number of search results to retrieve. Default is 10.
        
        Returns:
            list: A list of dictionaries, each containing:
                - 'title' (str): The title extracted from the search result pages.
                - 'url' (str): The URL extracted from the search result pages.
                - 'body' (str): The body content extracted from the search result pages.
                - 'site' (str): The domain name of the site.
        
        Raises:
            Exception: If there is an error during the search and extract process.
        """
        self.logger.info("Start searching and extracting query...")
        search_results = []

        try:
            results = self.ddg.text(query, max_results=num_results)

            self.logger.info("Scraped websites: %i sites", len(results))

            results = self.filter_sites(results)

            for result in results:
                url = result['href']
                
                extracted_data = self.extract_context(url)

                # TODO - Verifiy the body

                if extracted_data['title'] and extracted_data['body']:
                    
                    # Append score to exctracted data
                    extracted_data['score'] = result['score']

                    self.logger.info(f'{extracted_data['title']} - {extracted_data['url']} - {extracted_data['site']}')
                    self.logger.info(f"{extracted_data['body'][:200]}...")  # Preview body text
                    search_results.append(extracted_data)

        except Exception as e:
            self.logger.error(f"Error during search and extract for query '{query}': {e}")

        return search_results
    
    def filter_sites(self, sites_list, score_threshold=70):
        """
        Filters a list of sites by their rating from the NewsGuard API.
        
        Args:
            sites_list (list): List of dictionaries containing site information, each with an 'href' key.
            score_threshold (int): Minimum score threshold to filter sites (default is 70).
        
        Returns:
            list: A filtered list of sites that meet the criteria (rank == 'T' and score >= score_threshold) with append the score.
        
        Raises:
            None
        """
        filtered_sites = []

        for site in sites_list:
            # Extract the URL from the href position in the list
            href = site.get('href')
            if not href:
                continue  # If there is no href, skip this iteration
            
            # Parse the URL to get a "clean" URL
            parsed_url = urlparse(href)
            cleared_url = parsed_url.netloc

            # Get the site's rating
            rating = self.ng_client.get_rating(cleared_url)

            # Check if scraping is allowed for the site
            if not self.can_scrape(cleared_url):
                self.logger.info(f"Skipping {cleared_url} due to scraping restrictions.")
                continue
                        
            # If the rating is valid, check the rank and score
            if rating:
                rank = rating.get('rank')
                score = rating.get('score')
                
                # If the site has rank 'T' and score >= score_threshold, include it
                if rank == 'T' and score >= score_threshold:
                    site['score'] = score
                    filtered_sites.append(site)
                else:
                    # Log for sites that are excluded
                    self.logger.info("Excluded site %s with rank: %s, score: %s", cleared_url, rank, score)

        self.logger.info("Filtered websites: %s sites", len(filtered_sites))

        return filtered_sites

    
def main():
    # Create an instance of the Scraper class
    scraper = Scraper()

    # Define the search query and the number of results
    query = "Trump federal workers resignation program deferred resignation COVID-19 pandemic US Office Personnel Management OPM"

    # Call the search_and_extract method
    results = scraper.search_and_extract(query, num_results=30)

    # Print the extracted data
    if results:
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Site: {result['site']}")
            print(f"Body: {result['body'][:1000]}...")  # Preview of the body (first 200 characters)
            print(f"Score: {result['score']}")
            print("-" * 50)
    else:
        print("No results found or an error occurred during extraction.")

if __name__ == "__main__":
    main()
