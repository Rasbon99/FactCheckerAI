import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from log import Logger

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

    def extract_context(self, url):
        """
        Extracts the title and body of a web page from the given URL.
        
        Args:
            url (str): The URL of the web page to extract content from.
        
        Returns:
            dict: A dictionary containing:
                - 'title' (str): The title of the page, or 'Title not found' if not available.
                - 'url' (str): The URL of the web page.
                - 'body' (str): The text content of the page, with spaces separating content blocks.
        """
        self.logger.info(f"...Start body extraction: {url} ...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title and body
            title = soup.title.string if soup.title else 'Title not found'
            body = soup.get_text(separator=' ', strip=True)

            return {'title': title, 'url': url, 'body': body}

        except requests.RequestException as e:
            self.logger.error(f"Request error for URL '{url}': {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error while extracting body from URL '{url}': {e}")

        return {'title': 'Title not found', 'url': url, 'body': ''}

    def search_and_extract(self, query, num_results=5):
        """
        Performs a search using the provided query, and extracts the title and body of the resulting pages.
        
        Args:
            query (str): The search query to send to DuckDuckGo.
            num_results (int): The number of search results to retrieve. Default is 5.
        
        Returns:
            list: A list of dictionaries, each containing:
                - 'title' (str): The title extracted from the search result pages.
                - 'url' (str): The URL extracted from the search result pages.
                - 'body' (str): The body content extracted from the search result pages.
        """
        self.logger.info("...Start searching and extracting query...")
        search_results = []

        try:
            results = self.ddg.text(query, max_results=num_results)

            for result in results:
                url = result['href']
                extracted_data = self.extract_context(url)

                if extracted_data['title'] and extracted_data['body']:
                    self.logger.info(f'{extracted_data["title"]} - {extracted_data["url"]}')
                    self.logger.info(f"{extracted_data['body'][:200]}...")  # Preview body text
                    search_results.append(extracted_data)

        except Exception as e:
            self.logger.error(f"Error during search and extract for query '{query}': {e}")

        return search_results

    def search(self, query, num_results=5):
        """
        Performs a search using the provided query and returns search results containing:
        - title (str): The title of the search result.
        - url (str): The URL of the search result.
        - snippet (str): A brief summary of the search result.
        
        Args:
            query (str): The search query to send to DuckDuckGo.
            num_results (int): The number of search results to retrieve. Default is 5.
        
        Returns:
            list: A list of dictionaries, each containing:
                - 'title' (str): The title of the search result.
                - 'url' (str): The URL of the search result.
                - 'snippet' (str): The summary or snippet of the search result.
        
        Note:
            This function does not extract the body content of the web pages. It only returns search result metadata.
        """
        self.logger.info("...Start searching query...")
        search_results = []

        try:
            results = self.ddg.text(query, max_results=num_results)

            for result in results:
                title = result.get('title', 'No title found')
                url = result.get('href', '')
                snippet = result.get('body', 'No snippet found')

                search_results.append({'title': title, 'url': url, 'snippet': snippet})
                self.logger.info(f"Title: {title} - URL: {url}")

        except Exception as e:
            self.logger.error(f"Error during search for query '{query}': {e}")

        return search_results