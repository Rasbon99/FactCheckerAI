import os
import time
import urllib.robotparser
import urllib.parse
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from groq import Groq
import requests

from WebScraper.ng_client import NewsGuardClient

from log import Logger

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
        
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY_FILTER"))

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

    def search_and_extract(self, query, num_results=10, max_retries=3, min_valid_sources=3, search_results=None, retries=0, attempts=0):
        """
        Performs a search using the provided query, and extracts the title, body, and site of the resulting pages.
        
        Args:
            query (str): The search query to send to DuckDuckGo.
            num_results (int): The number of search results to retrieve. Default is 10.
            max_retries (int): The maximum number of retries in case of a rate limit or other errors. Default is 3.
            min_valid_sources (int): The minimum number of sources that must be valid after filtering. Default is 3.
            search_results (list): The accumulated list of search results.
            retries (int): The current number of retries.
            attempts (int): The current number of full search attempts.
        
        Returns:
            list: A list of dictionaries, each containing:
                - 'title' (str): The title extracted from the search result pages.
                - 'url' (str): The URL extracted from the search result pages.
                - 'body' (str): The body content extracted from the search result pages.
                - 'site' (str): The domain name of the site.
        
        Raises:
            Exception: If there is an error during the search and extract process after all retries.
        """
        # Initialize search_results on the first call
        if search_results is None:
            search_results = []

        self.logger.info("Start searching and extracting query...")
        
        while retries < max_retries and attempts < 3:
            try:
                # Phase 1: Perform the search
                results = self.ddg.text(query, max_results=num_results)

                if not results:  # If there are no results, log and return empty list
                    self.logger.warning(f"No results found for query '{query}'.")
                    return []

                self.logger.info("Scraped websites: %i sites", len(results))

                # Phase 2: Filter sites using NewsGuard Rating Database
                results = self.filter_sites(results)

                if not results:  # If no results remain after filtering, log and return empty list
                    self.logger.warning(f"No valid results after filtering for query '{query}'.")
                    return []

                for result in results:
                    url = result['href']
                    
                    extracted_data = self.extract_context(url)

                    if extracted_data['title'] and extracted_data['body']:
                        # Append score to extracted data
                        extracted_data['score'] = result['score']

                        self.logger.info(f"{extracted_data['title']} - {extracted_data['url']} - {extracted_data['site']}")
                        self.logger.info(f"{extracted_data['body'][:200]}...")  
                        search_results.append(extracted_data)

                # Phase 3: Apply correlation filter
                self.logger.info("Applying correlation filter...")
                filtered_results = self.correlation_filter(query, search_results)

                # Check if there are fewer than the required number of valid sources
                if len(filtered_results) < min_valid_sources:
                    self.logger.warning(f"Only {len(filtered_results)} correlated sources found. Initiating new search for more sources.")
                    remaining_sources_needed = min_valid_sources - len(filtered_results)
                    
                    attempts += 1
                                        
                    # Perform another search to get more results
                    more_sources = self.search_and_extract(query, num_results=remaining_sources_needed, max_retries=max_retries, 
                                                            min_valid_sources=min_valid_sources, search_results=search_results, retries=retries, attempts=attempts)
                    # Ensure more_sources is not None before extending
                    if more_sources:
                        search_results.extend(more_sources)

                # Phase 4: Return only the filtered results
                if len(filtered_results) < min_valid_sources:
                    self.logger.error(f"Attempt {attempts} failed to return enough valid sources.")
                    if attempts >= 3:
                        self.logger.error("Max attempts reached. Aborting.")
                        raise Exception("Unable to retrieve at least 3 valid sources after 3 attempts.")
                    continue

                self.logger.info(f"Filtered results: {len(filtered_results)} sources correlated to the claim.")
                return filtered_results

            except Exception as e:
                self.logger.error(f"Error during search and extract for query '{query}': {e}")
                
                # Check for rate limit error
                if "Ratelimit" in str(e):
                    retries += 1
                    if retries < max_retries:
                        self.logger.warning(f"Rate limit encountered. Retrying in 30 seconds... (Retry {retries}/{max_retries})")
                        time.sleep(30)
                    else:
                        self.logger.error("Max retries reached. Aborting.")
                        raise e
                else:
                    raise e
    
    def correlation_filter(self, claim, sources):
        correlated_sources = []
        
        for source in sources:
            try:
                # Estrarre il contenuto del campo "body"
                source_body = source.get("body", "")[:2000]
                
                # Creazione del prompt per il modello
                prompt = [
                    {"role": "system", "content": f"""
                    You are an expert validator tasked with determining whether a source found online is directly related to the provided claim ('{claim}'). 
                    Your goal is to check if the source discusses the same topic or provides relevant information about the claim. 
                    Focus on the core subject of the claim and the source. Ignore unrelated or vaguely related content.

                    Respond with one of the following:
                    - 'Correlated' if the source is about the same topic as the claim.
                    - 'Not Correlated' if the source is unrelated or only tangentially related.
                    Be concise and accurate in your evaluation.
                    Use only 'Correlated' or 'Not Correlated' in your response."""},
                    {"role": "user", "content": source_body}
                ]
                
                # Chiamata al modello
                response = self.client.chat.completions.create(
                    messages=prompt,
                    model=self.model,
                )
                
                # Estrazione del risultato
                result = response.choices[0].message.content.strip()

                # Aggiungi la fonte completa solo se correlata
                if result == "Correlated":
                    correlated_sources.append(source)
            except Exception as e:
                # Log degli errori per debugging
                print(f"Error processing source: {source}\n{e}")
        
        return correlated_sources
    
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