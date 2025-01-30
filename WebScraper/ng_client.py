import os
import requests
import sys
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

from log import Logger

class NewsGuardClient:
    def __init__(self):
        """
        Initializes the client by loading credentials from the .env file.

        Args:
            None

        Raises:
            KeyError: If the environment variables for CLIENT_API_ID or NG_API_KEY are not found.
        """
        load_dotenv("NGkey.env")
        self.logger = Logger(self.__class__.__name__).get_logger()
        try:
            self.client_id = os.getenv("CLIENT_API_ID")
            self.client_secret = os.getenv("NG_API_KEY")
        except KeyError as e:
            self.logger.error(f"Missing environment variable: {str(e)}")
            raise e
        self.access_token = self._authenticate()

    def _authenticate(self):
        """
        Authenticates the client and retrieves an access token.

        Args:
            None

        Returns:
            str: The access token if authentication is successful, None if failed.
        
        Raises:
            requests.exceptions.RequestException: If there is a network error or an invalid response from the API.
        """
        token_url = "https://account.newsguardtech.com/account-auth/oauth2/token"
        auth_data = {"grant_type": "client_credentials"}
        
        try:
            response = requests.post(token_url, auth=HTTPBasicAuth(self.client_id, self.client_secret), data=auth_data)
            
            if response.status_code != 200:
                self.logger.error("Error during authentication:", response.json())
                return None
            
            self.logger.info("Correct NewsGuard authentication")
            return response.json().get("access_token")
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed during authentication: {str(e)}")
            raise e

    def get_rating(self, url):
        """
        Retrieves the rating of a website from the NewsGuard API.
        
        Args:
            url (str): URL of the website to check.

        Returns:
            dict: A dictionary containing the NewsGuard rating details, including "identifier", "rank", and "score".
                  Returns None if there is an error or missing access token.
        
        Raises:
            requests.exceptions.RequestException: If there is a network error or invalid response from the API.
        """
        if not self.access_token:
            self.logger.error("Access token not available.")
            return None
        
        check_url = f"https://api.newsguardtech.com/v3/check/?url={url}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(check_url, headers=headers)
            
            if response.status_code != 200:
                self.logger.error("Error fetching data:", response.json())
                return None
            
            identifier = response.json().get("identifier")
            rank = response.json().get("rank")
            score = response.json().get("score")

            result = {"identifier": identifier, "rank": rank, "score": score}

            self.logger.info("Extract from %s, the NewsGuard ratings: {{rank: %s, score: %s}}", url, result["rank"], result["score"])
            return result
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed while fetching rating for {url}: {str(e)}")
            raise e