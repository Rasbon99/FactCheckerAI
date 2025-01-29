import requests
import os
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from log import Logger


class NewsGuardClient:
    """
    Class to interact with the NewsGuard API.
    """
    def __init__(self):
        """
        Initializes the client by loading credentials from the .env file.
        """
        load_dotenv("NGkey.env")
        self.logger = Logger(self.__class__.__name__).get_logger()
        self.client_id = os.getenv("CLIENT_API_ID")
        self.client_secret = os.getenv("NG_API_KEY")
        self.access_token = self._authenticate()
    
    def _authenticate(self):
        """
        Authenticates the client and retrieves an access token.
        
        :return: Access token or None in case of an error
        """
        token_url = "https://account.newsguardtech.com/account-auth/oauth2/token"
        auth_data = {"grant_type": "client_credentials"}
        
        response = requests.post(token_url, auth=HTTPBasicAuth(self.client_id, self.client_secret), data=auth_data)
        
        if response.status_code != 200:
            self.logger.error("Error during authentication:", response.json())
            return None
        
        self.logger.info("Correct NewsGuard authentication")

        return response.json().get("access_token")
    
    def get_rating(self, url):
        """
        Retrieves the rating of a website from the NewsGuard API.
        
        :param url: URL of the website to check
        :return: Dictionary with response data or None in case of an error
        """
        if not self.access_token:
            self.logger.error("Access token not available.")
            return None
                
        check_url = f"https://api.newsguardtech.com/v3/check/?url={url}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
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
