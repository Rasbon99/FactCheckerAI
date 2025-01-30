import json
import os
import sys

import dotenv
from groq import Groq

from log import Logger

class NER:
    def __init__(self, env_file="key.env", model=None):
        """
        Initializes the NER class with a specific model and configures the Groq API client.

        Args:
            env_file (str): The env file containing the API keys. Default is "Pkey.env".
            model (str, optional): The model to use for Named Entity Recognition. If not provided, the model is loaded from environment variables.
        
        Attributes:
            model (str): The specified model for Groq.
            client (Groq): Instance of the Groq client for API requests.
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If the specified environment file is not found.
            KeyError: If the model name is not found in the environment variables.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.client = Groq()
        
        if model:
            self.model = model
        
        if not self.model:
            raise KeyError("Model name not found in the environment variables.")

    def extract_entities_and_topic(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Extracts entities and the main topic from the given text using the Groq API.

        Args:
            text (str): The text from which to extract the topic and entities.
            max_tokens (int, optional): The maximum number of tokens for the completion. Default is 1024.
            temperature (float, optional): Controls randomness. Default is 0.5.
            stop (str or None, optional): Optional sequence indicating where the model should stop. Default is None.
        
        Returns:
            dict: A dictionary with the extracted "topic" and "entities".
                Example: {"topic": "Technology", "entities": ["Elon Musk", "SpaceX", "Tesla", "Paris"]}.
        
        Raises:
            Exception: If there is an error during the API request or entity extraction process.
            json.JSONDecodeError: If the response cannot be parsed into a valid JSON format.
        """
        self.logger.info("Starting entity and topic extraction process.")

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """you are an NER model that extracts entities and the topic from a text. 
                                                    the output must be like {"topic": "Technology", "entities": ["Elon Musk", "SpaceX", "Tesla", "Paris"]}"""},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop
            )
            self.logger.info("Groq API call successful.")
            
            result = response.choices[0].message.content.strip()
            self.logger.debug("Raw API response: %s", result)

            try:
                # Convert the result into a dictionary by parsing the structured response
                result_dict = json.loads(result)
                self.logger.info("Successfully parsed API response into JSON.")
                return result_dict
            except json.JSONDecodeError:
                self.logger.error("Error: Response could not be parsed into a valid JSON format.")
                return None

        except Exception as e:
            self.logger.error("Error extracting topic and entities: %s", e)
            return None