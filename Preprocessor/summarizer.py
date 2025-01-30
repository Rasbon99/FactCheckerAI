import os
import sys

import dotenv
from groq import Groq

from log import Logger

class Summarizer:
    def __init__(self, env_file="Pkey.env", model=None):
        """
        Initializes the Summarizer class with a specific model and configures the Groq API client.

        Args:
            env_file (str, optional): The environment file containing the API keys. Default is "Pkey.env".
            model (str, optional): The model to use for summarization. If not provided, defaults to the model set in the environment.

        Attributes:
            model (str): The specified model for Groq.
            client (Groq): Instance of the Groq client for API requests.

        Returns:
            None

        Raises:
            KeyError: If the environment variables for the API keys cannot be found.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.client = Groq()
        
        if model:
            self.model = model

    def summarize(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Generates a summary for the given text using the Groq API.

        Args:
            text (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens for the completion. Default is 1024.
            temperature (float, optional): Controls randomness. Default is 0.5.
            stop (str or None, optional): Optional sequence indicating where the model should stop. Default is None.

        Returns:
            str: The summary generated by the model.

        Raises:
            Exception: If there is an error during the summarization process.
        """
        self.logger.info("Starting summarization process.")
        self.logger.info("Input text: %s...", text[:200]) 

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a news summarizer. Create a sentence with logic that contains the most important
                                                    keywords to search for the given news on the web. Provide only the keywords string as output
                                                    without quotation marks."""},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stop=stop
            )

            summary = response.choices[0].message.content.strip()
            self.logger.info("Summarization completed successfully.")
            self.logger.info("Generated summary: %s...", summary[:200])
            return summary

        except Exception as e:
            self.logger.error("Error generating summary: %s", e)
            return None

    def summarize_texts(self, texts, max_tokens=1024, temperature=0.5, stop=None):
        """
        Generates summaries for a list of texts.

        Args:
            texts (list): List of strings to summarize.
            max_tokens (int, optional): Maximum number of tokens for each completion. Default is 1024.
            temperature (float, optional): Controls randomness. Default is 0.5.
            stop (str or None, optional): Optional sequence indicating where the model should stop. Default is None.

        Returns:
            list: A list of generated summaries.

        Raises:
            Exception: If there is an error during the batch summarization process.
        """
        self.logger.info("Starting batch summarization process for %d texts.", len(texts))
        summaries = []

        for index, text in enumerate(texts):
            self.logger.info("Summarizing text %d/%d...", index + 1, len(texts))
            self.logger.debug("Text %d content: %s", index + 1, text[:200])
            
            summary = self.summarize(text, max_tokens=max_tokens, temperature=temperature, stop=stop)

            if summary:
                summaries.append(summary)
                self.logger.info("Text %d summarized successfully.", index + 1)
            else:
                self.logger.error("Error summarizing text %d.", index + 1)

        self.logger.info("Batch summarization process completed.")
        return summaries