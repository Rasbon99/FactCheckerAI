import os
import re
from langdetect import detect

import dotenv
from groq import Groq

from Preprocessor.ner import NER
from Preprocessor.summarizer import Summarizer

from log import Logger

class Preprocessing_Pipeline():
    def __init__(self, env_file="key.env", config=None):
        """
        Initializes the preprocessing pipeline, setting up the necessary components like NER, Summarizer, and translation configuration.
        
        Args:
            env_file (str, optional): The environment file containing API keys. Default is "Pkey.env".
            config (dict, optional): Configuration options for translation, summarization, and NER. 
                                      Default is {"translation": True, "summarize": True, "NER": True}.
        
        Returns:
            None
        
        Raises:
            KeyError: If the environment variables for the API keys cannot be found.
        """
        dotenv.load_dotenv(env_file, override=True)

        self.logger = Logger(self.__class__.__name__).get_logger()
        self.ner = NER()
        self.summarizer = Summarizer()

        self.config = {
            "translation": True,
            "summarize": True,
            "NER": True
        }
        if config:
            self.config.update(config)

        self.client = Groq()

    def run_claim_pipe(self, claim, max_lenght=150):
        """
        Processes a claim by translating it to English and summarizing it.

        Args:
            claim (str): The claim text to preprocess.
            max_lenght (int, optional): The maximum length for the summary. Default is 150.

        Returns:
            str: The preprocessed claim, translated and/or summarized based on configuration.
        
        Raises:
            Exception: If there is an error during preprocessing (translation or summarization).
        """
        self.logger.info("Starting claim preprocessing...")

        if self.config.get("summarize", True):
            claim_title = self.summarizer.claim_title_summarize(claim, max_lenght)
            claim_summary = self.summarizer.generate_summary(claim, max_lenght)
            
            self.logger.info("Claim preprocessing completed.")
            
            return claim_title, claim_summary

        self.logger.info("Claim preprocessing completed.")

        return claim, claim

    def run_sources_pipe(self, sources, max_lenght=1024):
        """
        Processes a list of sources by translating and/or summarizing each source as required.

        Args:
            sources (list): A list of source objects (e.g., text articles) to preprocess.

        Returns:
            list: A list of preprocessed sources, each being a string of translated and/or summarized text.
        
        Raises:
            NotImplementedError: If the implementation for sources preprocessing is not provided.
        """
        self.logger.info("Starting sources preprocessing...")

        if self.config.get("summarize", True):
            new_bodies = self.summarizer.summarize_texts([d['body'] for d in sources], max_lenght)
            for d, new_body in zip(sources, new_bodies):
                d['body'] = new_body
        
        if self.config.get("NER", True):
            for source in sources:
                topic_and_entities = self.ner.extract_entities_and_topic(source['body'])
                source['topic'] = topic_and_entities['topic']
                source['entities'] = topic_and_entities['entities']

        self.logger.info("Sources preprocessing completed.")
        
        return sources
