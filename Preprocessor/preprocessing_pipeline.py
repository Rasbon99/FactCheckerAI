import os
import re
import sys
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

    def translate_to_english(self, text):
        """
        Translates a text to English using langdetect first, then falls back to Groq if the text is not in English.

        Args:
            text (str): The text to translate.

        Returns:
            str: The text translated into English (or unchanged if it is already in English).
        
        Raises:
            Exception: If there is an error during the translation process.
        """
        self.logger.info("Starting translation process.")

        try:
            # Clean the text before detecting language
            cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,:!?\“\”\'\"àèéìòùÀÈÉÌÒÙ]", "", text)
            if not cleaned_text.strip():
                self.logger.warning("Cleaned text is empty, cannot detect language.")
                return None
            
            # Detect language using langdetect
            detected_language = detect(cleaned_text)
            self.logger.info(f"Detected language: {detected_language}")

            # If the detected language is English, return the text as is
            if detected_language == 'en':
                self.logger.info("Text is already in English, no translation needed.")
                return cleaned_text

            # If it's not in English, fall back to Groq for translation
            self.logger.info("Text is not in English, using Groq model for translation.")
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a translation model."},
                    {"role": "user", "content": f"Translate the following text to English and return only the translated text: {cleaned_text}."}
                ],
                model=os.getenv("GROQ_MODEL_NAME"),  # The Groq model to use for translation
                temperature=0.5,
                max_completion_tokens=500
            )

            result = response.choices[0].message.content.strip()

            # Assuming Groq's response is just the translated text
            self.logger.info(f"Text translated to English: {result[:200]}...")
            return result

        except Exception as e:
            self.logger.error(f"Translation failed for text: {cleaned_text[:200]}... Error: {e}")
            return text  # If translation fails, return the original text

    def run_claim_pipe(self, claim, max_lenght=150, min_lenght=50):
        """
        Processes a claim by translating it to English and summarizing it.

        Args:
            claim (str): The claim text to preprocess.
            max_lenght (int, optional): The maximum length for the summary. Default is 150.
            min_lenght (int, optional): The minimum length for the summary. Default is 50.

        Returns:
            str: The preprocessed claim, translated and/or summarized based on configuration.
        
        Raises:
            Exception: If there is an error during preprocessing (translation or summarization).
        """
        self.logger.info("Starting claim preprocessing...")

        if self.config.get("translation", True):
            claim = self.translate_to_english(claim)

        if self.config.get("summarize", True):
            claim = self.summarizer.keywords_summarize(claim, max_lenght)

        self.logger.info("Claim preprocessing completed.")

        return claim

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
            self.logger.debug(sources)
        
        if self.config.get("NER", True):
            topic_and_entities = self.ner.extract_entities_and_topic([d['body'] for d in sources])
            for d, t_e in zip(sources, topic_and_entities):
                d.update(t_e)

        self.logger.info("Sources preprocessing completed.")
        
        return sources
