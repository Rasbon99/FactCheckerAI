from log import Logger
from summarizer import Summarizer
from ner import NER
from groq import Groq
from langdetect import detect
import dotenv
import os

class Preprocessor():
    def __init__(self, env_file="key.env", config=None):
        dotenv.load_dotenv(env_file, override=True)

        self.logger = Logger(self.__class__.__name__).get_logger()
        self.ner = NER()
        self.summarizer = Summarizer()

        self.config = {
            "translation": True,
            "NER": True,
            "summarize": True
        }
        if config:
            self.config.update(config)

        self.preprocessed_data_path = os.getenv("SOURCES_DATA_PATH")
        self.client = Groq()

    def translate_to_english(self, text):
        """
        Translate a text to English using langdetect first, then fall back to Groq if not in English.
        Args:
            text (str): The text to translate.

        Returns:
            str: The text translated into English (or unchanged if it is already in English).
        """
        self.logger.info("Starting translation process.")

        try:
            # Detect language using langdetect
            detected_language = detect(text)
            self.logger.info(f"Detected language: {detected_language}")

            # If the detected language is English, return the text as is
            if detected_language == 'en':
                self.logger.info("Text is already in English, no translation needed.")
                return text

            # If it's not in English, fall back to Groq for translation
            self.logger.info("Text is not in English, using Groq model for translation.")
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a translation model."},
                    {"role": "user", "content": f"Translate the following text to English and return only the translated text: {text}."}
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
            self.logger.error(f"Translation failed for text: {text[:200]}... Error: {e}")
            return text  # If translation fails, return the origina
    
    def sources_to_csv(self, summarized_sources):
        pass

    def pipe_claim_preprocessing(self, claim, max_lenght=150, min_lenght=50):
        """
        Process a claim by translating it to English and summarizing it.
        Args:
            claim (str): The claim text to preprocess.
            max_lenght (int): The maximum length for the summary.
            min_lenght (int): The minimum length for the summary.

        Returns:
            str: The preprocessed claim.
        """
        self.logger.info("Starting claim preprocessing.")

        if self.config.get("translation", True):
            claim = self.translate_to_english(claim)

        if self.config.get("NER", True):
            topic_and_entities = self.ner.extract_entities_and_topic(claim)

        if self.config.get("summarize", True):
            claim = self.summarizer.summarize(claim, max_lenght)

        self.logger.info("Claim preprocessing completed.")

        if self.config.get("NER", True):
            return claim, topic_and_entities

        return claim

    def pipe_sources_preprocessing(self, sources):
        """
        Process a list of sources by translating and/or summarizing each source as required.
        Args:
            sources (list): A list of source objects (e.g., text articles) to preprocess.

        Returns:
            list: A list of preprocessed sources.
        """
        self.logger.info("Starting sources preprocessing.")
        
        #TODO Placeholder for implementation
        pass