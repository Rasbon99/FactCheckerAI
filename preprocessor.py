from googletrans import Translator
from log import Logger
from summarizer import Summarizer
import dotenv
import os

class Preprocessor():
    def __init__(self, env_file="key.env", config=None):
        dotenv.load_dotenv(env_file, override=True)

        self.logger = Logger(self.__class__.__name__).get_logger()
        self.translator = Translator()
        self.summarizer = Summarizer()

        self.config = {
            "translation": True,
            "summarize": True
        }
        if config:
            self.config.update(config)

        self.preprocessed_data_path = os.getenv("SOURCES_DATA_PATH")

    def translate_to_english(self, text):
        """
        Translate a text to English only if it is not already in English.
        Args:
            text (str): The text to translate.

        Returns:
            str: The text translated into English (or unchanged if it is already in English).
        """

        self.logger.info("Starting translation process.")

        charc_to_detect = int(len(text)/4)

        detected_language = self.translator.detect(text[:charc_to_detect])
        self.logger.info(f"Detected language: {detected_language.lang}")

        # If the language is English, return the original text
        if detected_language.lang == 'en':
            self.logger.info("Text is already in English, no translation needed.")
            return text
        
        chunk_size = 500  # Define the chunk size
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]  # Split text into chunks
        translated_chunks = []

        try:
            for chunk in chunks:
                translated_chunk = self.translator.translate(chunk, src='auto', dest='en').text
                translated_chunks.append(translated_chunk)
            translated_text = ' '.join(translated_chunks)  # Concatenate translated chunks
            return translated_text
        except Exception as e:
            self.logger.error(f"Translation failed for text: {text}. Error: {e}")
            return text  # If translation fails, return the original text
    
    def sources_to_csv(self, summarized_sources):
        pass

    def pipe_claim_preprocessing(self, claim, max_lenght=250, min_lenght=100):
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

        #TODO Try to implement NER

        if self.config.get("summarize", True):
            claim = self.summarizer.summarize(claim, max_lenght, min_lenght)

        self.logger.info("Claim preprocessing completed.")
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