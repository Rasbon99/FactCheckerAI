from transformers import pipeline
from log import Logger

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the Summarizer class with a default or user-defined model
        and sets up the logger for logging events.
        
        Args:
            model_name (str): The name of the Hugging Face model to use for summarization. 
                              Default is 'facebook/bart-large-cnn'.
        
        Attributes:
            summarizer (pipeline): A pipeline for summarization using the specified model.
            logger (Logger): A logger instance for logging messages.
        """
        # Initialize the summarization model pipeline with the specified model
        self.summarizer = pipeline("summarization", model=model_name)
        
        # Initialize the logger
        self.logger = Logger(self.__class__.__name__).get_logger()
        
        # Log the model initialization
        self.logger.info(f"Summarizer initialized with model: {model_name}")

    def summarize(self, text, max_length=70, min_length=20, do_sample=False):
        """
        Summarizes the input text based on the provided parameters and logs the process.
        
        Args:
            text (str): The text to summarize.
            max_length (int): The maximum length of the summary. Default is 70 characters.
            min_length (int): The minimum length of the summary. Default is 20 characters.
            do_sample (bool): Whether to use sampling during generation. Default is False (greedy decoding).
        
        Returns:
            str: The summarized version of the input text.
        
        Example:
            summary = summarizer.summarize("Long text here...", max_length=100, min_length=50)
        """
        self.logger.info("Starting summarization process...")
        
        try:
            # Generate the summary using the summarizer pipeline
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            
            # Return the summary text
            return summary[0]["summary_text"]
        
        except Exception as e:
            # Log any error that occurs during summarization
            self.logger.error(f"Error during summarization: {e}")
            return None

    def summarize_texts(self, texts, max_length=150, min_length=100, do_sample=False):
        """
        Summarizes a list of texts and logs the process for each text.
        
        Args:
            texts (list): A list of strings, where each string is a body of text to summarize.
            max_length (int): The maximum length of each summary. Default is 150 characters.
            min_length (int): The minimum length of each summary. Default is 100 characters.
            do_sample (bool): Whether to use sampling during generation for each text. Default is False.
        
        Returns:
            list: A list of summarized texts.
        
        Example:
            summarized_texts = summarizer.summarize_texts(corpi, max_length=150, min_length=100)
        """
        self.logger.info("Starting batch summarization process...")
        summarized_texts = []

        for index, text in enumerate(texts):
            self.logger.info(f"Summarizing text {index + 1}/{len(texts)}...")
            summary = self.summarize(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            
            if summary:  # If summary is not None, append to the results
                summarized_texts.append(summary)
                self.logger.info(f"Summarized text {index + 1}: {summary[:100]}...")  # Log the first 100 chars of the summary
            else:
                self.logger.error(f"Failed to summarize text {index + 1}.")

        return summarized_texts