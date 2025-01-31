import dotenv
import os
from transformers import pipeline

from log import Logger

class Validator:
    def __init__(self, env_file="key.env"):
        """
        Initializes the Validator class with the model name and sets up logging.

        :param env_file: The environment file to load the credentials from (default: "key.env")
        """
        dotenv.load_dotenv(env_file, override=True)
        self.model_name = os.getenv("MODEL_PATH_DEBERTA")
        
        self.logger = Logger(self.__class__.__name__).get_logger()

        # Create zero-shot classification pipeline
        self.zero_shot_classifier = pipeline("zero-shot-classification", model=self.model_name)

        self.logger.info(f"Validator initialized with model: {self.model_name}")

    def predict(self, texts, hypothesis_template: str):
        """
        Makes predictions on the given texts using zero-shot classification.

        :param texts: A list of texts to classify
        :param hypothesis_template: The sentence to use as the hypothesis for classification
        :return: The classification results
        """
        try:
            self.logger.info(f"Starting prediction for {len(texts)} texts.")
            results = self.zero_shot_classifier(texts, candidate_labels=["entailment", "semi_entailment", "not_entailment"], hypothesis_template=hypothesis_template)
            self.logger.info(f"Prediction completed successfully.")
            return results
        except Exception as e:
            self.logger.error(f"Prediction failed. Error: {e}")
            return None