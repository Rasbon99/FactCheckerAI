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

    def predict(self, texts, claim):
        """
        Makes predictions on the given texts using zero-shot classification.

        :param texts: A list of texts to classify
        :param claim: The claim to evaluate
        :return: The average classification results across all texts
        """
        
        # TODO: Aggiungere i pesi dei ratings di NewsGuard
        try:
            self.logger.info(f"Starting prediction for {len(texts)} texts.")
            question = "The News: '" + claim + "' is {}"

            # Classify texts with custom labels
            all_results = self.zero_shot_classifier(
                texts,
                candidate_labels=["confirmed", "neutral", "denied"],
                hypothesis_template=question
            )

            # Initialize counters to accumulate scores for each label
            confirm_score = 0
            neutral_score = 0
            deny_score = 0

            # Loop through results and accumulate scores
            for result in all_results:
                confirm_score += result['scores'][0]  # Assuming 'confirm' is the first label
                neutral_score += result['scores'][1]  # Assuming 'neutral' is the second label
                deny_score += result['scores'][2]    # Assuming 'deny' is the third label

            # Calculate the average score for each label
            num_texts = len(texts)
            avg_confirm_score = confirm_score / num_texts
            avg_neutral_score = neutral_score / num_texts
            avg_deny_score = deny_score / num_texts

            # Log the average results
            self.logger.info(f"Average scores - Confirm: {avg_confirm_score}, Neutral: {avg_neutral_score}, Deny: {avg_deny_score}")

            # Return the average results
            return {
                "confirm": avg_confirm_score,
                "neutral": avg_neutral_score,
                "deny": avg_deny_score
            }

        except Exception as e:
            self.logger.error(f"Prediction failed. Error: {e}")
            return None