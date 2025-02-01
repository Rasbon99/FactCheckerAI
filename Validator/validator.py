import dotenv
import os
from transformers import pipeline
from log import Logger

class Validator:
    def __init__(self, env_file="key.env"):
        """
        Initializes the Validator class with the model for zero-shot classification and sets up logging.

        Args:
            env_file (str): The environment file containing the model path. Default is "key.env".
        
        Attributes:
            model_name (str): The name or path of the model used for zero-shot classification.
            logger (Logger): Instance of the custom Logger for logging messages.
            zero_shot_classifier (Pipeline): Zero-shot classification pipeline from Hugging Face.

        Returns:
            None
        
        Raises:
            FileNotFoundError: If the specified environment file is not found.
            ValueError: If the model path is not specified in the environment file.
        """
        dotenv.load_dotenv(env_file, override=True)
        self.model_name = os.getenv("MODEL_PATH_DEBERTA")
        
        if not self.model_name:
            raise ValueError("MODEL_PATH_DEBERTA is not specified in the environment file.")

        self.logger = Logger(self.__class__.__name__).get_logger()

        # Create zero-shot classification pipeline
        self.zero_shot_classifier = pipeline("zero-shot-classification", model=self.model_name)

        self.logger.info(f"Validator initialized with model: {self.model_name}")

    def predict(self, texts, claim):
        """
        Makes predictions on the provided texts using zero-shot classification.

        Args:
            texts (list): A list of texts to evaluate against the claim.
            claim (str): The claim to validate through classification.

        Returns:
            dict: A dictionary containing the average scores for each label:
                - "confirm" (float): Average score for the "confirmed" label.
                - "neutral" (float): Average score for the "neutral" label.
                - "deny" (float): Average score for the "denied" label.

        Raises:
            ValueError: If the input texts are not a list.
            Exception: For any error encountered during prediction.
        """
        # TODO: Add weighting factors based on NewsGuard ratings
        try:
            if not isinstance(texts, list):
                raise ValueError("The 'texts' parameter must be a list of strings.")
            
            self.logger.info(f"Starting prediction for {len(texts)} texts.")
            question = f"The News: '{claim}' is {{}}"

            # Perform zero-shot classification with custom labels
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
                confirm_score += result['scores'][0]  # Assuming 'confirmed' is the first label
                neutral_score += result['scores'][1]  # Assuming 'neutral' is the second label
                deny_score += result['scores'][2]    # Assuming 'denied' is the third label

            # Calculate the average score for each label
            num_texts = len(texts)
            avg_confirm_score = (confirm_score / num_texts) * 100
            avg_neutral_score = (neutral_score / num_texts) * 100
            avg_deny_score = (deny_score / num_texts) * 100

            self.logger.info(f"Average scores - Confirm: {avg_confirm_score:.2f}%, Neutral: {avg_neutral_score:.2f}%, Deny: {avg_deny_score:.2f}%")

            # Return the average results
            return {
                "confirm": avg_confirm_score,
                "neutral": avg_neutral_score,
                "deny": avg_deny_score
            }

        except Exception as e:
            self.logger.error(f"Prediction failed. Error: {e}")
            return None