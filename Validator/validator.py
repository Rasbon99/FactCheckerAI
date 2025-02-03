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

    def predict(self, texts, claim, scores):
        """
        Makes predictions on the provided texts using zero-shot classification.

        Args:
            texts (list): A list of texts to evaluate against the claim.
            claim (str): The claim to validate through classification.
            scores (list): A list of weights for each text (e.g., based on reliability or NewsGuard ratings).

        Returns:
            dict: A dictionary containing the weighted average scores for each label:
                - "confirm" (float): Weighted average score for the "confirmed" label.
                - "neutral" (float): Weighted average score for the "neutral" label.
                - "deny" (float): Weighted average score for the "denied" label.

        Raises:
            ValueError: If the input texts or scores are not lists.
            Exception: For any error encountered during prediction.
        """
        
        # TODO: Non funziona molto bene sulle fake news
        try:
            if not isinstance(texts, list) or not isinstance(scores, list):
                raise ValueError("Both 'texts' and 'scores' parameters must be lists.")
            
            if len(texts) != len(scores):
                raise ValueError("The number of texts and scores must be the same.")
            
            self.logger.info(f"Starting prediction for {len(texts)} texts.")
            
            # Modifica il template dell'ipotesi
            hypothesis_template = f"The article provides evidence to determine that the claim: '{claim}' is {{}}."

            # Perform zero-shot classification with custom labels
            all_results = self.zero_shot_classifier(
                texts,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template=hypothesis_template
            )

            # Initialize counters to accumulate weighted scores for each label
            weighted_confirm_score = 0
            weighted_neutral_score = 0
            weighted_deny_score = 0
            total_weight = 0

            # Loop through results and accumulate weighted scores
            for result, weight in zip(all_results, scores):
                # Ensure the result contains valid scores
                if 'scores' not in result:
                    self.logger.warning(f"Missing scores in result: {result}")
                    continue

                # Multiply each score by its corresponding weight and accumulate
                weighted_confirm_score += result['scores'][0] * weight
                weighted_neutral_score += result['scores'][1] * weight
                weighted_deny_score += result['scores'][2] * weight
                total_weight += weight

            # Ensure total weight is not zero to avoid division by zero
            if total_weight == 0:
                self.logger.warning("Total weight is zero, cannot compute weighted average.")
                return None

            # Calculate the weighted average score for each label
            weighted_avg_confirm_score = (weighted_confirm_score / total_weight) * 100
            weighted_avg_neutral_score = (weighted_neutral_score / total_weight) * 100
            weighted_avg_deny_score = (weighted_deny_score / total_weight) * 100

            self.logger.info(f"Weighted average scores - Confirm: {weighted_avg_confirm_score:.2f}%, Neutral: {weighted_avg_neutral_score:.2f}%, Deny: {weighted_avg_deny_score:.2f}%")

            # Return the weighted average results
            return {
                "confirmed": weighted_avg_confirm_score,
                "uncertain": weighted_avg_neutral_score,
                "denied": weighted_avg_deny_score
            }

        except Exception as e:
            self.logger.error(f"Prediction failed. Error: {e}")
            return None

    def filter_sources(self, claim, sources, score_threshold=0.7):
        """
        Filters the sources by evaluating whether they are correlated to the given claim using a zero-shot classifier.
        Returns a list of dictionaries with correlated source information.
        """
        self.logger.info(f"Evaluating {len(sources)} sources compared to the claim: '{claim}'.")
        
        correlated_sources = []  
        excluded_sources = 0  
        
        for source in sources:
            # Template for the hypothesis to check if the source is correlated to the claim
            question = f"The claim: '{claim}' is {{}} with the information provided in the article"
            
            # Run zero-shot classification to check if the source is correlated to the claim
            all_results = self.zero_shot_classifier(
                source['body'],  
                candidate_labels=["correlated", "not correlated"],  
                hypothesis_template=question
            )

            # Get the scores for "Correlated" and "Not Correlated"
            correlated_score = all_results["scores"][0]  
            not_correlated_score = all_results["scores"][1]  

            # Convert scores to percentage
            correlated_score_percent = correlated_score * 100
            not_correlated_score_percent = not_correlated_score * 100

            # Limit the title to the first few words 
            title_words = source['title'].split()
            truncated_title = ' '.join(title_words[:5])  

            # Log the result of the classification with percentage scores and truncated title
            self.logger.info(f"Source: {truncated_title} - Correlated score: {correlated_score_percent:.2f}%, Not Correlated score: {not_correlated_score_percent:.2f}%")

            # If the "Correlated" score is above the threshold, we consider it correlated
            if correlated_score >= score_threshold:
                correlated_sources.append({
                    'title': source['title'],
                    'site': source['site'],
                    'url': source['url'],
                    'body': source['body'],
                    'score': source['score'],  
                    'topic': source.get('topic', 'N/A'),  
                    'entities': source.get('entities', [])  
                })
                self.logger.info(f"Source '{source['title']}' is correlated to the claim.")
            else:
                excluded_sources += 1  # Increment counter for excluded sources
                self.logger.info(f"Source '{source['title']}' is not correlated to the claim.")

        # Log the number of correlated and excluded sources
        self.logger.info(f"Number of correlated sources: {len(correlated_sources)}")
        self.logger.info(f"Number of excluded sources: {excluded_sources}")

        return correlated_sources