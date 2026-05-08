import os
import json
import dotenv
from log import Logger

logger = Logger("dataset_manager").get_logger()


class DatasetManager:
    def __init__(self, env_file="key.env"):
        dotenv.load_dotenv(env_file, override=True)

        # Load controls
        self.active_dataset = os.getenv("EXPERIMENT_ACTIVE_DATASET", "FEVER").upper()
        self.use_metadata = (
            os.getenv("AVERITEC_USE_METADATA", "false").lower() == "true"
        )

        # Load paths
        self.fever_path = os.getenv(
            "FEVER_DATASET_PATH", "Datasets/FEVER/fever_dev_dataset.jsonl"
        )
        self.averitec_path = os.getenv(
            "AVERITEC_DATASET_PATH", "Datasets/AVeriTeC/averitec_dev_dataset.json"
        )

    def load_data(self, max_claims=5):
        """Intelligently loads either JSONL (FEVER) or JSON Array (AVeriTeC)"""
        data_list = []

        if self.active_dataset == "FEVER":
            logger.info(f"Loading FEVER Dataset from {self.fever_path}...")
            with open(self.fever_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_claims:
                        break
                    claim_data = json.loads(line)
                    # FEVER doesn't have an explicit numeric ID for the knowledge store, so we add an index
                    claim_data["internal_id"] = str(i)
                    data_list.append(claim_data)

        elif self.active_dataset == "AVERITEC":
            logger.info(f"Loading AVeriTeC Dataset from {self.averitec_path}...")
            with open(self.averitec_path, "r", encoding="utf-8") as f:
                full_array = json.load(f)

                for i, claim_data in enumerate(full_array):
                    if i >= max_claims:
                        break
                    # The AVeriTeC knowledge store uses the array index (0, 1, 2) as the claim_id
                    claim_data["internal_id"] = str(i)
                    data_list.append(claim_data)

        return data_list

    def build_search_query(self, data):
        """Builds an enriched query using metadata for AVeriTeC."""
        if self.active_dataset != "AVERITEC":
            return data.get("claim", "")

        claim = data.get("claim", "")
        speaker = data.get("speaker", "")
        location_ISO_code = data.get("location_ISO_code", "")
        reporting_source = data.get("reporting_source", "")

        # Add everything to a single list
        final_query_parts = [claim]

        if speaker:
            final_query_parts.append(speaker)
        if location_ISO_code:
            final_query_parts.append(location_ISO_code)
        if reporting_source:
            final_query_parts.append(f"Source: {reporting_source}")

        # Join them all perfectly with semicolons
        return " ; ".join(final_query_parts)

    def get_prompt_instructions(self):
        """Returns the exact prompt rules based on the active dataset."""
        if self.active_dataset == "FEVER":
            return """
            You must format your response EXACTLY like this:
            VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
            REASONING: [Your brief explanation citing the provided evidence]"""
        else:
            return """
            You must format your response EXACTLY like this:
            VERDICT: [Supported or Refuted or Not Enough Evidence or Conflicting Evidence/Cherry-picking]
            REASONING: [Your brief explanation citing the provided evidence]
            
            RULES FOR VERDICT:
            - Supported: The evidence clearly proves the claim is true.
            - Refuted: The evidence clearly proves the claim is false.
            - Not Enough Evidence: The evidence does not contain the information needed to judge the claim.
            - Conflicting Evidence/Cherry-picking: The claim is technically true but leaves out crucial context, is misleading, or the evidence is heavily mixed."""

    def get_tracker_dataset_name(self, environment="OpenWeb"):
        """
        Returns the correct dataset string for the Experiment Tracker.
        If an EXPERIMENT_NAME is set in the .env file (e.g., for Robustness tests), it overrides the default.
        """
        custom_name = os.getenv("EXPERIMENT_NAME")
        if custom_name:
            return custom_name

        return f"{self.active_dataset}-{environment}"
