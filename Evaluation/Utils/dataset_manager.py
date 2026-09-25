import os
import json
import dotenv
from log import Logger

logger = Logger("dataset_manager").get_logger()


class DatasetManager:
    def __init__(self, env_file="key.env"):
        dotenv.load_dotenv(env_file, override=False)

        self.active_dataset = os.getenv("EXPERIMENT_ACTIVE_DATASET", "FEVER").upper()

        if self.active_dataset == "AVERITEC":
            self.use_metadata = (
                os.getenv("AVERITEC_USE_METADATA", "false").lower() == "true"
            )
        else:
            self.use_metadata = None

        self.fever_path = os.getenv(
            "FEVER_DATASET_PATH", "Datasets/FEVER/fever_dev_dataset.jsonl"
        )
        self.averitec_path = os.getenv(
            "AVERITEC_DATASET_PATH", "Datasets/AVeriTeC/averitec_dev_dataset.json"
        )

    def load_data(self, max_claims=None):
        """Intelligently loads either JSONL (FEVER) or JSON Array (AVeriTeC).
        If max_claims is None, it loads the entire dataset."""
        data_list = []

        if self.active_dataset == "FEVER":
            logger.info(f"Loading FEVER Dataset from {self.fever_path}...")
            with open(self.fever_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_claims is not None and i >= max_claims:
                        break
                    claim_data = json.loads(line)
                    claim_data["internal_id"] = str(i)
                    data_list.append(claim_data)

        elif self.active_dataset == "AVERITEC":
            logger.info(f"Loading AVeriTeC Dataset from {self.averitec_path}...")
            with open(self.averitec_path, "r", encoding="utf-8") as f:
                full_array = json.load(f)

                for i, claim_data in enumerate(full_array):
                    if max_claims is not None and i >= max_claims:
                        break
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

        final_query_parts = [claim]

        if speaker:
            final_query_parts.append(speaker)
        if location_ISO_code:
            final_query_parts.append(location_ISO_code)
        if reporting_source:
            final_query_parts.append(f"Source: {reporting_source}")

        return " ; ".join(final_query_parts)

    def get_experiment_metadata(self, environment="open_web"):
        """
        Returns the specific breakdown of the current experiment configuration
        for precise database logging across four normalized columns.
        """
        dataset_name = self.active_dataset

        experiment_type = os.getenv("EXPERIMENT_TYPE", "").strip().lower()
        if not experiment_type:
            experiment_type = "standard"

        return {
            "environment": environment,
            "dataset_name": dataset_name,
            "experiment_type": experiment_type,
            "use_metadata": self.use_metadata,
        }
