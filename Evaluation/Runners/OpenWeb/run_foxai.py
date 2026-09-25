import os
import time
import requests
import dotenv
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_prompt_instructions
from Database.data_entities import Experiment

# Load environment variables
dotenv.load_dotenv("key.env", override=False)

BACKEND_URL = os.getenv("BACKEND_API_URL")
API_URL = f"{BACKEND_URL}/run_pipeline"

MAX_CLAIMS_TO_TEST = 5
logger = Logger("FoxAI-OpenWeb").get_logger()


def run_experiment():
    dataset_manager = DatasetManager()
    metadata = dataset_manager.get_experiment_metadata(environment="open_web")
    use_meta = metadata["use_metadata"]

    prompt_instructions = get_dataset_prompt_instructions(metadata["dataset_name"])

    logger.info(
        f"Starting FoxAI GraphRAG (Open Web) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {metadata['dataset_name']}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Metadata Super Query: {use_meta}")
    logger.info(f"Sending requests to: {API_URL}")

    successful_runs = 0
    failed_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            search_query = claim_text
            if metadata["dataset_name"] == "AVERITEC" and use_meta:
                search_query = dataset_manager.build_search_query(data)

            logger.info(
                f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Processing: {claim_text[:50]}..."
            )
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")

            payload = {
                "text": claim_text,
                "search_query": search_query,
                "prompt_instructions": prompt_instructions,
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=300)

                if response.status_code == 200:
                    res_data = response.json()

                    Experiment(
                        claim_id=res_data.get("claim_id"),
                        predicted_label=res_data.get("predicted_label"),
                        ground_truth=ground_truth,
                        latencies=res_data.get("metrics", {}).get("latencies", {}),
                        tokens=res_data.get("metrics", {}).get("tokens", {}),
                        calls=res_data.get("metrics", {}).get("calls", {}),
                        evidence_data=res_data.get("evidence_data", {}),
                        system_type="FoxAI-GraphRAG",
                        environment="open_web",
                        dataset_name=metadata["dataset_name"],
                        experiment_type=metadata["experiment_type"],
                        use_metadata=metadata["use_metadata"],
                    )

                    logger.info(
                        "Success! FoxAI Verdict generated and logged by baseline script."
                    )
                    successful_runs += 1
                else:
                    logger.error(
                        f"Backend Error {response.status_code}: {response.text}"
                    )
                    failed_runs += 1

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timed out! Scraping and Graph building took too long. Skipping to next claim."
                )
                failed_runs += 1
            except Exception as e:
                logger.error(f"Request failed: {e}")
                failed_runs += 1

            time.sleep(15)

    except Exception as e:
        logger.error(f"Fatal Error during execution: {e}")
        return

    logger.info("=" * 20)
    logger.info("FOXAI (OPEN WEB) COMPLETE!")
    logger.info(f"Successful processing: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_experiment()
