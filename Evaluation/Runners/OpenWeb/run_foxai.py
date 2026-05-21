import os
import time
import requests
import dotenv
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.dataset_manager import DatasetManager

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL")
API_URL = f"{BACKEND_URL}/run_pipeline"

MAX_CLAIMS_TO_TEST = 5
logger = Logger("FoxAI-OpenWeb").get_logger()

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"


def run_experiment():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("open-web")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting FoxAI GraphRAG (Open Web) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")
    logger.info(f"Sending requests to: {API_URL}")

    successful_runs = 0
    failed_runs = 0

    try:
        # 1. Use the Smart Manager to load the data
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            # --- 2. CONDITIONAL METADATA QUERY ---
            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            logger.info(
                f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Processing: {claim_text[:50]}..."
            )
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")

            # --- 3. Prepare the Enriched Payload ---
            # We must send both the raw claim AND the search query to the backend!
            payload = {
                "text": claim_text,  # For the final LLM Verifier
                "search_query": search_query,  # For the DuckDuckGo Scraper inside FoxAI
                "ground_truth": ground_truth,
                "dataset_setting": tracker_env_name,
                "active_dataset": active_dataset,
                "prompt_instructions": prompt_instructions,
                "nei_label": nei_label,
            }

            # --- 4. Send the request to FoxAI Backend ---
            try:
                # Graph building takes time! Increased timeout to 5 minutes (300s).
                response = requests.post(API_URL, json=payload, timeout=300)

                if response.status_code == 200:
                    logger.info(
                        "Success! FoxAI Verdict generated and logged by backend."
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

            # Sleep to ensure DuckDuckGo limits on the backend aren't tripped
            time.sleep(15)

    except Exception as e:
        logger.error(f"Fatal Error during execution: {e}")
        return

    logger.info("=" * 20)
    logger.info("FOXAI (OPEN WEB) COMPLETE!")
    logger.info(f"Successful processing: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("Check your SQLite 'experiments' table to see the logged metrics!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_experiment()
