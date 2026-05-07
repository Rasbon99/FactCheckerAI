import os
import json
import requests
import time
import dotenv
from log import Logger

dotenv.load_dotenv("key.env", override=True)

# 1. Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
BACKEND_URL = os.getenv("BACKEND_API_URL")
API_URL = f"{BACKEND_URL}/run_pipeline"

MAX_CLAIMS_TO_TEST = 5
logger = Logger(__name__).get_logger()


def run_experiment():
    logger.info(f"Starting Experiment B (Open Web) with {MAX_CLAIMS_TO_TEST} claims...")
    logger.info(f"Reading dataset from: {DATASET_PATH}")
    logger.info(f"Sending requests to: {API_URL}")

    successful_runs = 0
    failed_runs = 0

    # 2. Open and read the FEVER dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):

                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                logger.info(
                    f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Processing: {claim_text[:50]}..."
                )
                logger.info(f"Ground Truth: {ground_truth}")

                # 3. Prepare the payload
                payload = {
                    "text": claim_text,
                    "ground_truth": ground_truth,
                    "dataset_setting": "FEVER-OpenWeb",
                }

                # 4. Send the request to FoxAI
                try:
                    response = requests.post(API_URL, json=payload, timeout=180)

                    if response.status_code == 200:
                        logger.info("Success! FoxAI Verdict generated.")
                        successful_runs += 1
                    else:
                        logger.error(
                            f"Backend Error {response.status_code}: {response.text}"
                        )
                        failed_runs += 1

                except requests.exceptions.Timeout:
                    logger.warning(
                        "Request timed out! The backend took too long (likely a slow scrape or Ollama freeze). Skipping to next claim."
                    )
                    failed_runs += 1
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    failed_runs += 1

                time.sleep(5)

    except FileNotFoundError:
        logger.error(
            f"ERROR: Could not find the dataset at {DATASET_PATH}. Please check your .env file!"
        )
        return

    logger.info("=" * 40)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info(f"Successful processing: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("Check your SQLite 'experiments' table to see the logged metrics!")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_experiment()
