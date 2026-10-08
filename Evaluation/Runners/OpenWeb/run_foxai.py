import os
import time
import requests
import dotenv
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_response_format_instructions
from Database.data_entities import Experiment

# Load environment variables
dotenv.load_dotenv("key.env", override=False)

BACKEND_URL = os.getenv("BACKEND_API_URL")
if not BACKEND_URL:
    raise RuntimeError("BACKEND_API_URL is not set in key.env")

API_URL = f"{BACKEND_URL}/run_pipeline"

MAX_CLAIMS_TO_TEST = 5
logger = Logger("FoxAI-OpenWeb").get_logger()


def run_experiment():
    dataset_manager = DatasetManager()
    metadata = dataset_manager.get_experiment_metadata(environment="open_web")
    use_meta = metadata["use_metadata"]

    response_format_instructions = get_dataset_response_format_instructions(
        metadata["dataset_name"]
    )

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
            claim_id = None
            claim_text = ""
            ground_truth = ""
            search_query = ""
            current_stage = "claim_setup"

            latencies = {}
            tokens = {}
            calls = {}
            evidence_data = {}

            try:
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
                    "response_format_instructions": response_format_instructions,
                    "preserve_failed_claim": True,
                }

                current_stage = "backend_request"

                response = requests.post(API_URL, json=payload, timeout=300)

                if response.status_code == 200:
                    res_data = response.json()

                    claim_id = res_data.get("claim_id")
                    latencies = res_data.get("metrics", {}).get("latencies", {})
                    tokens = res_data.get("metrics", {}).get("tokens", {})
                    calls = res_data.get("metrics", {}).get("calls", {})
                    evidence_data = res_data.get("evidence_data", {})

                    current_stage = "database_logging"

                    Experiment(
                        claim_id=claim_id,
                        predicted_label=res_data.get("predicted_label"),
                        ground_truth=ground_truth,
                        latencies=latencies,
                        tokens=tokens,
                        calls=calls,
                        evidence_data=evidence_data,
                        system_type="FoxAI-GraphRAG",
                        environment="open_web",
                        dataset_name=metadata["dataset_name"],
                        experiment_type=metadata["experiment_type"],
                        use_metadata=metadata["use_metadata"],
                        error_details=None,
                    )

                    logger.info(
                        "Success! FoxAI Verdict generated and logged by baseline script."
                    )
                    successful_runs += 1

                else:
                    try:
                        response_data = response.json()
                        detail = response_data.get("detail", response.text)
                    except ValueError:
                        detail = response.text

                    logger.error(f"Backend Error {response.status_code}: {detail}")

                    if isinstance(detail, dict):
                        claim_id = detail.get("claim_id")
                        error_stage = detail.get("stage", "backend")
                        error_type = detail.get("type", "BackendError")
                        error_message = detail.get("message", str(detail))

                        metrics = detail.get("metrics", {})
                        latencies = metrics.get("latencies", {})
                        tokens = metrics.get("tokens", {})
                        calls = metrics.get("calls", {})

                    else:
                        error_stage = "backend"
                        error_type = f"HTTP{response.status_code}"
                        error_message = str(detail)

                    if claim_id is not None:
                        Experiment(
                            claim_id=claim_id,
                            predicted_label="Error",
                            ground_truth=ground_truth,
                            latencies=latencies,
                            tokens=tokens,
                            calls=calls,
                            evidence_data={
                                "claim_text": claim_text,
                                "search_query": search_query,
                            },
                            system_type="FoxAI-GraphRAG",
                            environment="open_web",
                            dataset_name=metadata["dataset_name"],
                            experiment_type=metadata["experiment_type"],
                            use_metadata=metadata["use_metadata"],
                            error_details=(
                                f"Stage: {error_stage} | "
                                f"Type: {error_type} | "
                                f"Message: {error_message}"
                            ),
                        )

                    failed_runs += 1

            except requests.exceptions.Timeout as e:
                logger.warning(
                    "Request timed out! Scraping and Graph building took too long. Skipping to next claim."
                )
                logger.exception(f"Backend request timeout: {e}")
                failed_runs += 1

            except requests.exceptions.RequestException as e:
                logger.exception(f"Request failed: {e}")
                failed_runs += 1

            except Exception as e:
                logger.exception(
                    f"Unexpected error while processing claim {line_number + 1}: {e}"
                )

                if claim_id is not None:
                    try:
                        Experiment(
                            claim_id=claim_id,
                            predicted_label="Error",
                            ground_truth=ground_truth,
                            latencies=latencies,
                            tokens=tokens,
                            calls=calls,
                            evidence_data={
                                "claim_text": claim_text,
                                "search_query": search_query,
                            },
                            system_type="FoxAI-GraphRAG",
                            environment="open_web",
                            dataset_name=metadata["dataset_name"],
                            experiment_type=metadata["experiment_type"],
                            use_metadata=metadata["use_metadata"],
                            error_details=(
                                f"Stage: {current_stage} | "
                                f"Type: {type(e).__name__} | "
                                f"Message: {str(e)}"
                            ),
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to save failed experiment for claim {line_number + 1}"
                        )

                failed_runs += 1

            time.sleep(15)

    except Exception as e:
        logger.exception(f"Fatal Error during execution: {e}")
        return

    logger.info("=" * 20)
    logger.info("FOXAI (OPEN WEB) COMPLETE!")
    logger.info(f"Successful processing: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_experiment()
