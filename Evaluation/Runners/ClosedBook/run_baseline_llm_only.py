import os
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_response_format_instructions
from Database.data_entities import Claim, Answer, Experiment

dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)
logger = Logger("ClosedBook-Baseline").get_logger()


def get_closed_book_verdict(
    claim_text, response_format_instructions, metadata_context=""
):
    """Asks the LLM to verify the claim using ONLY its internal weights, providing context if available."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY your internal knowledge. 

    {response_format_instructions}

    CLAIM: {claim_text}
    {metadata_context}
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.0,  # Zero temperature for maximum factual consistency
        max_tokens=200,
    )

    result_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    return result_text, tokens_used


def run_closed_book_baseline():
    dataset_manager = DatasetManager()

    metadata = dataset_manager.get_experiment_metadata(environment="closed_book")
    active_dataset = metadata["dataset_name"]
    use_meta = metadata["use_metadata"]

    # Tweak the instructions slightly since this baseline has no "provided evidence"
    base_instructions = get_dataset_response_format_instructions(active_dataset)
    response_format_instructions = base_instructions.replace(
        "citing the provided evidence", "based on your internal knowledge"
    )

    logger.info(
        f"Starting Baseline (Closed-Book / LLM-Only) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Model: {GROQ_MODEL}")
    logger.info(f"Using Metadata Context: {use_meta}")

    successful_runs = 0
    failed_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_id = None
            claim_text = ""
            ground_truth = ""
            metadata_context = ""
            current_stage = "claim_setup"

            latency_generation = 0.0
            tokens_used = 0
            query_result = None

            try:
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                # --- OPTIONAL METADATA INJECTION ---
                metadata_context = ""
                if active_dataset == "AVERITEC" and use_meta:
                    speaker = data.get("speaker", "")
                    date = data.get("claim_date", "")
                    location_ISO_code = data.get("location_ISO_code", "")
                    reporting_source = data.get("reporting_source", "")

                    meta_parts = []
                    if speaker:
                        meta_parts.append(f"- Speaker: {speaker}")
                    if location_ISO_code:
                        meta_parts.append(f"- Location: {location_ISO_code}")
                    if date:
                        meta_parts.append(f"- Date: {date}")
                    if reporting_source:
                        meta_parts.append(f"- Source: {reporting_source}")

                    if meta_parts:
                        metadata_context = (
                            "\nCONTEXT PROVIDED FOR THIS CLAIM:\n"
                            + "\n".join(meta_parts)
                        )

                logger.info(
                    f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}"
                )
                logger.info(f"Ground Truth: {ground_truth}")

                claim_id = str(uuid.uuid4())

                Claim(
                    text=claim_text,
                    title="[ClosedBook] " + claim_text[:30] + "...",
                    claim_id=claim_id,
                )

                # --- GENERATION STEP ---
                current_stage = "generation"

                t0 = time.time()
                query_result, tokens_used = get_closed_book_verdict(
                    claim_text, response_format_instructions, metadata_context
                )
                latency_generation = time.time() - t0

                # Verdict Parsing
                current_stage = "parsing"

                error_type = None
                error_message = None
                error_stage = None

                try:
                    if not query_result or not query_result.strip():
                        predicted_label = "Error"
                        error_type = "EmptyLLMResponse"
                        error_message = "The LLM returned an empty response."
                        error_stage = "parsing"
                        query_result = "The LLM failed to generate a response."

                    elif "VERDICT:" in query_result and "REASONING:" in query_result:
                        predicted_label = (
                            query_result.split("REASONING:")[0]
                            .replace("VERDICT:", "")
                            .strip()
                        )

                    else:
                        predicted_label = "Error"
                        error_type = "UnstructuredResponse"
                        error_message = "The LLM response did not contain both VERDICT and REASONING."
                        error_stage = "parsing"

                except Exception as e:
                    logger.exception(f"Error parsing verdict: {e}")
                    predicted_label = "Error"
                    error_type = type(e).__name__
                    error_message = str(e)
                    error_stage = "parsing"

                logger.info(f"Closed-Book Verdict: {predicted_label}")

                current_stage = "database_logging"

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                # --- LOG TO EXPERIMENTS DATABASE ---
                Experiment(
                    claim_id=claim_id,
                    predicted_label=predicted_label,
                    ground_truth=ground_truth,
                    latencies={
                        "preprocessor": 0.0,
                        "retrieval": 0.0,
                        "generation": latency_generation,
                    },
                    tokens={
                        "preprocessor": 0,
                        "retrieval": 0,
                        "generation": tokens_used,
                    },
                    calls={
                        "preprocessor": 0,
                        "retrieval": 0,
                        "generation": 1,
                    },
                    evidence_data={
                        "claim_text": claim_text,
                        "raw_sources": [],
                        "query_result": query_result,
                    },
                    system_type="LLM-Only",
                    environment=metadata["environment"],
                    dataset_name=metadata["dataset_name"],
                    experiment_type=metadata["experiment_type"],
                    use_metadata=use_meta,
                    error_details=(
                        f"Stage: {error_stage} | "
                        f"Type: {error_type} | "
                        f"Message: {error_message}"
                        if error_type is not None
                        else None
                    ),
                )

                if predicted_label == "Error":
                    failed_runs += 1
                else:
                    successful_runs += 1

                time.sleep(2)

            except Exception as e:
                logger.exception(f"Error processing claim {line_number + 1}: {e}")

                try:
                    if claim_id is not None:
                        Experiment(
                            claim_id=claim_id,
                            predicted_label="Error",
                            ground_truth=ground_truth,
                            latencies={
                                "preprocessor": 0.0,
                                "retrieval": 0.0,
                                "generation": latency_generation,
                            },
                            tokens={
                                "preprocessor": 0,
                                "retrieval": 0,
                                "generation": tokens_used,
                            },
                            calls={
                                "preprocessor": 0,
                                "retrieval": 0,
                                "generation": 1 if query_result is not None else 0,
                            },
                            evidence_data={
                                "claim_text": claim_text,
                                "raw_sources": [],
                                "query_result": query_result,
                            },
                            system_type="LLM-Only",
                            environment=metadata["environment"],
                            dataset_name=metadata["dataset_name"],
                            experiment_type=metadata["experiment_type"],
                            use_metadata=use_meta,
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
                continue

    except Exception as e:
        logger.exception(f"Fatal error during experiment: {e}")

    logger.info("=" * 20)
    logger.info("CLOSED-BOOK BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info(f"Failed processing: {failed_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_closed_book_baseline()
