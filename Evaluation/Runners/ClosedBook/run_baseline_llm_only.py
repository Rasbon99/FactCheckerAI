import os
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_prompt_instructions
from Database.data_entities import Claim, Answer, Experiment

dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)
logger = Logger("ClosedBook-Baseline").get_logger()


def get_closed_book_verdict(claim_text, prompt_instructions, metadata_context=""):
    """Asks the LLM to verify the claim using ONLY its internal weights, providing context if available."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY your internal knowledge. 

    {prompt_instructions}

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
    base_instructions = get_dataset_prompt_instructions(active_dataset)
    prompt_instructions = base_instructions.replace(
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

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
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
                        "\nCONTEXT PROVIDED FOR THIS CLAIM:\n" + "\n".join(meta_parts)
                    )

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[ClosedBook] " + claim_text[:30] + "...",
                summary="Tested without any external evidence.",
                claim_id=claim_id,
            )

            # --- GENERATION STEP ---
            t0 = time.time()
            query_result, tokens_used = get_closed_book_verdict(
                claim_text, prompt_instructions, metadata_context
            )
            latency_generation = time.time() - t0

            # Verdict Parsing
            try:
                if query_result and "VERDICT:" in query_result:
                    predicted_label = (
                        query_result.split("REASONING:")[0]
                        .replace("VERDICT:", "")
                        .strip()
                    )
                else:
                    predicted_label = "Error: Unstructured Response"
            except Exception:
                predicted_label = "Parsing Error"

            logger.info(f"Closed-Book Verdict: {predicted_label}")

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
                tokens={"preprocessor": 0, "retrieval": 0, "generation": tokens_used},
                calls={"preprocessor": 0, "retrieval": 0, "generation": 1},
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
            )

            successful_runs += 1

            time.sleep(2)

    except Exception as e:
        logger.error(f"Error during execution: {e}")

    logger.info("=" * 20)
    logger.info("CLOSED-BOOK BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_closed_book_baseline()
