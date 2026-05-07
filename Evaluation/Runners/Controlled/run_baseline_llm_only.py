import os
import json
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Evaluation.Utils.dataset_manager import DatasetManager
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger("LLM-Only-Baseline").get_logger()


def get_llm_only_verdict(claim_text, prompt_instructions, nei_label):
    """Asks the LLM to verify the claim using ONLY its internal weights."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY your internal knowledge. 
    Do not assume any external context. If you do not know the answer with absolute certainty, answer exactly: {nei_label}.

    {prompt_instructions}

    Claim: {claim_text}
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


def run_llm_baseline():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("ZeroShot")

    # Define the exact NEI label to prevent LLM hallucination
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    # Tweak the instructions slightly since this baseline has no "provided evidence"
    base_instructions = dataset_manager.get_prompt_instructions()
    prompt_instructions = base_instructions.replace(
        "citing the provided evidence", "based on your internal knowledge"
    )

    logger.info(f"Starting Baseline 1 (LLM-Only) with {MAX_CLAIMS_TO_TEST} claims...")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Model: {GROQ_MODEL}")

    successful_runs = 0

    try:
        # Ask the DatasetManager for the claims
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())
            tracker = ExperimentTracker(
                claim_id=claim_id,
                ground_truth=ground_truth,
                system_type="LLM-Only",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[LLM-Only] " + claim_text[:30] + "...",
                summary="Tested without any external evidence.",
                claim_id=claim_id,
            )

            # --- UPDATED: Pass nei_label to the LLM Call ---
            def run_llm():
                res_text, toks = get_llm_only_verdict(
                    claim_text, prompt_instructions, nei_label
                )
                return (res_text, None), {"total": toks, "calls": 1}

            query_result = tracker.run_stage("generation", run_llm)

            if isinstance(query_result, tuple):
                query_result = query_result[0]

            # 3. Verdict Parsing
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

            logger.info(f"LLM Verdict: {predicted_label}")

            # 4. Create Answer Entity for the UI
            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # 5. Log Experiment Metrics
            tracker.finalize(
                predicted_label,
                {
                    "claim_text": claim_text,
                    "raw_sources": [],  # No sources used!
                    "query_result": query_result,
                },
            )

            successful_runs += 1

            # Sleep for 2 seconds to avoid hitting Groq API rate limits
            time.sleep(2)

    except Exception as e:
        logger.error(f"Error during execution: {e}")

    logger.info("=" * 20)
    logger.info("LLM-ONLY BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_llm_baseline()
