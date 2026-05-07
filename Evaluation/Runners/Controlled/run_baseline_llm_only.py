import os
import json
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger(__name__).get_logger()


def get_llm_only_verdict(claim_text):
    """Asks the LLM to verify the claim using ONLY its internal weights."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY your internal knowledge. 
    Do not assume any external context. If you do not know the answer with absolute certainty, answer NOT ENOUGH INFO.

    You must format your response EXACTLY like this:
    VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
    REASONING: [Your brief explanation based on your internal knowledge]

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
    logger.info(f"Starting Baseline 1 (LLM-Only) with {MAX_CLAIMS_TO_TEST} claims...")
    logger.info(f"Using Model: {GROQ_MODEL}")

    successful_runs = 0

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                logger.info(
                    f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}"
                )
                logger.info(f"Ground Truth: {ground_truth}")

                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id,
                    ground_truth=ground_truth,
                    system_type="LLM-Only",  # Labels the architecture!
                    dataset_setting="FEVER-Controlled",  # Labels the dataset!
                )

                # 1. Create a Claim so it shows up in the UI sidebar
                Claim(
                    text=claim_text,
                    title="[LLM-Only] " + claim_text[:30] + "...",
                    summary="Tested without any external evidence.",
                    claim_id=claim_id,
                )

                # 2. Call the LLM
                def run_llm():
                    res_text, toks = get_llm_only_verdict(claim_text)
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

    except FileNotFoundError:
        logger.error(f"Could not find dataset at {DATASET_PATH}.")

    logger.info("=" * 40)
    logger.info("LLM-ONLY BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_llm_baseline()
