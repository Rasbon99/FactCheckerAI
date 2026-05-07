import os
import json
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from WebScraper.scraper import Scraper
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
MAX_CLAIMS_TO_TEST = 5
logger = Logger("run_baseline_prompt_stuffing").get_logger()

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


def get_prompt_stuffing_verdict(claim_text, massive_evidence_string):
    """Asks the LLM to verify the claim using the massive wall of scraped text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer NOT ENOUGH INFO.

    You must format your response EXACTLY like this:
    VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
    REASONING: [Your brief explanation citing the provided evidence]

    EVIDENCE:
    {massive_evidence_string}

    CLAIM: {claim_text}
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=200,
    )

    result_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    return result_text, tokens_used


def run_prompt_stuffing_baseline():
    logger.info(
        f"Starting Baseline Prompt Stuffing with {MAX_CLAIMS_TO_TEST} claims..."
    )

    scraper = Scraper()

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

                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id,
                    ground_truth=ground_truth,
                    system_type="Baseline-PromptStuffing",
                    dataset_setting="FEVER-OpenWeb",
                )

                Claim(
                    text=claim_text,
                    title="[PromptStuff] " + claim_text[:30] + "...",
                    summary="Tested by stuffing 10 scraped web pages into the prompt.",
                    claim_id=claim_id,
                )

                # --- 1. Retrieval (Scraper) ---
                def run_retrieval():
                    # Unpack the new tuple and dictionary
                    raw_scraped_sources, scraper_metrics = scraper.search_and_extract(
                        claim_text, num_results=10
                    )

                    combined_evidence = ""
                    for src in raw_scraped_sources:
                        combined_evidence += f"\n--- Source: {src.get('url')} ---\n{src.get('body', '')}\n"

                    # Pass the scraper's token dictionary directly to the tracker
                    return combined_evidence, scraper_metrics

                best_evidence = tracker.run_stage("retrieval", run_retrieval)
                # Safely clear tuple bug
                if isinstance(best_evidence, tuple):
                    best_evidence = best_evidence[0]

                # --- 2. Generation (The LLM Call) ---
                def run_llm():
                    # Assuming your get_verdict returns (result_text, tokens_used)
                    res_text, toks = get_prompt_stuffing_verdict(
                        claim_text, best_evidence
                    )

                    # Format as the required dictionary
                    return res_text, {"total": toks, "calls": 1}

                query_result = tracker.run_stage("generation", run_llm)
                if isinstance(query_result, tuple):
                    query_result = query_result[0]

                # --- Verdict Parsing ---
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

                logger.info(f"Prompt Stuffing Verdict: {predicted_label}")

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                # --- Log to DB ---
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "best_evidence": best_evidence,
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                logger.info(
                    "Sleeping for 10 seconds to respect DuckDuckGo rate limits..."
                )
                time.sleep(10)

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("PROMPT STUFFING (OPEN WEB) COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_prompt_stuffing_baseline()
