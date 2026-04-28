import os
import json
import time
import uuid
import dotenv
from groq import Groq

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from WebScraper.scraper import Scraper
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
    print(f"Starting Baseline Prompt Stuffing with {MAX_CLAIMS_TO_TEST} claims...")

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

                print(f"\n[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")

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
                    return scraper.search_and_extract(
                        claim_text, num_results=10, use_llm_filter=False
                    )

                raw_scraped_sources = tracker.run_stage("retrieval", run_retrieval)
                print(f"  -> Scraped {len(raw_scraped_sources)} pages from the web.")

                # --- 2. Concatenation ---
                massive_evidence = ""

                # Limit to 2000 chars per page so Groq doesn't crash from Token Limits
                CHARACTER_LIMIT_PER_PAGE = 2000

                for src in raw_scraped_sources:
                    url = src.get("url", "Unknown URL")
                    # Slice the body text to prevent rate limit
                    body = src.get("body", "")[:CHARACTER_LIMIT_PER_PAGE]
                    massive_evidence += f"\n--- Source: {url} ---\n{body}\n"

                if not massive_evidence.strip():
                    massive_evidence = "No relevant articles could be scraped."

                # --- 3. Generation ---
                def run_llm():
                    res_text, toks = get_prompt_stuffing_verdict(
                        claim_text, massive_evidence
                    )
                    return res_text, {"total": toks, "calls": 1}

                query_result = tracker.run_stage("generation", run_llm)

                # Safety check to clear the tuple bug!
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

                print(f"  -> Prompt Stuffing Verdict: {predicted_label}")

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                # --- Log to DB ---
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "raw_sources": raw_scraped_sources,
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                print("Sleeping for 10 seconds to respect DuckDuckGo rate limits...")
                time.sleep(10)

    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "=" * 40)
    print("PROMPT STUFFING (OPEN WEB) COMPLETE!")
    print("=" * 40)


if __name__ == "__main__":
    run_prompt_stuffing_baseline()
