import os
import json
import sqlite3
import time
import uuid
import re
import dotenv
from groq import Groq
from log import Logger

from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Database.data_entities import Claim, Answer

dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
WIKI_DB_PATH = os.getenv("FEVER_WIKIPEDIA_DB_PATH", "Datasets/fever_wiki.db")
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger(__name__).get_logger()


def clean_query_for_fts(text):
    """Removes punctuation that breaks SQLite FTS syntax."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()


def get_bm25_verdict(claim_text, retrieved_evidence):
    """Asks the LLM to verify the claim using the BM25 retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer NOT ENOUGH INFO.

    You must format your response EXACTLY like this:
    VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
    REASONING: [Your brief explanation citing the provided evidence]

    EVIDENCE:
    {retrieved_evidence}

    CLAIM: {claim_text}
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=200,
    )

    return (
        response.choices[0].message.content,
        response.usage.total_tokens if response.usage else 0,
    )


def run_bm25_baseline():
    logger.info(
        f"Starting Baseline 2 (BM25 Keyword Search) with {MAX_CLAIMS_TO_TEST} claims..."
    )

    # Connect to the Wiki DB to use the FTS5 engine
    wiki_conn = sqlite3.connect(WIKI_DB_PATH)
    wiki_cursor = wiki_conn.cursor()

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
                    system_type="Baseline-BM25",
                    dataset_setting="FEVER-Controlled",
                )

                Claim(
                    text=claim_text,
                    title="[BM25] " + claim_text[:30] + "...",
                    summary="Tested with SQLite BM25 Retrieval.",
                    claim_id=claim_id,
                )

                # --- 1. THE PREPROCESSING STEP ---
                def run_prep():
                    return clean_query_for_fts(claim_text)

                clean_claim = tracker.run_stage("preprocessor", run_prep)

                # --- 2. THE RETRIEVAL STEP (BM25) ---
                # Fetch Top 2 articles using FTS5 MATCH, ordered by BM25 rank
                def run_retrieval():
                    wiki_cursor.execute(
                        """
                        SELECT page_id, lines FROM wiki_fts 
                        WHERE wiki_fts MATCH ? 
                        ORDER BY rank 
                        LIMIT 2
                    """,
                        (clean_claim,),
                    )
                    return wiki_cursor.fetchall()

                results = tracker.run_stage("retrieval", run_retrieval)

                combined_evidence = ""
                raw_sources = []

                for idx, row in enumerate(results):
                    page_id = row[0]
                    # Truncate lines to avoid blowing up Groq's context window
                    text_snippet = row[1][:1500]
                    combined_evidence += (
                        f"\n--- Article: {page_id} ---\n{text_snippet}...\n"
                    )
                    raw_sources.append({"title": page_id, "snippet": text_snippet})

                if not combined_evidence:
                    combined_evidence = "No relevant Wikipedia articles found."

                # --- THE GENERATION STEP ---
                def run_llm():
                    res_text, toks = get_bm25_verdict(claim_text, combined_evidence)
                    return (res_text, None), {"total": toks, "calls": 1}

                query_result = tracker.run_stage("generation", run_llm)

                if isinstance(query_result, tuple):
                    query_result = query_result[0]

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

                logger.info(f"BM25 Verdict: {predicted_label}")

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                # Log to DB
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "raw_sources": raw_sources,
                        "query_result": query_result,
                    },
                )
                successful_runs += 1
                time.sleep(2)

    except Exception as e:
        logger.error(f"{e}")
    finally:
        wiki_conn.close()

    logger.info("=" * 40)
    logger.info("BM25 BASELINE COMPLETE!")
    logger.info("=" * 40)


if __name__ == "__main__":
    run_bm25_baseline()
