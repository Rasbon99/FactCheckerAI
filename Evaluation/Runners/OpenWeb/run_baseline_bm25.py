import os
import json
import time
import uuid
import dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
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
logger = Logger("run_baseline_bm25").get_logger()

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


def simple_chunker(text, chunk_word_size=150):
    """Breaks massive web pages into smaller paragraph-sized chunks for BM25 to analyze."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_word_size):
        chunk = " ".join(words[i : i + chunk_word_size])
        chunks.append(chunk)
    return chunks


def get_bm25_verdict(claim_text, best_evidence_string):
    """Asks the LLM to verify the claim using ONLY the top chunks found by BM25."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer NOT ENOUGH INFO.

    You must format your response EXACTLY like this:
    VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
    REASONING: [Your brief explanation citing the provided evidence]

    EVIDENCE:
    {best_evidence_string}

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


def run_bm25_baseline():
    logger.info(
        f"Starting Baseline BM25 Keyword RAG with {MAX_CLAIMS_TO_TEST} claims..."
    )

    successful_runs = 0

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                # 1. Instantiate Scraper inside the loop to avoid DuckDuckGo session bans!
                scraper = Scraper()

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
                    dataset_setting="FEVER-OpenWeb",
                )

                Claim(
                    text=claim_text,
                    title="[BM25] " + claim_text[:30] + "...",
                    summary="Tested by scoring scraped pages using the BM25 algorithm.",
                    claim_id=claim_id,
                )

                # --- 2. Retrieval & Filtering (Scraper + BM25) ---
                def run_retrieval():
                    raw_scraped_sources, scraper_metrics = scraper.search_and_extract(
                        claim_text, num_results=10
                    )

                    logger.info(
                        f"Scraped {len(raw_scraped_sources)} pages. Running BM25 math..."
                    )
                    # Combine all text from all scraped pages
                    all_text = ""
                    for src in raw_scraped_sources:
                        all_text += src.get("body", "") + " "

                    # Break it into chunks
                    chunks = simple_chunker(all_text)

                    if not chunks:
                        return "No relevant articles could be scraped.", scraper_metrics

                    # --- THE BM25 ALGORITHM ---
                    # Tokenize the chunks and the claim (lowercase, split by spaces)
                    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = claim_text.lower().split()

                    # Get the Top 3 most mathematically relevant chunks
                    top_3_chunks = bm25.get_top_n(tokenized_query, chunks, n=3)
                    best_evidence = "\n--- BM25 TOP MATCH ---\n".join(top_3_chunks)

                    return best_evidence, scraper_metrics

                best_evidence = tracker.run_stage("retrieval", run_retrieval)

                if isinstance(best_evidence, tuple):
                    best_evidence = best_evidence[0]

                # --- 3. Generation (The LLM Call) ---
                def run_llm():
                    res_text, toks = get_bm25_verdict(claim_text, best_evidence)
                    return res_text, {"total": toks, "calls": 1}

                query_result = tracker.run_stage("generation", run_llm)

                if isinstance(query_result, tuple):
                    query_result = query_result[0]

                # --- 4. Verdict Parsing ---
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

                # --- 5. Log to DB ---
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "bm25_evidence": best_evidence,
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                logger.info(
                    "Sleeping for 15 seconds to respect DuckDuckGo rate limits..."
                )
                time.sleep(15)

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("BM25 (OPEN WEB) COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_bm25_baseline()
