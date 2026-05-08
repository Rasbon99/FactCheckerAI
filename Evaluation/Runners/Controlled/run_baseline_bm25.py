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
from Evaluation.Utils.dataset_manager import DatasetManager
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer
from rank_bm25 import BM25Okapi

dotenv.load_dotenv("key.env", override=True)

# Configuration
MAX_CLAIMS_TO_TEST = 5

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger("BM25-Baseline").get_logger()


def clean_query_for_fts(text):
    """Removes punctuation that breaks SQLite FTS syntax."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()


def get_bm25_verdict(claim_text, retrieved_evidence, prompt_instructions, nei_label):
    """Asks the LLM to verify the claim using the BM25 retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer exactly: {nei_label}.

    {prompt_instructions}

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
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("Controlled")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline 2 (BM25 Keyword Search) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")

    # --- Load Databases based on Environment ---
    wiki_conn = None
    wiki_cursor = None
    averitec_retriever = None

    if active_dataset == "FEVER":
        wiki_db_path = os.getenv(
            "FEVER_WIKIPEDIA_DB_PATH", "Datasets/FEVER/fever_wiki.db"
        )
        wiki_conn = sqlite3.connect(wiki_db_path)
        wiki_cursor = wiki_conn.cursor()
    elif active_dataset == "AVERITEC":
        averitec_retriever = AVeriTeCKnowledgeRetriever()

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            # --- CONDITIONAL METADATA QUERY ---
            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")

            claim_id = str(uuid.uuid4())
            tracker = ExperimentTracker(
                claim_id=claim_id,
                ground_truth=ground_truth,
                system_type="Baseline-BM25",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[BM25] " + claim_text[:30] + "...",
                summary="Tested with Lexical BM25 Retrieval.",
                claim_id=claim_id,
            )

            # --- THE RETRIEVAL STEP (BM25) ---
            def run_retrieval():
                formatted_results = []

                if active_dataset == "FEVER":
                    # Preprocess and search SQLite FTS5 (BM25)
                    if wiki_cursor is None:
                        logger.error("wiki_cursor is None: cannot query FEVER database")
                    else:
                        clean_claim = clean_query_for_fts(search_query)
                        wiki_cursor.execute(
                            """
                            SELECT page_id, lines FROM wiki_fts 
                            WHERE wiki_fts MATCH ? 
                            ORDER BY rank 
                            LIMIT 2
                            """,
                            (clean_claim,),
                        )

                        for row in wiki_cursor.fetchall():
                            # Truncate lines to avoid context window blowup
                            formatted_results.append(
                                {"title": row[0], "snippet": row[1][:1500]}
                            )

                elif active_dataset == "AVERITEC":
                    # Load all sentences instantly and apply BM25 math
                    claim_id_internal = data.get("internal_id")
                    if averitec_retriever is None:
                        logger.error(
                            "averitec_retriever is None: cannot query AVERITEC database"
                        )
                        sentences = []
                    else:
                        sentences = averitec_retriever.get_evidence_for_claim(
                            claim_id_internal
                        )

                        # INJECT NOISE (If running the Noisy robustness test)
                        if "noisy_ids" in data and sentences is not None:
                            for n_id in data["noisy_ids"]:
                                noisy_sentences = (
                                    averitec_retriever.get_evidence_for_claim(n_id)
                                )
                                if noisy_sentences:
                                    sentences.extend(noisy_sentences)

                    if sentences:
                        tokenized_corpus = [s.lower().split() for s in sentences]
                        bm25 = BM25Okapi(tokenized_corpus)

                        # Use the enriched query for retrieval
                        tokenized_query = search_query.lower().split()

                        # Grab exactly Top 2 to match FEVER methodology
                        top_sentences = bm25.get_top_n(tokenized_query, sentences, n=2)

                        for i, sentence in enumerate(top_sentences):
                            formatted_results.append(
                                {
                                    "title": f"AVeriTeC Store ID: {claim_id_internal} (Rank {i+1})",
                                    "snippet": sentence,
                                }
                            )

                return formatted_results

            results = tracker.run_stage("retrieval", run_retrieval)

            combined_evidence = ""
            raw_sources = []

            # Unify the output formatting for the LLM
            for res in results:
                combined_evidence += (
                    f"\n--- Source: {res['title']} ---\n{res['snippet']}...\n"
                )
                raw_sources.append(res)

            if not combined_evidence:
                combined_evidence = (
                    f"No relevant evidence found in the {active_dataset} database."
                )

            # --- THE GENERATION STEP ---
            def run_llm():
                # Strictly pass the original claim_text to the LLM Verifier
                res_text, toks = get_bm25_verdict(
                    claim_text, combined_evidence, prompt_instructions, nei_label
                )
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
        if wiki_conn:
            wiki_conn.close()

    logger.info("=" * 20)
    logger.info("BM25 BASELINE COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_bm25_baseline()
