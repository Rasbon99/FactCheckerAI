import os
import sqlite3
import time
import uuid
import re
import dotenv
from groq import Groq
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer, Experiment
from rank_bm25 import BM25Okapi

dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)
logger = Logger("SparseRAG-Controlled").get_logger()


def clean_query_for_fts(text):
    """Removes punctuation that breaks SQLite FTS syntax."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()


def get_bm25_verdict(claim_text, retrieved_evidence, prompt_instructions):
    """Asks the LLM to verify the claim using the BM25 retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 

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


def run_sparse_baseline():
    dataset_manager = DatasetManager()
    metadata = dataset_manager.get_experiment_metadata(environment="controlled")
    active_dataset = metadata["dataset_name"]
    use_meta = metadata["use_metadata"]

    prompt_instructions = dataset_manager.get_prompt_instructions()

    logger.info(f"Starting Baseline (Sparse/BM25) with {MAX_CLAIMS_TO_TEST} claims...")
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Metadata Super Query: {use_meta}")

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

            search_query = claim_text
            if active_dataset == "AVERITEC" and use_meta:
                search_query = dataset_manager.build_search_query(data)

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[Sparse] " + claim_text[:30] + "...",
                summary="Tested with Lexical BM25 Retrieval.",
                claim_id=claim_id,
            )

            # --- THE RETRIEVAL STEP (BM25) ---
            t0 = time.time()
            formatted_results = []

            if active_dataset == "FEVER":
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
                        formatted_results.append(
                            {"title": row[0], "snippet": row[1][:1500]}
                        )

            elif active_dataset == "AVERITEC":
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

                    if "noisy_ids" in data and sentences is not None:
                        for n_id in data["noisy_ids"]:
                            noisy_sentences = averitec_retriever.get_evidence_for_claim(
                                n_id
                            )
                            if noisy_sentences:
                                sentences.extend(noisy_sentences)

                if sentences:
                    tokenized_corpus = [s.lower().split() for s in sentences]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = search_query.lower().split()
                    top_sentences = bm25.get_top_n(tokenized_query, sentences, n=2)

                    for i, sentence in enumerate(top_sentences):
                        formatted_results.append(
                            {
                                "title": f"AVeriTeC Store ID: {claim_id_internal} (Rank {i+1})",
                                "snippet": sentence,
                            }
                        )

            latency_retrieval = time.time() - t0

            combined_evidence = ""
            raw_sources = []

            for res in formatted_results:
                combined_evidence += (
                    f"\n--- Source: {res['title']} ---\n{res['snippet']}...\n"
                )
                raw_sources.append(res)

            if not combined_evidence:
                combined_evidence = (
                    f"No relevant evidence found in the {active_dataset} database."
                )

            # --- THE GENERATION STEP ---
            t0 = time.time()
            query_result, toks = get_bm25_verdict(
                claim_text, combined_evidence, prompt_instructions
            )
            latency_generation = time.time() - t0

            try:
                if not query_result or not query_result.strip():
                    predicted_label = "Error: Empty LLM Response"
                    query_result = "The LLM failed to generate a response."
                elif "VERDICT:" in query_result:
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

            # --- LOG TO EXPERIMENTS DATABASE ---
            Experiment(
                claim_id=claim_id,
                predicted_label=predicted_label,
                ground_truth=ground_truth,
                latencies={
                    "preprocessor": 0.0,
                    "retrieval": latency_retrieval,
                    "generation": latency_generation,
                },
                tokens={"preprocessor": 0, "retrieval": 0, "generation": toks},
                calls={"preprocessor": 0, "retrieval": 0, "generation": 1},
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": raw_sources,
                    "query_result": query_result,
                },
                system_type="SparseRAG",
                environment=metadata["environment"],
                dataset_name=metadata["dataset_name"],
                experiment_type=metadata["experiment_type"],
                use_metadata=use_meta,
            )

            successful_runs += 1
            time.sleep(2)

    except Exception as e:
        logger.error(f"{e}")
    finally:
        if wiki_conn:
            wiki_conn.close()

    logger.info("=" * 20)
    logger.info("SPARSE BASELINE COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_sparse_baseline()
