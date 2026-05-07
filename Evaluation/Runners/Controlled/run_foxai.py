import os
import json
import sqlite3
import time
import uuid
import dotenv
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Evaluation.Utils.dataset_manager import DatasetManager
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from GraphRAG.rag_pipeline import RAG_Pipeline
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

MAX_CLAIMS_TO_TEST = 5
USE_METADATA = bool(os.getenv("AVERITEC_USE_METADATA"))
logger = Logger("Fox-AI-Controlled").get_logger()


def extract_perfect_evidence(evidence_data, wiki_cursor):
    """
    Parses the FEVER evidence JSON, queries the SQLite DB, and extracts the exact text.
    """
    extracted_text = ""

    for evidence_set in evidence_data:
        for ev in evidence_set:
            page_id = ev[2]
            sentence_num = str(ev[3])

            if page_id is None:
                continue

            wiki_cursor.execute(
                "SELECT lines FROM wiki_articles WHERE page_id = ?", (page_id,)
            )
            result = wiki_cursor.fetchone()

            if result:
                raw_lines = result[0]
                sentences = raw_lines.split("\n")
                for sentence in sentences:
                    parts = sentence.split("\t")
                    if parts[0] == sentence_num and len(parts) > 1:
                        extracted_text += parts[1] + " "
                        break

    return extracted_text.strip()


def run_controlled_experiment():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("Controlled")
    prompt_instructions = dataset_manager.get_prompt_instructions()

    logger.info(
        f"Starting Controlled Experiment (FoxAI) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")

    # --- 1. Load the correct Database/Retriever based on environment ---
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

    # Initialize the heavy pipeline components once
    preprocessor = Preprocessing_Pipeline()
    rag = RAG_Pipeline()

    successful_runs = 0

    try:
        # Ask the DatasetManager for the claims
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            # --- CONDITIONAL METADATA QUERY ---
            # If the flag is True, build the enriched query. Otherwise, just use the claim.
            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            # Use dynamic labels for missing info depending on the dataset
            nei_label = (
                "NOT ENOUGH INFO"
                if active_dataset == "FEVER"
                else "Not Enough Evidence"
            )

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")
            logger.info(f"Ground Truth: {ground_truth}")

            # --- 3. Extract Evidence Dynamically ---
            perfect_evidence = ""
            if active_dataset == "FEVER":
                evidence_data = data.get("evidence", [])
                logger.info(f"Raw Evidence Array from FEVER: {evidence_data}")
                perfect_evidence = extract_perfect_evidence(evidence_data, wiki_cursor)

            elif active_dataset == "AVERITEC" and averitec_retriever is not None:
                claim_id_internal = data.get("internal_id")
                all_sentences = averitec_retriever.get_evidence_for_claim(
                    claim_id_internal
                )

                if all_sentences:
                    # --- THE FOXAI LIFESAVER + CONDITIONAL METADATA FILTER ---
                    from rank_bm25 import BM25Okapi

                    # Tokenize corpus and the search_query (which may or may not include metadata)
                    tokenized_corpus = [s.lower().split() for s in all_sentences]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = search_query.lower().split()

                    # Grab only the 30 most keyword-relevant sentences for GraphRAG
                    top_sentences = bm25.get_top_n(tokenized_query, all_sentences, n=30)
                    perfect_evidence = " ".join(top_sentences)

            if not perfect_evidence and ground_truth != nei_label:
                logger.warning(
                    "Could not find evidence in knowledge base for this claim."
                )

            logger.info(f"Perfect Evidence Retrieved: {perfect_evidence[:100]}...")

            # --- THE SHORT-CIRCUIT ---
            if not perfect_evidence:
                logger.info(f"No evidence available. Short-circuiting to {nei_label}.")
                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id,
                    ground_truth=ground_truth,
                    system_type="FoxAI-GraphRAG",
                    dataset_setting=tracker_env_name,
                )

                Claim(
                    text=claim_text,
                    title="[NEI] " + claim_text[:30] + "...",
                    summary="Short-circuited due to lack of evidence.",
                    claim_id=claim_id,
                )

                predicted_label = nei_label
                query_result = f"VERDICT: {nei_label}\nREASONING: No evidence provided by the dataset."

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "raw_sources": [],
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                continue

            # --- NORMAL PIPELINE ---
            claim_id = str(uuid.uuid4())
            tracker = ExperimentTracker(
                claim_id=claim_id,
                ground_truth=ground_truth,
                system_type="FoxAI-GraphRAG",
                dataset_setting=tracker_env_name,
            )

            # --- 1. Preprocessing (Pass the original claim to keep LLM focused) ---
            def run_prep():
                return preprocessor.run_claim_pipe(claim_text)

            claim_title, claim_summary = tracker.run_stage("preprocessor", run_prep)
            claim = Claim(claim_text, claim_title, claim_summary, claim_id=claim_id)

            # --- 2. Retrieval (BYPASSED SCRAPER) ---
            def run_retrieval():
                mock_srcs = [
                    {
                        "title": f"{active_dataset} Perfect Evidence",
                        "url": "Local_DB",
                        "site": "Dataset",
                        "body": perfect_evidence,
                    }
                ]
                prep_srcs, prep_tokens = preprocessor.run_sources_pipe(mock_srcs)
                return (prep_srcs, mock_srcs), prep_tokens

            preprocessed_sources, sources = tracker.run_stage(
                "retrieval", run_retrieval
            )
            claim.add_sources(preprocessed_sources)

            # =========================================================
            # 🛡️ SAFETY CHECK: Did Groq find any entities?
            # =========================================================
            has_entities = False
            for src in preprocessed_sources:
                if (
                    src.get("entities")
                    and src.get("entities") != "[]"
                    and len(src.get("entities")) > 0
                ):
                    has_entities = True
                    break

            if not has_entities:
                logger.info(
                    "NER extracted 0 entities. Short-circuiting to prevent Neo4j/Ollama crash."
                )
                predicted_label = "Error: No Entities"
                query_result = f"VERDICT: {nei_label}\nREASONING: Evidence was provided, but the NER model failed to extract any entities to build a graph."

                Answer(claim_id=claim.id, answer=query_result, graphs_folder=None)
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "raw_sources": sources,
                        "query_result": query_result,
                    },
                )
                successful_runs += 1
                continue
            # =========================================================

            # --- 3. GraphRAG ---
            def run_rag():
                q_res, g_folder, t_usage = rag.run_pipeline(
                    preprocessed_sources, claim.text, claim.id, prompt_instructions
                )
                return (q_res, g_folder), t_usage

            query_result, graphs_folder = tracker.run_stage("generation", run_rag)

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

            logger.info(f"FoxAI Verdict: {predicted_label}")

            # --- Create Answer Entity for the UI ---
            Answer(claim_id=claim.id, answer=query_result, graphs_folder=graphs_folder)

            # --- 5. Log Experiment Metrics ---
            tracker.finalize(
                predicted_label,
                {
                    "claim_text": claim_text,
                    "raw_sources": sources,
                    "query_result": query_result,
                },
            )

            successful_runs += 1

            logger.info("Sleeping for 5 seconds before next claim...")
            time.sleep(5)

    except FileNotFoundError as e:
        logger.error(f"Could not find dataset files. {e}")
    finally:
        if wiki_conn:
            wiki_conn.close()

    logger.info("=" * 20)
    logger.info("CONTROLLED EXPERIMENT COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_controlled_experiment()
