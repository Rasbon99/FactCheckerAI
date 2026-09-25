import os
import sqlite3
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Utils.prompt_manager import get_dataset_prompt_instructions
from Evaluation.Utils.averitec_retriever import AVeriTeCKnowledgeRetriever
from Database.data_entities import Claim, Answer, Experiment

# Load environment variables
dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5
logger = Logger("PromptStuffing-Controlled").get_logger()

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


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


def get_prompt_stuffing_verdict(
    claim_text, massive_evidence_string, prompt_instructions
):
    """Asks the LLM to verify the claim using the massive wall of retrieved text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 

    {prompt_instructions}

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


def run_prompt_stuffing_baseline_controlled():
    dataset_manager = DatasetManager()
    metadata = dataset_manager.get_experiment_metadata(environment="controlled")
    active_dataset = metadata["dataset_name"]
    use_meta = metadata["use_metadata"]

    prompt_instructions = get_dataset_prompt_instructions(active_dataset)

    logger.info(
        f"Starting Baseline (Prompt Stuffing - Controlled) with {MAX_CLAIMS_TO_TEST} claims..."
    )
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
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[PromptStuff] " + claim_text[:30] + "...",
                summary="Tested by stuffing all database evidence directly into the prompt.",
                claim_id=claim_id,
            )

            # --- 1. Retrieval (Database Extract) ---
            t0 = time.time()
            combined_evidence = ""

            if active_dataset == "FEVER":
                evidence_data = data.get("evidence", [])
                combined_evidence = extract_perfect_evidence(evidence_data, wiki_cursor)

            elif active_dataset == "AVERITEC" and averitec_retriever is not None:
                claim_id_internal = data.get("internal_id")
                all_sentences = averitec_retriever.get_evidence_for_claim(
                    claim_id_internal
                )

                if "noisy_ids" in data and all_sentences is not None:
                    for n_id in data["noisy_ids"]:
                        noisy_sentences = averitec_retriever.get_evidence_for_claim(
                            n_id
                        )
                        if noisy_sentences:
                            all_sentences.extend(noisy_sentences)

                if all_sentences:
                    # In prompt stuffing, we just dump EVERYTHING into the prompt
                    combined_evidence = "\n".join(all_sentences)

            if not combined_evidence.strip():
                best_evidence = "No relevant evidence could be found in the dataset."
            else:
                # --- API SAFETY VALVE FOR PROMPT STUFFING ---
                MAX_CHARS = 20000
                if len(combined_evidence) > MAX_CHARS:
                    logger.warning(
                        f"Evidence massive ({len(combined_evidence)} chars). Truncating to {MAX_CHARS} to survive API limits."
                    )
                    best_evidence = (
                        combined_evidence[:MAX_CHARS]
                        + "\n...[EVIDENCE TRUNCATED DUE TO CONTEXT LIMITS]..."
                    )
                else:
                    best_evidence = combined_evidence

            latency_retrieval = time.time() - t0

            # --- 2. Generation (The LLM Call) ---
            t0 = time.time()
            query_result, tokens_used = get_prompt_stuffing_verdict(
                claim_text, best_evidence, prompt_instructions
            )
            latency_generation = time.time() - t0

            # --- 3. Verdict Parsing ---
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

            logger.info(f"Prompt Stuffing Verdict: {predicted_label}")

            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # --- 4. Log to DB ---
            Experiment(
                claim_id=claim_id,
                predicted_label=predicted_label,
                ground_truth=ground_truth,
                latencies={
                    "preprocessor": 0.0,
                    "retrieval": latency_retrieval,
                    "generation": latency_generation,
                },
                tokens={
                    "preprocessor": 0,
                    "retrieval": 0,
                    "generation": tokens_used,
                },
                calls={
                    "preprocessor": 0,
                    "retrieval": 0,
                    "generation": 1,
                },
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": [],
                    "best_evidence": best_evidence,
                    "query_result": query_result,
                },
                system_type="PromptStuffing",
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
    logger.info("PROMPT STUFFING (CONTROLLED) COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_prompt_stuffing_baseline_controlled()
