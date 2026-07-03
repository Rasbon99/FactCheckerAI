import os
import time
import uuid
import dotenv
from llamacpp_client import ChatLlamaCppServer, load_models, set_alias_map
from langchain_core.messages import HumanMessage
from log import Logger

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Evaluation.Utils.dataset_manager import DatasetManager
from WebScraper.scraper import Scraper
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")
model_port = int(os.getenv("LLM_MODEL_PORT", "8080"))

print(f"[Backend] Connecting to local llama.cpp server on port {model_port}...")
set_alias_map({model_alias: model_port})
load_models([model_alias])

# Configuration
MAX_CLAIMS_TO_TEST = 5
logger = Logger("PromptStuffing-OpenWeb").get_logger()

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"


def get_prompt_stuffing_verdict(
    claim_text, massive_evidence_string, prompt_instructions, nei_label
):
    """Asks the LLM to verify the claim using the massive wall of scraped text."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer exactly: {nei_label}.

    {prompt_instructions}

    EVIDENCE:
    {massive_evidence_string}

    CLAIM: {claim_text}
    """

    client = ChatLlamaCppServer(
        model=model_alias,
        temperature=0.0,
        max_tokens=200,
    )

    messages = [HumanMessage(content=prompt)]
    response = client.invoke(messages)

    result_text = response.content
    tokens_used = (
        response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        if hasattr(response, "response_metadata")
        else 0
    )

    return result_text, tokens_used


def run_prompt_stuffing_baseline_openweb():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("open-web")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline Prompt Stuffing (Open Web) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            # 1. Instantiate Scraper inside the loop to avoid DuckDuckGo session bans!
            scraper = Scraper()

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
                system_type="Baseline-PromptStuffing",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[PromptStuff] " + claim_text[:30] + "...",
                summary="Tested by stuffing all scraped web pages directly into the prompt.",
                claim_id=claim_id,
            )

            # --- 1. Retrieval (Scraper) ---
            def run_retrieval():
                # We pass the ENRICHED search query to DuckDuckGo
                raw_scraped_sources, scraper_metrics = scraper.search_and_extract(
                    search_query, num_results=10
                )

                combined_evidence = ""
                for src in raw_scraped_sources:
                    combined_evidence += (
                        f"\n--- Source: {src.get('url')} ---\n{src.get('body', '')}\n"
                    )

                if not combined_evidence.strip():
                    return "No relevant articles could be scraped.", scraper_metrics

                # --- API SAFETY VALVE FOR PROMPT STUFFING ---
                # Local LLM context limit is typically 8k tokens (~32,000 characters).
                # We cap it safely at 25,000 characters so the prompt never crashes.
                MAX_CHARS = 25000
                if len(combined_evidence) > MAX_CHARS:
                    logger.warning(
                        f"Evidence massive ({len(combined_evidence)} chars). Truncating to {MAX_CHARS} to survive context limits."
                    )
                    combined_evidence = (
                        combined_evidence[:MAX_CHARS]
                        + "\n...[EVIDENCE TRUNCATED DUE TO CONTEXT LIMITS]..."
                    )

                return combined_evidence, scraper_metrics

            best_evidence = tracker.run_stage("retrieval", run_retrieval)

            # Safely clear tuple bug
            if isinstance(best_evidence, tuple):
                best_evidence = best_evidence[0]

            # --- 2. Generation (The LLM Call) ---
            def run_llm():
                # Strictly pass the unedited claim_text to the generator
                res_text, toks = get_prompt_stuffing_verdict(
                    claim_text, best_evidence, prompt_instructions, nei_label
                )
                return res_text, {"total": toks, "calls": 1}

            query_result = tracker.run_stage("generation", run_llm)

            if isinstance(query_result, tuple):
                query_result = query_result[0]

            # --- 3. Verdict Parsing ---
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

            # --- 4. Log to DB ---
            tracker.finalize(
                predicted_label,
                {
                    "claim_text": claim_text,
                    "best_evidence": best_evidence,
                    "query_result": query_result,
                },
            )

            successful_runs += 1
            logger.info("Sleeping for 15 seconds to respect DuckDuckGo rate limits...")
            time.sleep(15)

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("PROMPT STUFFING (OPEN WEB) COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_prompt_stuffing_baseline_openweb()
