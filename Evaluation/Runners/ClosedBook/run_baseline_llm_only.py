import os
import time
import uuid
import dotenv
from llamacpp_client import ChatLlamaCppServer, load_models, set_alias_map
from langchain_core.messages import HumanMessage
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager
from Database.data_entities import Claim, Answer, Experiment

dotenv.load_dotenv("key.env", override=False)

model_alias = os.getenv("LLM_MODEL_ALIAS", "meta-llama-3")
model_port = int(os.getenv("LLM_MODEL_PORT", "8080"))

print(f"[Backend] Connecting to local llama.cpp server on port {model_port}...")
set_alias_map({model_alias: model_port})
load_models([model_alias])

# Configuration
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"
logger = Logger("ClosedBook-Baseline").get_logger()


def get_closed_book_verdict(claim_text, prompt_instructions, metadata_context=""):
    """Asks the LLM to verify the claim using ONLY its internal weights, providing context if available."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY your internal knowledge. 

    {prompt_instructions}

    CLAIM: {claim_text}
    {metadata_context}
    """

    client = ChatLlamaCppServer(
        model=model_alias,
        temperature=0.0,  # Zero temperature for maximum factual consistency
        max_tokens=200,
    )

    messages = [HumanMessage(content=prompt)]
    response = client.invoke(messages)

    content = response.content
    if isinstance(content, str):
        result_text = content
    else:
        result_text = "\n".join(
            item if isinstance(item, str) else str(item) for item in content
        )
    tokens_used = (
        response.response_metadata.get("token_usage", {}).get("total_tokens", 0)
        if hasattr(response, "response_metadata")
        else 0
    )

    return result_text, tokens_used


def run_closed_book_baseline():
    dataset_manager = DatasetManager()

    metadata = dataset_manager.get_experiment_metadata(environment="closed_book")
    active_dataset = metadata["dataset_name"]

    # Tweak the instructions slightly since this baseline has no "provided evidence"
    base_instructions = dataset_manager.get_prompt_instructions()
    prompt_instructions = base_instructions.replace(
        "citing the provided evidence", "based on your internal knowledge"
    )

    logger.info("Starting Baseline (Closed-Book / LLM-Only)...")
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
    logger.info(f"Using Metadata Context: {USE_METADATA}")
    logger.info(f"Using Model Alias: {model_alias}")

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data()

        for line_number, data in enumerate(claims_data):
            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            # --- OPTIONAL METADATA INJECTION ---
            metadata_context = ""
            if active_dataset == "AVERITEC" and USE_METADATA:
                speaker = data.get("speaker", "")
                date = data.get("claim_date", "")
                location_ISO_code = data.get("location_ISO_code", "")
                reporting_source = data.get("reporting_source", "")

                meta_parts = []
                if speaker:
                    meta_parts.append(f"- Speaker: {speaker}")
                if location_ISO_code:
                    meta_parts.append(f"- Location: {location_ISO_code}")
                if date:
                    meta_parts.append(f"- Date: {date}")
                if reporting_source:
                    meta_parts.append(f"- Source: {reporting_source}")

                if meta_parts:
                    metadata_context = (
                        "\nCONTEXT PROVIDED FOR THIS CLAIM:\n" + "\n".join(meta_parts)
                    )

            logger.info(f"[{line_number + 1}] Claim: {claim_text}")
            logger.info(f"Ground Truth: {ground_truth}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[ClosedBook] " + claim_text[:30] + "...",
                summary="Tested without any external evidence.",
                claim_id=claim_id,
            )

            # --- GENERATION STEP ---
            t0 = time.time()
            query_result, tokens_used = get_closed_book_verdict(
                claim_text, prompt_instructions, metadata_context
            )
            latency_generation = time.time() - t0

            # Verdict Parsing
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

            logger.info(f"Closed-Book Verdict: {predicted_label}")

            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # --- LOG TO EXPERIMENTS DATABASE ---
            Experiment(
                claim_id=claim_id,
                predicted_label=predicted_label,
                ground_truth=ground_truth,
                latencies={
                    "preprocessor": 0.0,
                    "retrieval": 0.0,
                    "generation": latency_generation,
                },
                tokens={"preprocessor": 0, "retrieval": 0, "generation": tokens_used},
                calls={"preprocessor": 0, "retrieval": 0, "generation": 1},
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": [],
                    "query_result": query_result,
                },
                system_type="ClosedBook",
                environment=metadata["environment"],
                dataset_name=metadata["dataset_name"],
                experiment_type=metadata["experiment_type"],
                use_metadata=metadata["use_metadata"],
            )

            successful_runs += 1

            # Sleep for 2 seconds to avoid overwhelming the local server
            time.sleep(2)

    except Exception as e:
        logger.error(f"Error during execution: {e}")

    logger.info("=" * 20)
    logger.info("CLOSED-BOOK BASELINE COMPLETE!")
    logger.info(f"Successfully processed: {successful_runs}")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_closed_book_baseline()
