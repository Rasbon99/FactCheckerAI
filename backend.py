import os
import shutil
import uuid
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline
from log import Logger

backend_app = FastAPI()
db = Database()
logger = Logger("Backend").get_logger()


@backend_app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled backend error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": {"type": type(exc).__name__, "message": str(exc)}},
    )


class InputText(BaseModel):
    text: str
    search_query: Optional[str] = None
    response_format_instructions: Optional[str] = None
    preserve_failed_claim: bool = False


@backend_app.post("/run_pipeline")
def process_text(input_text: InputText):
    text = input_text.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Claim text cannot be empty.")

    claim_id = str(uuid.uuid4())

    preprocessor = Preprocessing_Pipeline()
    scraper = Scraper()
    rag = RAG_Pipeline()

    latencies = {}
    tokens = {}
    calls = {}
    current_stage = "setup"

    # --- 1. Preprocessing ---
    current_stage = "preprocessing"

    t0 = time.time()

    claim_title, prep_claim_metrics = preprocessor.run_claim_pipe(text)

    if not claim_title:
        raise RuntimeError("Claim preprocessing returned an empty title.")

    latencies["preprocessor"] = time.time() - t0

    tokens["preprocessor"] = prep_claim_metrics.get("total", 0)
    calls["preprocessor"] = prep_claim_metrics.get("calls", 0)

    claim = None

    try:
        claim = Claim(text, claim_title, claim_id=claim_id)

        target_search_query = (
            input_text.search_query if input_text.search_query else claim_title
        )

        # --- 2. Retrieval (Scraper + Preprocessor) ---
        current_stage = "retrieval"

        t0 = time.time()
        sources, scraper_metrics = scraper.search_and_extract(
            target_search_query, num_results=10
        )
        preprocessed_sources, prep_metrics = preprocessor.run_sources_pipe(sources)

        claim.add_sources(preprocessed_sources)

        latencies["retrieval"] = time.time() - t0

        tokens["retrieval"] = scraper_metrics.get("total", 0) + prep_metrics.get(
            "total", 0
        )
        calls["retrieval"] = scraper_metrics.get("calls", 0) + prep_metrics.get(
            "calls", 0
        )

        # --- 3. GraphRAG ---
        current_stage = "generation"

        t0 = time.time()
        query_result, graphs_folder, t_usage = rag.run_pipeline(
            preprocessed_sources,
            claim.text,
            claim.id,
            response_format_instructions=input_text.response_format_instructions,
        )
        latencies["generation"] = time.time() - t0

        tokens["generation"] = t_usage.get("llm_total", t_usage.get("total", 0))
        calls["generation"] = t_usage.get("llm_calls", t_usage.get("calls", 0))

        # --- 4. Verdict Parsing ---
        current_stage = "parsing"

        if not isinstance(query_result, str) or not query_result.strip():
            raise RuntimeError("GraphRAG returned an empty response.")

        # If it's a strict experiment prompt, it will have the tags
        if input_text.response_format_instructions:
            if "VERDICT:" not in query_result or "REASONING:" not in query_result:
                raise RuntimeError("GraphRAG returned an unstructured response.")

            predicted_label = (
                query_result.split("REASONING:")[0].replace("VERDICT:", "").strip()
            )
            # --- 5. Answer Entity (Experiment Mode) ---
            reasoning = query_result.split("REASONING:", 1)[1].strip()

        else:
            predicted_label = None
            # --- 5. Answer Entity (UI Mode) ---
            reasoning = query_result.strip()

        current_stage = "database_logging"

        answer = Answer(claim.id, reasoning, graphs_folder)

        evidence_data = {
            "claim_text": text,
            "raw_sources": sources,
            "query_result": reasoning,
        }

        return {
            "claim_id": claim_id,
            "claim_title": claim_title,
            "sources": preprocessed_sources,
            "query_result": reasoning,
            "predicted_label": predicted_label,
            "graphs_folder": graphs_folder,
            "answer": answer.answer,
            "metrics": {"latencies": latencies, "tokens": tokens, "calls": calls},
            "evidence_data": evidence_data,
        }

    except Exception as e:
        if claim is not None and not input_text.preserve_failed_claim:
            try:
                claim.clear_database()
            except Exception as cleanup_error:
                logger.exception(
                    f"Failed to clean up claim {claim.id} after pipeline error: {cleanup_error}"
                )

        graph_folder = os.path.join(rag.graph_folder, claim_id)

        if os.path.isdir(graph_folder):
            try:
                shutil.rmtree(graph_folder)
            except OSError as graph_cleanup_error:
                logger.exception(
                    f"Failed to clean up graph folder '{graph_folder}' "
                    f"after pipeline error: {graph_cleanup_error}"
                )

        if input_text.preserve_failed_claim:
            raise HTTPException(
                status_code=500,
                detail={
                    "claim_id": claim_id,
                    "stage": current_stage,
                    "type": type(e).__name__,
                    "message": str(e),
                    "metrics": {
                        "latencies": latencies,
                        "tokens": tokens,
                        "calls": calls,
                    },
                },
            )

        raise


@backend_app.post("/delete_db")
def delete_database():
    db.delete_all_conversations()


@backend_app.get("/get_history")
def get_database():
    history = db.get_history()
    return history
