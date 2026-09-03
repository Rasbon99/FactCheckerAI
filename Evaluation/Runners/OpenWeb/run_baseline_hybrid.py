import os
import time
import uuid
import dotenv
from groq import Groq
from log import Logger

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from Evaluation.Utils.dataset_manager import DatasetManager
from WebScraper.scraper import Scraper
from Database.data_entities import Claim, Answer
from Utils.nomic_embedding import get_embedding_model

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
MAX_CLAIMS_TO_TEST = 5
logger = Logger("Hybrid-OpenWeb").get_logger()

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


def get_hybrid_rag_verdict(
    claim_text, best_evidence_string, prompt_instructions, nei_label
):
    """Asks the LLM to verify the claim using ONLY the top chunks found by Hybrid RAG (BM25 + Dense Embeddings)."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 
    If the evidence does not contain enough information to make a definitive decision, answer exactly: {nei_label}.

    {prompt_instructions}

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


def run_hybrid_rag_baseline_openweb():
    # Initialize the Smart Dataset Manager
    dataset_manager = DatasetManager()
    active_dataset = dataset_manager.active_dataset
    tracker_env_name = dataset_manager.get_tracker_dataset_name("open-web")
    prompt_instructions = dataset_manager.get_prompt_instructions()
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    logger.info(
        f"Starting Baseline Hybrid RAG (Open Web) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Using Metadata Super Query: {USE_METADATA}")

    logger.info(
        "Loading Hugging Face Embeddings natively (This takes a few seconds)..."
    )
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"
    )
    embeddings = get_embedding_model(embedding_model_name)

    successful_runs = 0

    try:
        claims_data = dataset_manager.load_data(max_claims=MAX_CLAIMS_TO_TEST)

        for line_number, data in enumerate(claims_data):
            # 1. Instantiate Scraper inside the loop to avoid DuckDuckGo session bans
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
                system_type="Baseline-Hybrid",
                dataset_setting=tracker_env_name,
            )

            Claim(
                text=claim_text,
                title="[Hybrid] " + claim_text[:30] + "...",
                summary="Tested by scoring scraped pages using LangChain Hybrid RAG (BM25 + Dense Embeddings).",
                claim_id=claim_id,
            )

            # --- 1. Retrieval & Filtering (Scraper + LangChain Dense RAG) ---
            def run_retrieval():
                # Pass the enriched query to the scraper
                raw_scraped_sources, scraper_metrics = scraper.search_and_extract(
                    search_query, num_results=10
                )

                if not raw_scraped_sources:
                    return "No relevant articles could be scraped.", scraper_metrics

                # Step A: Convert raw scraped dictionaries into LangChain Documents
                docs = []
                for src in raw_scraped_sources:
                    body_text = src.get("body", "")
                    if body_text.strip():
                        docs.append(
                            Document(
                                page_content=body_text,
                                metadata={"source": src.get("url", "Unknown URL")},
                            )
                        )

                logger.info(f"Scraped {len(docs)} pages. Chunking and embedding...")

                # Step B: Split the massive pages into clean, overlapping paragraphs
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )
                splits = text_splitter.split_documents(docs)

                # Step C: Embed the chunks and store them in a temporary local vector space
                vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)

                # Step D: Perform the semantic search using the ENRICHED query
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                top_docs = retriever.invoke(search_query)

                # Step E: Format the winning chunks into a single evidence string
                best_evidence = ""
                for i, doc in enumerate(top_docs):
                    best_evidence += f"\n--- MATCH {i+1} (Source: {doc.metadata['source']}) ---\n{doc.page_content}\n"

                return best_evidence, scraper_metrics

            best_evidence = tracker.run_stage("retrieval", run_retrieval)

            if isinstance(best_evidence, tuple):
                best_evidence = best_evidence[0]

            # --- 2. Generation (The LLM Call) ---
            def run_llm():
                # Strictly pass the unedited claim_text to the generator
                res_text, toks = get_hybrid_rag_verdict(
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

            logger.info(f"Hybrid RAG Verdict: {predicted_label}")

            Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

            # --- 4. Log to DB ---
            tracker.finalize(
                predicted_label,
                {
                    "claim_text": claim_text,
                    "hybrid_evidence": best_evidence,
                    "query_result": query_result,
                },
            )

            successful_runs += 1
            logger.info("Sleeping for 15 seconds to respect DuckDuckGo rate limits...")
            time.sleep(15)

    except Exception as e:
        logger.error(f"{e}")

    logger.info("=" * 20)
    logger.info("HYBRID RAG (OPEN WEB) COMPLETE!")
    logger.info("=" * 20)


if __name__ == "__main__":
    run_hybrid_rag_baseline_openweb()
