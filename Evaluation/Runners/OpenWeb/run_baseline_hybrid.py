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
from Evaluation.Utils.dataset_manager import DatasetManager
from WebScraper.scraper import Scraper
from Database.data_entities import Claim, Answer, Experiment
from Utils.nomic_embedding import get_embedding_model

# Load environment variables
dotenv.load_dotenv("key.env", override=False)

# Configuration
MAX_CLAIMS_TO_TEST = 5
logger = Logger("HybridRAG-OpenWeb").get_logger()

# --- CONFIGURATION FLAG ---
USE_METADATA = os.getenv("AVERITEC_USE_METADATA") == "True"

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


def get_hybrid_rag_verdict(claim_text, best_evidence_string, prompt_instructions):
    """Asks the LLM to verify the claim using ONLY the top chunks found by Hybrid RAG (BM25 + Dense Embeddings)."""
    prompt = f"""You are a strict fact-checking AI.
    Verify the following claim using ONLY the provided evidence. 

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
    dataset_manager = DatasetManager()
    metadata = dataset_manager.get_experiment_metadata(environment="open_web")
    active_dataset = metadata["dataset_name"]

    prompt_instructions = dataset_manager.get_prompt_instructions()

    logger.info(
        f"Starting Baseline (HybridRAG - Open Web) with {MAX_CLAIMS_TO_TEST} claims..."
    )
    logger.info(f"Environment: {metadata['environment']}")
    logger.info(f"Active Dataset: {active_dataset}")
    logger.info(f"Experiment Type: {metadata['experiment_type']}")
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
            scraper = Scraper()

            claim_text = data.get("claim", "")
            ground_truth = data.get("label", "")

            search_query = claim_text
            if active_dataset == "AVERITEC" and USE_METADATA:
                search_query = dataset_manager.build_search_query(data)

            logger.info(f"[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")
            if search_query != claim_text:
                logger.info(f"Enriched Search Query: {search_query}")

            claim_id = str(uuid.uuid4())

            Claim(
                text=claim_text,
                title="[Hybrid] " + claim_text[:30] + "...",
                summary="Tested by scoring scraped pages using LangChain Dense Embeddings.",
                claim_id=claim_id,
            )

            # --- 1. Retrieval & Filtering (Scraper + LangChain Dense RAG) ---
            t0 = time.time()

            raw_scraped_sources, scraper_metrics = scraper.search_and_extract(
                search_query, num_results=10
            )

            if not raw_scraped_sources:
                best_evidence = "No relevant articles could be scraped."
            else:
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

                logger.info(
                    f"Scraped {len(docs)} pages. Chunking and embedding natively..."
                )

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100
                )
                splits = text_splitter.split_documents(docs)

                vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                top_docs = retriever.invoke(search_query)

                best_evidence = ""
                for i, doc in enumerate(top_docs):
                    best_evidence += f"\n--- MATCH {i+1} (Source: {doc.metadata['source']}) ---\n{doc.page_content}\n"

            latency_retrieval = time.time() - t0
            tokens_retrieval = scraper_metrics.get("total", 0)
            calls_retrieval = scraper_metrics.get("calls", 0)

            # --- 2. Generation (The LLM Call) ---
            t0 = time.time()
            query_result, tokens_used = get_hybrid_rag_verdict(
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

            logger.info(f"Hybrid RAG Verdict: {predicted_label}")

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
                    "retrieval": tokens_retrieval,
                    "generation": tokens_used,
                },
                calls={
                    "preprocessor": 0,
                    "retrieval": calls_retrieval,
                    "generation": 1,
                },
                evidence_data={
                    "claim_text": claim_text,
                    "raw_sources": raw_scraped_sources,
                    "hybrid_evidence": best_evidence,
                    "query_result": query_result,
                },
                system_type="HybridRAG",
                environment=metadata["environment"],
                dataset_name=metadata["dataset_name"],
                experiment_type=metadata["experiment_type"],
                use_metadata=metadata["use_metadata"],
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
