import os
import json
import time
import uuid
import dotenv
from groq import Groq

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

# --- Import Pipeline Components ---
from Evaluation.Utils.experiment_tracker import ExperimentTracker
from WebScraper.scraper import Scraper
from Database.data_entities import Claim, Answer

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# Configuration
DATASET_PATH = os.getenv("FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
MAX_CLAIMS_TO_TEST = 5

# Initialize Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)


def get_dense_rag_verdict(claim_text, best_evidence_string):
    """Asks the LLM to verify the claim using ONLY the top chunks found by Dense Vector Search."""
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


def run_dense_rag_baseline():
    print(f"\nStarting Baseline Dense Semantic RAG with {MAX_CLAIMS_TO_TEST} claims...")

    # --- Initialize Embeddings (Consistent with your Controlled Baseline) ---
    print("Loading Ollama Embeddings (This takes a few seconds)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    successful_runs = 0

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                # Instantiate Scraper inside the loop to avoid DuckDuckGo session bans
                scraper = Scraper()

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                print(f"\n[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Claim: {claim_text}")

                claim_id = str(uuid.uuid4())
                tracker = ExperimentTracker(
                    claim_id=claim_id,
                    ground_truth=ground_truth,
                    system_type="Baseline-DenseRAG",
                    dataset_setting="FEVER-OpenWeb",
                )

                Claim(
                    text=claim_text,
                    title="[Dense] " + claim_text[:30] + "...",
                    summary="Tested by scoring scraped pages using LangChain Dense Embeddings.",
                    claim_id=claim_id,
                )

                # --- 1. Retrieval & Filtering (Scraper + LangChain Dense RAG) ---
                def run_retrieval():
                    # Scrape with LLM filter OFF (saves tokens and ensures pure baseline comparison)
                    raw_scraped_sources = scraper.search_and_extract(
                        claim_text, num_results=10
                    )

                    if not raw_scraped_sources:
                        return "No relevant articles could be scraped.", {
                            "total": 0,
                            "calls": 1,
                        }

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

                    print(
                        f"  -> Scraped {len(docs)} pages. Chunking and embedding with Ollama..."
                    )

                    # Step B: Split the massive pages into clean, overlapping paragraphs
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=100
                    )
                    splits = text_splitter.split_documents(docs)

                    # Step C: Embed the chunks and store them in a temporary local vector space
                    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)

                    # Step D: Perform the semantic search to get the Top 3 chunks
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    top_docs = retriever.invoke(claim_text)

                    # Step E: Format the winning chunks into a single evidence string
                    best_evidence = ""
                    for i, doc in enumerate(top_docs):
                        best_evidence += f"\n--- MATCH {i+1} (Source: {doc.metadata['source']}) ---\n{doc.page_content}\n"

                    # Return evidence and 0 tokens (since scraping and embedding were free/local)
                    return best_evidence, {"total": 0, "calls": 1}

                # Run the retrieval stage (No tuple unpacking needed based on our previous fix)
                best_evidence = tracker.run_stage("retrieval", run_retrieval)

                # Safely clear tuple bug if it appears deep inside the tracker
                if isinstance(best_evidence, tuple):
                    best_evidence = best_evidence[0]

                # --- 2. Generation (The LLM Call) ---
                def run_llm():
                    # Groq reads the Top 3 dense chunks
                    res_text, toks = get_dense_rag_verdict(claim_text, best_evidence)
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

                print(f"  -> Dense RAG Verdict: {predicted_label}")

                Answer(claim_id=claim_id, answer=query_result, graphs_folder=None)

                # --- 4. Log to DB ---
                tracker.finalize(
                    predicted_label,
                    {
                        "claim_text": claim_text,
                        "dense_evidence": best_evidence,
                        "query_result": query_result,
                    },
                )

                successful_runs += 1
                print("Sleeping for 15 seconds to respect DuckDuckGo rate limits...")
                time.sleep(15)

    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "=" * 40)
    print("DENSE RAG (OPEN WEB) COMPLETE!")
    print("=" * 40)


if __name__ == "__main__":
    run_dense_rag_baseline()
