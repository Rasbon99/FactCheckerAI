import pandas as pd

from Database.sqldb import Database


def calculate_efficiency():
    print("Loading efficiency data using the Database manager...")

    db = Database()

    # 1. Pull all the efficiency metrics from the database
    query = """
        SELECT 
            latency_preprocessor, latency_retrieval, latency_graph_rag,
            tokens_preprocessor, tokens_retrieval, tokens_graph_rag,
            calls_preprocessor, calls_retrieval, calls_graph_rag
        FROM experiments
    """

    try:
        rows = db.fetch_all(query)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    if not rows:
        print(
            "No valid experiment data found! Have you run the open web evaluation yet?"
        )
        return

    print(f"Successfully loaded {len(rows)} fact-checking trials.\n")

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    # 3. Calculate the 'Total Pipeline' metrics for each claim
    df["total_latency"] = (
        df["latency_preprocessor"] + df["latency_retrieval"] + df["latency_graph_rag"]
    )
    df["total_tokens"] = (
        df["tokens_preprocessor"] + df["tokens_retrieval"] + df["tokens_graph_rag"]
    )
    df["total_calls"] = (
        df["calls_preprocessor"] + df["calls_retrieval"] + df["calls_graph_rag"]
    )

    # 4. Print the beautifully formatted Academic Table
    print("=" * 75)
    print("RQ2: EFFICIENCY METRICS (Averages per Claim)")
    print("=" * 75)

    # Helper function to print a row nicely
    def print_stage_metrics(stage_name, latency_col, token_col, call_col):
        avg_lat = df[latency_col].mean()
        avg_tok = df[token_col].mean()
        avg_cal = df[call_col].mean()
        print(
            f"{stage_name:<25} | {avg_lat:>9.2f} sec | {avg_tok:>10.1f} tokens | {avg_cal:>6.1f} calls"
        )

    # Table Header
    print(f"{'STAGE':<25} | {'LATENCY':>13} | {'TOKENS':>17} | {'CALLS':>12}")
    print("-" * 75)

    # Stage Rows
    print_stage_metrics(
        "1. Preprocessor",
        "latency_preprocessor",
        "tokens_preprocessor",
        "calls_preprocessor",
    )
    print_stage_metrics(
        "2. Retrieval (Web Scraping)",
        "latency_retrieval",
        "tokens_retrieval",
        "calls_retrieval",
    )
    print_stage_metrics(
        "3. GraphRAG Verification",
        "latency_graph_rag",
        "tokens_graph_rag",
        "calls_graph_rag",
    )
    print("-" * 75)

    # Total Row
    print_stage_metrics(
        "TOTAL END-TO-END", "total_latency", "total_tokens", "total_calls"
    )

    print("=" * 75)


if __name__ == "__main__":
    calculate_efficiency()
