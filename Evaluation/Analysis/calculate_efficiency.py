import pandas as pd

from Database.sqldb import Database
from log import Logger

logger = Logger("efficiency").get_logger()


def calculate_efficiency():
    logger.info("Loading efficiency data using the Database manager...")

    db = Database()

    query = """
        SELECT 
            system_type, environment, dataset_name, experiment_type,
            latency_preprocessor, latency_retrieval, latency_generation,
            tokens_preprocessor, tokens_retrieval, tokens_generation,
            calls_preprocessor, calls_retrieval, calls_generation
        FROM experiments
    """

    try:
        rows = db.fetch_all(query)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    if not rows:
        logger.warning(
            "No valid experiment data found! Have you run any evaluations yet?"
        )
        return

    total_rows = len(rows)
    logger.info(f"Successfully loaded {total_rows} fact-checking trials.")

    # Convert to Pandas DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    group_cols = ["system_type", "dataset_name", "environment", "experiment_type"]
    null_count = df[group_cols].isna().any(axis=1).sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} trials with NULL metadata in {group_cols}.")

    # Calculate the 'Total Pipeline' metrics for each claim
    df["total_latency"] = (
        df["latency_preprocessor"] + df["latency_retrieval"] + df["latency_generation"]
    )
    df["total_tokens"] = (
        df["tokens_preprocessor"] + df["tokens_retrieval"] + df["tokens_generation"]
    )
    df["total_calls"] = (
        df["calls_preprocessor"] + df["calls_retrieval"] + df["calls_generation"]
    )

    # =========================================================
    # MULTI-SYSTEM EFFICIENCY REPORT
    # =========================================================

    # dropna=False ensures trials with NULL metadata are not silently dropped
    grouped_experiments = df.groupby(group_cols, dropna=False)

    total_analyzed = 0

    def is_nullish(value):
        if value is None or value is pd.NA:
            return True
        if isinstance(value, float):
            return pd.isna(value)
        return False

    for (sys_type, ds_name, env, exp_type), group in grouped_experiments:
        group_size = len(group)
        total_analyzed += group_size

        display_sys = "UNKNOWN / NULL" if is_nullish(sys_type) else sys_type
        display_ds = "UNKNOWN / NULL" if is_nullish(ds_name) else ds_name
        display_env = "UNKNOWN / NULL" if is_nullish(env) else env
        display_exp = "UNKNOWN / NULL" if is_nullish(exp_type) else exp_type

        logger.info(f"SYSTEM: {display_sys}")
        logger.info(
            f"DATASET: {display_ds} | ENV: {display_env} | EXP TYPE: {display_exp}"
        )
        logger.info(f"SAMPLE SIZE: {group_size} claims")

        # Build the table as a single formatted string
        table = "\n"
        table += f"{'STAGE':<28} | {'LATENCY':>12} | {'TOKENS':>12} | {'CALLS':>10}\n"
        table += "-" * 72 + "\n"

        def get_row(stage_name, lat_col, tok_col, call_col):
            avg_lat = group[lat_col].mean()
            avg_tok = group[tok_col].mean()
            avg_cal = group[call_col].mean()
            return f"{stage_name:<28} | {avg_lat:>8.2f} sec | {avg_tok:>12.1f} | {avg_cal:>6.1f} calls\n"

        # Stage Rows
        table += get_row(
            "1. Preprocessor",
            "latency_preprocessor",
            "tokens_preprocessor",
            "calls_preprocessor",
        )
        table += get_row(
            "2. Retrieval", "latency_retrieval", "tokens_retrieval", "calls_retrieval"
        )
        table += get_row(
            "3. Generation/Verification",
            "latency_generation",
            "tokens_generation",
            "calls_generation",
        )
        table += "-" * 65 + "\n"

        # Total Row
        table += get_row(
            "TOTAL END-TO-END", "total_latency", "total_tokens", "total_calls"
        )

        logger.info(table)
        logger.info("=" * 20)

    # Final sanity check: make sure all loaded rows were analyzed
    if total_analyzed != total_rows:
        logger.warning(
            f"Discrepancy detected: Loaded {total_rows} trials, but analyzed {total_analyzed} trials."
        )


if __name__ == "__main__":
    calculate_efficiency()
