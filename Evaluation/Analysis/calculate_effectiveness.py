import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from Database.sqldb import Database
from log import Logger

logger = Logger("effectiveness").get_logger()


def calculate_effectiveness():
    logger.info("Loading experiment data using the Database manager...")

    # 1. Initialize your Database connection
    db = Database()

    query = """
        SELECT system_type, environment, dataset_name, experiment_type, use_metadata, ground_truth, predicted_label 
        FROM experiments 
        WHERE ground_truth IS NOT NULL AND ground_truth NOT IN ('', 'Not Provided')
    """

    try:
        # 2. Use our existing fetch_all method
        rows = db.fetch_all(query)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    if not rows:
        logger.warning(
            "No valid experiment data found! Have you run any evaluations yet?"
        )
        return

    logger.info(f"Successfully loaded {len(rows)} fact-checking trials.")

    # 3. Convert the SQLite rows into a Pandas DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    # Standardize BOTH columns to uppercase to avoid case-mismatch errors
    df["ground_truth"] = df["ground_truth"].apply(lambda x: str(x).strip().upper())
    df["predicted_label"] = df["predicted_label"].apply(
        lambda x: str(x).strip().upper()
    )

    # =========================================================
    # MULTI-SYSTEM COMPARISON REPORT
    # =========================================================

    grouped_experiments = df.groupby(
        [
            "system_type",
            "dataset_name",
            "environment",
            "experiment_type",
            "use_metadata",
        ]
    )

    for (sys_type, ds_name, env, exp_type, use_meta), group in grouped_experiments:
        logger.info(f"SYSTEM: {sys_type}")
        meta_str = "ON" if use_meta else "OFF"
        logger.info(
            f"DATASET: {ds_name} | ENV: {env} | EXP TYPE: {exp_type} | METADATA: {meta_str}"
        )
        logger.info(f"SAMPLE SIZE: {len(group)} claims")

        y_true = group["ground_truth"].tolist()
        y_pred = group["predicted_label"].tolist()

        # Calculate Overall Accuracy
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

        dataset_upper = str(ds_name).upper()
        if "AVERITEC" in dataset_upper:
            expected_labels = [
                "SUPPORTED",
                "REFUTED",
                "NOT ENOUGH EVIDENCE",
                "CONFLICTING EVIDENCE/CHERRY-PICKING",
            ]
        elif "FEVER" in dataset_upper:
            expected_labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        else:
            expected_labels = sorted(list(set(y_true + y_pred)))

        # Clean the predictions and track hallucinations
        cleaned_y_pred = []
        hallucinated_labels = set()

        for pred in y_pred:
            if pred in expected_labels:
                cleaned_y_pred.append(pred)
            else:
                cleaned_y_pred.append("INVALID_PREDICTION")
                hallucinated_labels.add(pred)

        # Generate the detailed per-label report
        report = classification_report(
            y_true,
            cleaned_y_pred,
            labels=expected_labels,
            zero_division=0,
        )

        logger.info("Per-Label Breakdown:")
        logger.info("\n%s", report)

        if hallucinated_labels:
            invalid_count = cleaned_y_pred.count("INVALID_PREDICTION")
            logger.warning(
                f"⚠️ This system produced {invalid_count} invalid/unstructured responses."
            )
            logger.warning(f"Raw Invalid Labels Caught: {hallucinated_labels}")

        logger.info("=" * 20)


if __name__ == "__main__":
    calculate_effectiveness()
