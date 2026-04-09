import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from Database.sqldb import Database


def calculate_effectiveness():
    print("Loading experiment data using the Database manager...")

    # 1. Initialize your Database connection
    db = Database()

    query = """
        SELECT system_type, dataset_setting, ground_truth, predicted_label 
        FROM experiments 
        WHERE ground_truth IN ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO')
    """

    try:
        # 2. Use our existing fetch_all method
        rows = db.fetch_all(query)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    if not rows:
        print("No valid experiment data found! Have you run any evaluations yet?")
        return

    print(f"Successfully loaded {len(rows)} fact-checking trials.\n")

    # 3. Convert the SQLite rows into a Pandas DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    # Clean up predictions just in case the LLM added extra spaces or formatting
    df["predicted_label"] = df["predicted_label"].apply(
        lambda x: str(x).strip().upper()
    )

    # =========================================================
    # MULTI-SYSTEM COMPARISON REPORT
    # =========================================================

    # Group the dataframe by the specific system and dataset
    grouped_experiments = df.groupby(["system_type", "dataset_setting"])

    for (sys_type, ds_setting), group in grouped_experiments:
        print("=" * 70)
        print(f"🚀 SYSTEM: {sys_type}")
        print(f"📁 DATASET: {ds_setting}")
        print(f"📊 SAMPLE SIZE: {len(group)} claims")
        print("=" * 70)

        y_true = group["ground_truth"].tolist()
        y_pred = group["predicted_label"].tolist()

        # Calculate Overall Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

        # Calculate FEVER Score (Currently matches accuracy for controlled evidence)
        fever_score = acc
        print(f"Strict FEVER Score: {fever_score:.4f} ({fever_score*100:.2f}%)")

        if ds_setting == "FEVER-Controlled":
            print(
                "*(Note: In the Controlled setting, evidence is perfectly provided, so FEVER Score == Accuracy.)*\n"
            )
        else:
            print("\n")

        # Generate the detailed per-label report
        report = classification_report(
            y_true,
            y_pred,
            labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            zero_division=0,
        )

        print("Per-Label Breakdown:")
        print(report)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    calculate_effectiveness()
