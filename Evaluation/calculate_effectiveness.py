import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from Database.sqldb import Database


def calculate_effectiveness():
    print("Loading experiment data using the Database manager...")

    # 1. Initialize your Database connection
    db = Database()

    query = """
        SELECT ground_truth, predicted_label 
        FROM experiments 
        WHERE ground_truth IN ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO')
    """

    try:
        # 2. Use your existing fetch_all method
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

    # 3. Convert the SQLite rows into a Pandas DataFrame
    # Since your sqldb.py uses sqlite3.Row, we can easily convert them to dictionaries
    df = pd.DataFrame([dict(row) for row in rows])

    # 4. Extract the columns into lists for Scikit-Learn
    y_true = df["ground_truth"].tolist()

    # Clean up predictions just in case the LLM added extra spaces or formatting
    y_pred = df["predicted_label"].apply(lambda x: str(x).strip().upper()).tolist()

    # 5. Calculate and print the metrics!
    print("=" * 60)
    print("RQ1: EFFECTIVENESS METRICS (Precision, Recall, F1)")
    print("=" * 60)

    # Calculate overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    # Generate the detailed per-label report
    report = classification_report(
        y_true,
        y_pred,
        labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        zero_division=0,  # Prevents ugly warnings if a class has 0 predictions right now
    )

    print("Per-Label Breakdown:")
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    calculate_effectiveness()
