import os
import json
import random
import dotenv
from log import Logger

from Evaluation.Utils.dataset_manager import DatasetManager

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

BASE_ROBUSTNESS_DIR = "Datasets/RobustnessTests"
logger = Logger("generate_robustness_datasets").get_logger()

# Set a random seed so your thesis experiments are 100% reproducible!
random.seed(42)


def save_dataset(dataset, filename_base, active_dataset):
    """Saves as JSONL for FEVER, or JSON Array for AVeriTeC, in specific subfolders."""
    # Ensure the dataset-specific subfolder exists
    target_dir = os.path.join(BASE_ROBUSTNESS_DIR, active_dataset)
    os.makedirs(target_dir, exist_ok=True)

    if active_dataset == "FEVER":
        filepath = os.path.join(target_dir, f"{filename_base}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
    else:
        filepath = os.path.join(target_dir, f"{filename_base}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)

    logger.info(f"Generated {filename_base} dataset: {filepath}")


def generate_missing_dataset(base_claims, active_dataset):
    """
    TRAP 1: MISSING EVIDENCE
    FEVER: Empties the evidence array.
    AVeriTeC: Corrupts the internal_id so the retriever fails.
    """
    dataset = []
    nei_label = (
        "NOT ENOUGH INFO" if active_dataset == "FEVER" else "Not Enough Evidence"
    )

    for row in base_claims:
        new_row = row.copy()
        new_row["label"] = nei_label  # Update ground truth

        if active_dataset == "FEVER":
            new_row["evidence"] = []
        else:
            new_row["internal_id"] = "TRAP_MISSING"

        dataset.append(new_row)

    save_dataset(dataset, "missing", active_dataset)


def generate_noisy_dataset(base_claims, active_dataset):
    """
    TRAP 2: NOISY EVIDENCE
    FEVER: Appends random evidence blocks directly.
    AVeriTeC: Adds a 'noisy_ids' array. (Note: To test this on AVeriTeC, you will
    need to tweak your retriever to also load these extra IDs during the test).
    """
    dataset = []

    if active_dataset == "FEVER":
        all_evidence = [
            row.get("evidence", []) for row in base_claims if row.get("evidence")
        ]
    else:
        all_ids = [row.get("internal_id") for row in base_claims]

    for row in base_claims:
        new_row = row.copy()

        if active_dataset == "FEVER":
            current_evidence = new_row.get("evidence", [])
            noise = random.sample(all_evidence, 3)
            for noise_block in noise:
                current_evidence.extend(noise_block)
            new_row["evidence"] = current_evidence

        elif active_dataset == "AVERITEC":
            # Pass instructions to the retriever to grab extra noise
            new_row["noisy_ids"] = random.sample(all_ids, 3)

        dataset.append(new_row)

    save_dataset(dataset, "noisy", active_dataset)


def generate_conflicting_dataset(base_claims, active_dataset):
    """
    TRAP 3: CONFLICTING EVIDENCE
    FEVER: Injects evidence from opposite labels.
    AVeriTeC: Swaps the internal_id with an opposite claim's ID.
    """
    dataset = []

    # Map labels dynamically
    supports_label = "SUPPORTS" if active_dataset == "FEVER" else "Supported"
    refutes_label = "REFUTES" if active_dataset == "FEVER" else "Refuted"

    supports_rows = [row for row in base_claims if row["label"] == supports_label]
    refutes_rows = [row for row in base_claims if row["label"] == refutes_label]

    for row in base_claims:
        new_row = row.copy()

        if row["label"] == supports_label and refutes_rows:
            conflict_source = random.choice(refutes_rows)
            if active_dataset == "FEVER":
                new_row["evidence"].extend(conflict_source.get("evidence", []))
            else:
                new_row["internal_id"] = conflict_source["internal_id"]

        elif row["label"] == refutes_label and supports_rows:
            conflict_source = random.choice(supports_rows)
            if active_dataset == "FEVER":
                new_row["evidence"].extend(conflict_source.get("evidence", []))
            else:
                new_row["internal_id"] = conflict_source["internal_id"]

        dataset.append(new_row)

    save_dataset(dataset, "conflicting", active_dataset)


if __name__ == "__main__":
    logger.info("Starting Robustness Dataset Generation...")

    # Use DatasetManager to handle loading logic intelligently
    manager = DatasetManager()
    active = manager.active_dataset

    logger.info(f"Targeting active environment: {active}")

    try:
        # Load exactly 100 claims for the robustness tests
        claims = manager.load_data(max_claims=100)

        generate_missing_dataset(claims, active)
        generate_noisy_dataset(claims, active)
        generate_conflicting_dataset(claims, active)

        logger.info(
            f"All 3 robustness datasets are ready and safely stored in Datasets/RobustnessTests/{active}/!"
        )
    except Exception as e:
        logger.error(f"Failed to generate datasets: {e}")
