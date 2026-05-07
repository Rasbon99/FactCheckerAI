import os
import json
import random
import dotenv
from log import Logger

# Load environment variables
dotenv.load_dotenv("key.env", override=True)

# File Paths
ORIGINAL_DATASET_PATH = os.getenv(
    "FEVER_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl"
)
ROBUSTNESS_DIR = "Datasets/RobustnessTests"
logger = Logger("generate_robustness_datasets").get_logger()

# Ensure the new directory exists
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

# Set a random seed so your thesis experiments are 100% reproducible!
random.seed(42)


def load_base_claims(filepath, max_claims=100):
    """Loads the first 100 claims from the original dataset."""
    claims = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if line_number >= max_claims:
                break
            claims.append(json.loads(line))
    return claims


def generate_missing_dataset(base_claims):
    """
    TRAP 1: MISSING EVIDENCE
    Empties the evidence array. The ground truth label MUST be changed to
    'NOT ENOUGH INFO' because without evidence, the claim is unprovable.
    """
    dataset = []
    for row in base_claims:
        new_row = row.copy()
        new_row["evidence"] = []  # Wipe the evidence
        new_row["label"] = "NOT ENOUGH INFO"  # Update ground truth
        dataset.append(new_row)

    out_path = os.path.join(ROBUSTNESS_DIR, "missing.jsonl")
    save_dataset(dataset, out_path)
    logger.info(f"Generated Missing Dataset: {out_path}")


def generate_noisy_dataset(base_claims):
    """
    TRAP 2: NOISY EVIDENCE
    Keeps the true evidence, but appends evidence from 3 completely random
    claims to simulate a bloated, noisy web scrape. Ground truth remains unchanged.
    """
    dataset = []
    # Extract all evidence blocks to use as a pool of "noise"
    all_evidence_blocks = [
        row.get("evidence", []) for row in base_claims if row.get("evidence")
    ]

    for row in base_claims:
        new_row = row.copy()
        current_evidence = new_row.get("evidence", [])

        # Pick 3 random evidence blocks to act as noise
        noise = random.sample(all_evidence_blocks, 3)

        # Flatten the noise into the current evidence array
        for noise_block in noise:
            current_evidence.extend(noise_block)

        new_row["evidence"] = current_evidence
        dataset.append(new_row)

    out_path = os.path.join(ROBUSTNESS_DIR, "noisy.jsonl")
    save_dataset(dataset, out_path)
    logger.info(f"Generated Noisy Dataset: {out_path}")


def generate_conflicting_dataset(base_claims):
    """
    TRAP 3: CONFLICTING EVIDENCE
    Pairs a claim's true evidence with the evidence of a claim that has the
    OPPOSITE label. This forces the LLM to look at conflicting 'facts'.
    """
    dataset = []

    # Separate claims by label to create conflicts
    supports_evidence = [
        row["evidence"] for row in base_claims if row["label"] == "SUPPORTS"
    ]
    refutes_evidence = [
        row["evidence"] for row in base_claims if row["label"] == "REFUTES"
    ]

    for row in base_claims:
        new_row = row.copy()
        current_evidence = new_row.get("evidence", [])

        # Inject conflicting evidence based on the current label
        if row["label"] == "SUPPORTS" and refutes_evidence:
            conflict = random.choice(refutes_evidence)
            current_evidence.extend(conflict)
        elif row["label"] == "REFUTES" and supports_evidence:
            conflict = random.choice(supports_evidence)
            current_evidence.extend(conflict)

        new_row["evidence"] = current_evidence
        dataset.append(new_row)

    out_path = os.path.join(ROBUSTNESS_DIR, "conflicting.jsonl")
    save_dataset(dataset, out_path)
    logger.info(f"Generated Conflicting Dataset: {out_path}")


def save_dataset(dataset, filepath):
    """Writes the dataset list back to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        logger.error(f"Could not find original dataset at {ORIGINAL_DATASET_PATH}")
    else:
        logger.info("Starting Robustness Dataset Generation...")
        claims = load_base_claims(ORIGINAL_DATASET_PATH)

        generate_missing_dataset(claims)
        generate_noisy_dataset(claims)
        generate_conflicting_dataset(claims)

        logger.info("All 3 robustness datasets are ready for evaluation!")
