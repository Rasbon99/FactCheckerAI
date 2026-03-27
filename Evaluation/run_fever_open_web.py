import os
import json
import requests
import time
import dotenv

dotenv.load_dotenv("key.env", override=True)

# 1. Configuration
DATASET_PATH = os.getenv("EVALUATION_DATASET_PATH", "Datasets/fever_dev_dataset.jsonl")
BACKEND_URL = os.getenv("BACKEND_API_URL")
API_URL = f"{BACKEND_URL}/run_pipeline"

MAX_CLAIMS_TO_TEST = 5  # Start small for the first test!


def run_experiment():
    print(f"Starting Experiment B (Open Web) with {MAX_CLAIMS_TO_TEST} claims...")
    print(f"Reading dataset from: {DATASET_PATH}")
    print(f"Sending requests to: {API_URL}")

    successful_runs = 0
    failed_runs = 0

    # 2. Open and read the FEVER dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):

                if line_number >= MAX_CLAIMS_TO_TEST:
                    break

                data = json.loads(line)
                claim_text = data.get("claim", "")
                ground_truth = data.get("label", "")

                print(
                    f"\n[{line_number + 1}/{MAX_CLAIMS_TO_TEST}] Processing: {claim_text[:50]}..."
                )
                print(f"Ground Truth: {ground_truth}")

                # 3. Prepare the payload
                payload = {"text": claim_text, "ground_truth": ground_truth}

                # 4. Send the request to FoxAI
                try:
                    response = requests.post(API_URL, json=payload, timeout=120)

                    if response.status_code == 200:
                        print(f"Success! FoxAI Verdict generated.")
                        successful_runs += 1
                    else:
                        print(f"Backend Error {response.status_code}: {response.text}")
                        failed_runs += 1

                except Exception as e:
                    print(f"Request failed: {e}")
                    failed_runs += 1

                # Small sleep for the Groq API rate limits
                time.sleep(2)

    except FileNotFoundError:
        print(
            f"ERROR: Could not find the dataset at {DATASET_PATH}. Please check your .env file!"
        )
        return

    print("\n" + "=" * 40)
    print(f"EXPERIMENT COMPLETE!")
    print(f"Successful processing: {successful_runs}")
    print(f"Failed processing: {failed_runs}")
    print("Check your SQLite 'experiments' table to see the logged metrics!")
    print("=" * 40)


if __name__ == "__main__":
    run_experiment()
