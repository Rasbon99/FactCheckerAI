import time
from Database.data_entities import Experiment


class ExperimentTracker:
    def __init__(
        self,
        claim_id,
        ground_truth="Not Provided",
    ):
        """
        Initializes the tracker to monitor latency, tokens, calls, and cost across pipeline stages.
        """
        self.claim_id = claim_id
        self.ground_truth = ground_truth

        # Dictionaries to hold the metrics for each stage
        self.latencies = {}
        self.tokens = {}
        self.calls = {}

    def run_stage(self, stage_name, func, *args, **kwargs):
        """
        Executes a pipeline stage, times it, and safely extracts token/call metrics.
        """
        start_time = time.time()

        # 1. Execute the target function
        output = func(*args, **kwargs)

        # 2. Record Latency
        latency = time.time() - start_time
        self.latencies[stage_name] = latency

        # 3. Safely unpack token data (if the function provides it)
        # We expect a tuple like: (result, {"total": 150, "calls": 1})
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[1], dict)
            and "total" in output[1]
        ):
            result, token_data = output
            tokens = token_data.get("total", 0)
            calls = token_data.get("calls", 0)
        else:
            # Fallback for stages that don't use LLMs or haven't been updated yet
            result = output
            tokens = 0
            calls = 0

        # 4. Store Metrics
        self.tokens[stage_name] = tokens
        self.calls[stage_name] = calls

        return result

    def finalize(self, predicted_label, evidence_data):
        """
        Packages all collected metrics and saves the final Experiment to the database.
        """
        experiment = Experiment(
            claim_id=self.claim_id,
            predicted_label=predicted_label,
            ground_truth=self.ground_truth,
            latencies=self.latencies,
            tokens=self.tokens,
            calls=self.calls,
            evidence_data=evidence_data,
        )
        return experiment
