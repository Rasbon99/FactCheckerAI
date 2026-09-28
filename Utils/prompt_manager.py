def get_dataset_prompt_instructions(dataset_name: str) -> str:
    """Returns the exact prompt rules based on the dataset or use-case."""
    dataset = dataset_name.upper()

    if dataset == "FEVER":
        return """
        You must format your response EXACTLY like this:
        VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
        REASONING: [Your brief explanation citing the provided evidence]
        
        RULES FOR VERDICT:
        - SUPPORTS: The provided evidence clearly proves the claim is true.
        - REFUTES: The provided evidence clearly proves the claim is false.
        - NOT ENOUGH INFO: The provided evidence does not contain sufficient information to judge the claim."""

    elif dataset == "AVERITEC":
        return """
        You must format your response EXACTLY like this:
        VERDICT: [Supported or Refuted or Not Enough Evidence or Conflicting Evidence/Cherry-picking]
        REASONING: [Your brief explanation citing the provided evidence]
        
        RULES FOR VERDICT:
        - Supported: The evidence clearly proves the claim is true.
        - Refuted: The evidence clearly proves the claim is false.
        - Not Enough Evidence: The evidence does not contain the information needed to judge the claim.
        - Conflicting Evidence/Cherry-picking: The claim is technically true but leaves out crucial context, is misleading, or the evidence is heavily mixed."""
