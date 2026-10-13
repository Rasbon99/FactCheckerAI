#!/bin/bash

# ==============================================================================
# FACTCHECKER-AI: MASTER EXPERIMENT EXECUTION SCRIPT
# ==============================================================================

# Load variables from key.env (if it exists) to get robustness directories
if [ -f "key.env" ]; then
    export $(grep -v '^#' key.env | xargs)
fi

# Fallbacks in case key.env doesn't have them
FEVER_ROB_DIR=${FEVER_ROBUSTNESS_DIR:-"Datasets/FEVER/Robustness"}
AVERITEC_ROB_DIR=${AVERITEC_ROBUSTNESS_DIR:-"Datasets/AVERITEC/Robustness"}

# 1. Define the parameters for the grid search
EXPERIMENT_TYPES=("standard" "noisy" "missing" "conflicting")

# 2. Define the Python runner scripts
RUNNERS=(
    "Evaluation.Runners.ClosedBook.run_baseline_llm_only"
    "Evaluation.Runners.Controlled.run_baseline_sparse"
    "Evaluation.Runners.OpenWeb.run_baseline_sparse"
    "Evaluation.Runners.Controlled.run_baseline_hybrid"
    "Evaluation.Runners.OpenWeb.run_baseline_hybrid"
    "Evaluation.Runners.Controlled.run_baseline_prompt_stuffing"
    "Evaluation.Runners.OpenWeb.run_baseline_prompt_stuffing"
    "Evaluation.Runners.Controlled.run_foxai"
    "Evaluation.Runners.OpenWeb.run_foxai"
)

# 3. Start the Infrastructure Services

# A. LLM Server
echo "[Infrastructure] Starting llama.cpp server..."
python start_llamacpp_server.py &
LLAMA_PID=$!
sleep 10  # Give it time to load the model into VRAM

# B. Backend Server
echo "[Infrastructure] Starting the backend server..."
python start_backend_server.py &
BACKEND_PID=$!
sleep 15  # Give it time to connect to Neo4j and load embedding models

# Safely catch Ctrl+C to shut EVERYTHING down cleanly
trap "echo 'Stopping all infrastructure...'; kill $BACKEND_PID; kill $LLAMA_PID; exit" INT

echo "=========================================================="
echo "STARTING LARGE-SCALE EXPERIMENT PIPELINE"
echo "=========================================================="

# ------------------------------------------------------------------------------
# BLOCK 1: FEVER EXPERIMENTS (No Metadata Flag)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "STARTING FEVER EXPERIMENTS"
echo "=========================================================="
export EXPERIMENT_ACTIVE_DATASET="FEVER"

# Explicitly unset the metadata flag to guarantee it is empty during FEVER runs
unset AVERITEC_USE_METADATA 

for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "=> CONFIGURATION: FEVER | Type: $EXP_TYPE"
    echo "----------------------------------------------------------"

    export EXPERIMENT_TYPE=$EXP_TYPE
    
    # Set the FEVER dataset paths
    if [ "$EXP_TYPE" == "standard" ]; then
        export FEVER_DATASET_PATH="Datasets/FEVER/fever_dev_dataset.jsonl"
    else
        export FEVER_DATASET_PATH="${FEVER_ROB_DIR}/fever_dev_${EXP_TYPE}.jsonl"
    fi

    # Execute all baselines for this configuration
    for RUNNER in "${RUNNERS[@]}"; do
        echo "[Running] $RUNNER..."
        python -m "$RUNNER"
        
        # Check for crash
        if [ $? -ne 0 ]; then
            echo "[Error] $RUNNER encountered an error. Moving to next script."
        fi
        sleep 5 # Brief pause
    done
done


# ------------------------------------------------------------------------------
# BLOCK 2: AVERITEC EXPERIMENTS (Iterates over Metadata Flag)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "STARTING AVERITEC EXPERIMENTS"
echo "=========================================================="
export EXPERIMENT_ACTIVE_DATASET="AVERITEC"

for EXP_TYPE in "${EXPERIMENT_TYPES[@]}"; do
    for USE_META in "False" "True"; do
        
        echo ""
        echo "----------------------------------------------------------"
        echo "=> CONFIGURATION: AVERITEC | Type: $EXP_TYPE | Metadata: $USE_META"
        echo "----------------------------------------------------------"

        export EXPERIMENT_TYPE=$EXP_TYPE
        export AVERITEC_USE_METADATA=$USE_META
        
        # Set the AVERITEC dataset paths
        if [ "$EXP_TYPE" == "standard" ]; then
            export AVERITEC_DATASET_PATH="Datasets/AVERITEC/averitec_dev_dataset.json"
        else
            export AVERITEC_DATASET_PATH="${AVERITEC_ROB_DIR}/averitec_dev_${EXP_TYPE}.json"
        fi

        # Execute all baselines for this configuration
        for RUNNER in "${RUNNERS[@]}"; do
            echo "[Running] $RUNNER..."
            python -m "$RUNNER"
            
            # Check for crash
            if [ $? -ne 0 ]; then
                echo "[Error] $RUNNER encountered an error. Moving to next script."
            fi
            sleep 5 # Brief pause
        done
        
    done
done

echo ""
echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================================="

# Shut down all background services in reverse order
echo "Shutting down Backend and LLM servers..."
kill $BACKEND_PID
kill $LLAMA_PID