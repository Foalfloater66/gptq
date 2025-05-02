#!/bin/bash

# --- Configuration ---
MODEL="facebook/opt-125m"  # Model to evaluate
DATASET="wikitext2"       # Calibration dataset
WBITS=4                   # Bit width for quantization (adjust as needed)
NSAMPLES=128              # Number of calibration samples
OUTPUT_FILE="opt125m_w${WBITS}_eval_results.jsonl" # Output file for results
USE_NEW_EVAL="--new-eval" # Add '--new-eval' flag if needed, otherwise set to ""
# Add other common flags like --groupsize, --sym, --act-order etc. here
# Example: COMMON_FLAGS="--groupsize 128 --act-order --static-groups"
COMMON_FLAGS=""

# List of quantizers to test (match choices in opt.py)
QUANTIZERS=("uniform_minmax" "apot" "lloydmax" "logarithm") # Add others: "quantile", "kmeans"

# GPUs to use
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# --- Script Logic ---

# Optional: Clear previous results file if starting fresh
# > "$OUTPUT_FILE"

echo "Starting parallel evaluations for $MODEL (W=${WBITS}) across $NUM_GPUS GPUs"
echo "Results will be saved to $OUTPUT_FILE"
echo "================================================="

pids=()
quant_idx=0
# Ensure output file exists and is writable before starting background jobs
touch "$OUTPUT_FILE" || { echo "Error: Cannot create or write to output file $OUTPUT_FILE"; exit 1; }

for QUANTIZER in "${QUANTIZERS[@]}"; do
  # Assign GPU in a round-robin fashion
  GPU_ID=${GPUS[$((quant_idx % NUM_GPUS))]}

  echo "--- Assigning Quantizer: $QUANTIZER to GPU: $GPU_ID ---"

  # Construct the command for this quantizer and GPU
  # Ensure arguments with spaces or special characters are quoted
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python opt.py \
    \"$MODEL\" \
    \"$DATASET\" \
    --wbits $WBITS \
    --nsamples $NSAMPLES \
    --quantizer \"$QUANTIZER\" \
    --output-file \"$OUTPUT_FILE\" \
    --quiet \
    $USE_NEW_EVAL \
    $COMMON_FLAGS"
    # Add any other specific flags if needed, escaping appropriately

  echo "Executing (background): $CMD"
  # Run the command in the background
  eval $CMD &
  # Store the process ID of the background job
  pids+=($!)

  # Optional: Add a small delay to stagger GPU starts if needed
  # sleep 1

  ((quant_idx++))
done

# Wait for all background processes launched in the loop to complete
echo "Waiting for all background jobs (${pids[@]}) to complete..."
wait "${pids[@]}"

# Check exit status of background jobs (optional but recommended)
EXIT_STATUS=0
for pid in "${pids[@]}"; do
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        echo "Warning: Process $pid exited with status $status."
        EXIT_STATUS=1 # Record that at least one job failed
    fi
done

echo "================================================="
if [ $EXIT_STATUS -eq 0 ]; then
  echo "All evaluations completed successfully."
else
  echo "Some evaluations may have failed. Check logs and $OUTPUT_FILE."
fi
echo "Results saved in $OUTPUT_FILE"
echo "================================================="

# Example: Show summary using jq (if installed)
# if command -v jq &> /dev/null; then
#    echo "Summary of results:"
#    jq -c '{quantizer, wbits, results}' "$OUTPUT_FILE"
# fi

exit $EXIT_STATUS
