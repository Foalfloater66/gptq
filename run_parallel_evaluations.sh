#!/bin/bash

# --- Configuration ---
MODEL="bigscience/bloom-560m"  # Model to evaluate (e.g., BLOOM 560m)
DATASET="wikitext2"       # Calibration dataset
WBITS=4                   # Bit width for quantization (adjust as needed)
NSAMPLES=128              # Number of calibration samples
OUTPUT_FILE="bloom560m_w${WBITS}_eval_results.json" # Output file for results (using .jsonl)
# Common flags for bloom.py (adjust as needed)
# Note: BLOOM uses --percdamp, --groupsize, --sym. Add --new-eval if desired.
COMMON_FLAGS="--groupsize 128 --percdamp 0.01" # Example flags

# List of quantizers to test (match choices in bloom.py)
QUANTIZERS=("uniform_minmax" "apot" "lloydmax" "logarithm" "quantile" "kmeans") # All 6 quantizers

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
# Using 'a' mode with touch/redirection to avoid clearing if it exists
: > "$OUTPUT_FILE" # Clear the file at the start, or use `>` if preferred over `touch`
if [ $? -ne 0 ]; then
    echo "Error: Cannot create or write to output file $OUTPUT_FILE"
    exit 1
fi


for QUANTIZER in "${QUANTIZERS[@]}"; do
  # Assign GPU in a round-robin fashion
  GPU_ID=${GPUS[$((quant_idx % NUM_GPUS))]}

  # If we have already launched NUM_GPUS jobs, wait for the job
  # that previously used the GPU we are about to assign.
  if [ $quant_idx -ge $NUM_GPUS ]; then
    WAIT_PID_INDEX=$((quant_idx - NUM_GPUS))
    echo "--- GPU $GPU_ID busy. Waiting for job ${pids[$WAIT_PID_INDEX]} (Quantizer: ${QUANTIZERS[$WAIT_PID_INDEX]}) to finish... ---"
    wait "${pids[$WAIT_PID_INDEX]}"
    # Optional: Check exit status of the waited job here if needed
    # status=$?
    # if [ $status -ne 0 ]; then ... fi
  fi

  echo "--- Launching Quantizer: $QUANTIZER on GPU: $GPU_ID ---"

  # Construct the command for this quantizer and GPU
  # Ensure arguments with spaces or special characters are quoted
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python bloom.py \
    \"$MODEL\" \
    \"$DATASET\" \
    --wbits $WBITS \
    --nsamples $NSAMPLES \
    --quantizer \"$QUANTIZER\" \
    --output-file \"$OUTPUT_FILE\" \
    --quiet \
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

# Wait for the remaining background processes (the last batch)
echo "--- All jobs launched. Waiting for the last batch of jobs to complete... ---"
# Wait for all remaining child processes without specific PIDs
wait

# Check the status of the last batch of jobs specifically
# These jobs haven't been explicitly waited for with status check yet inside the loop.
# The number of jobs in the last batch is NUM_GPUS or less.
LAST_BATCH_START_INDEX=$(( ${#pids[@]} - NUM_GPUS ))
if [ $LAST_BATCH_START_INDEX -lt 0 ]; then
    LAST_BATCH_START_INDEX=0
fi

echo "--- Checking exit status of the last batch of jobs (indices ${LAST_BATCH_START_INDEX} to $((${#pids[@]} - 1))) ---"
for i in $(seq $LAST_BATCH_START_INDEX $((${#pids[@]} - 1)) ); do
    pid_to_check=${pids[$i]}
    quantizer_to_check=${QUANTIZERS[$i]}
    # Use `wait $pid` again here; it's safe even if already waited for.
    # It will return immediately if the process is already finished and give the exit status.
    wait "$pid_to_check"
    status=$?
     if [ $status -ne 0 ]; then
        # Check if we already recorded a failure for this PID during the in-loop wait
        # This avoids duplicate warnings if a job failed before the final wait.
        # However, the simplest robust approach is just to report failure if status is non-zero here.
        echo "Warning: Final check shows process $pid_to_check (Quantizer: $quantizer_to_check) exited with status $status."
        EXIT_STATUS=1 # Record failure if not already recorded
    else
        echo "--- Final check confirms job $pid_to_check (Quantizer: $quantizer_to_check) finished successfully. ---"
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
