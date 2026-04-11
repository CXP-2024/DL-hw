#!/bin/bash
# Quick experiment runner
# Usage: bash run_experiment.sh <config_name> <config_file>

set -e

CONFIG_NAME=$1
CONFIG_FILE=$2

echo "========================================="
echo "Experiment: $CONFIG_NAME"
echo "Config: $CONFIG_FILE"
echo "========================================="

# Clean old checkpoint
rm -rf checkpoints/final

# Train
echo "[TRAIN] Starting..."
python train.py \
  --icon-qa-train-dataset data/icon-qa-train.arrow \
  --custom-train-dataset custom.arrow \
  --sft-config "$CONFIG_FILE" 2>&1 | grep -E "loss|train_runtime|Num examples|Total steps|Batch"

# Evaluate
echo "[EVAL] Starting..."
python evaluate.py \
  --dataset data/icon-qa-val.arrow \
  --checkpoint checkpoints/final 2>&1 | grep -E "Accuracy"

echo "========================================="
echo "Experiment $CONFIG_NAME DONE"
echo "========================================="
