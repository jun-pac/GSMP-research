#!/bin/bash
#
# Convenience script to run temporal message passing experiments
#

set -e

# Default values
METHOD="smp"
EPOCHS=200
DEVICE="cuda:0"
NUM_RUNS=1
SEED=42
SAVE_DIR="./results"
HIDDEN_DIM=256
NUM_LAYERS=2
LR=0.01

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --hidden-dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --method METHOD           Temporal method: baseline|smp|ump|gsmp (default: smp)"
      echo "  --epochs EPOCHS           Number of epochs (default: 200)"
      echo "  --device DEVICE           Device: cuda:0|cpu (default: cuda:0)"
      echo "  --num-runs RUNS           Number of runs (default: 1)"
      echo "  --seed SEED               Random seed (default: 42)"
      echo "  --save-dir DIR            Results directory (default: ./results)"
      echo "  --hidden-dim DIM          Hidden dimension (default: 256)"
      echo "  --lr LR                   Learning rate (default: 0.01)"
      echo ""
      echo "Examples:"
      echo "  $0 --method smp --epochs 200 --device cuda:0"
      echo "  $0 --method gsmp --num-runs 3 --seed 42"
      echo "  $0 --method ump --hidden-dim 128 --device cpu"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Temporal Message Passing on ogbn-mag"
echo "========================================"
echo "Method:       $METHOD"
echo "Epochs:       $EPOCHS"
echo "Device:       $DEVICE"
echo "Num runs:     $NUM_RUNS"
echo "Seed:         $SEED"
echo "Hidden dim:   $HIDDEN_DIM"
echo "LR:           $LR"
echo "Save dir:     $SAVE_DIR"
echo "========================================"
echo ""

# Run experiment
python main.py \
  --method "$METHOD" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --num-runs "$NUM_RUNS" \
  --seed "$SEED" \
  --save-dir "$SAVE_DIR" \
  --hidden-dim "$HIDDEN_DIM" \
  --lr "$LR"

echo ""
echo "========================================"
echo "Experiment complete!"
echo "Results saved to: $SAVE_DIR"
echo "========================================"
