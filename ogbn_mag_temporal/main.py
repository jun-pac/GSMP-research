"""
Main entry point for temporal message passing experiments on ogbn-mag.

Usage:
    python main.py --method smp --epochs 200 --device cuda:0
    python main.py --method ump --epochs 200 --device cuda:0
    python main.py --method gsmp --epochs 200 --device cuda:0
"""

import argparse
import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    set_seed, get_device, move_heterodata_to_device,
    configure_logger, print_heterodata_info, print_method_info
)
from data import (
    load_ogbn_mag, compute_timestamps, get_temporal_bounds,
    ensure_undirected_edges
)
from temporal_mp import (
    compute_smp_edge_weights, apply_ump_edge_filter,
    compute_gsmp_edge_weights, add_baseline_edge_weights
)
from model import create_model
from train import train_model

logger = logging.getLogger(__name__)


METHOD_LABELS = {
    "baseline": "HH",
    "smp": "HH+SMP",
    "ump": "HH+UMP",
    "gsmp": "HH+GSMP",
}


def main():
    parser = argparse.ArgumentParser(description="Temporal Message Passing on ogbn-mag")
    
    # Dataset and method
    parser.add_argument("--dataset", type=str, default="ogbn-mag",
                       help="Dataset name (default: ogbn-mag)")
    parser.add_argument("--method", type=str, default="smp",
                       choices=["baseline", "smp", "ump", "gsmp"],
                       help="Temporal message passing method")
    
    # Model architecture
    parser.add_argument("--model", type=str, default="hgamplp_hope",
                       choices=["hgamplp_hope", "hgamLP_hope"],
                       help="Model type (default: hgamplp_hope)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension (default: 256)")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (default: 0.1)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay (default: 0.0)")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience (default: 50)")
    parser.add_argument("--eval-steps", type=int, default=1,
                       help="Evaluate and log every N epochs (default: 1)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of runs with different seeds (default: 1)")
    
    # Temporal
    parser.add_argument("--non-paper-time-strategy", type=str, default="mean",
                       choices=["mean", "min", "max"],
                       help="Strategy for non-paper node timestamps (default: mean)")
    parser.add_argument("--use-reverse-edges", action="store_true",
                       help="Include reverse edges in the graph")
    
    # I/O
    parser.add_argument("--root", type=str, default="./data",
                       help="Data root directory (default: ./data)")
    parser.add_argument("--save-dir", type=str, default="./results",
                       help="Directory to save results (default: ./results)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (default: cpu)")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_file = os.path.join(args.save_dir, f"log_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    configure_logger(log_file)
    
    logger.info("="*70)
    logger.info("TEMPORAL MESSAGE PASSING EXPERIMENTS ON OGBN-MAG")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Device: {device}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Num runs: {args.num_runs}")
    logger.info("="*70)
    
    # Results storage
    all_results = []
    
    for run in range(args.num_runs):
        logger.info(f"\n{'='*70}")
        logger.info(f"RUN {run + 1}/{args.num_runs}")
        logger.info(f"{'='*70}")
        
        # Set seed
        current_seed = args.seed + run
        set_seed(current_seed)
        logger.info(f"Random seed: {current_seed}")
        
        # Load dataset
        logger.info(f"\nLoading dataset from {args.root}...")
        data, split_idx, num_classes, _ = load_ogbn_mag(root=args.root)
        
        # Add reverse edges if requested
        if args.use_reverse_edges:
            logger.info("Adding reverse edges...")
            data = ensure_undirected_edges(data)
        
        # Print dataset info
        print_heterodata_info(data)
        
        # Compute timestamps
        logger.info("\nComputing timestamps...")
        timestamp_dict = compute_timestamps(data, strategy=args.non_paper_time_strategy)
        t_min, t_max = get_temporal_bounds(timestamp_dict)
        
        # Move data to device
        logger.info(f"Moving data to device {device}...")
        data = move_heterodata_to_device(data, device)
        
        # Update timestamp_dict to be on device
        for node_type in timestamp_dict:
            timestamp_dict[node_type] = timestamp_dict[node_type].to(device)
        
        # Apply temporal method
        logger.info(f"\nApplying temporal method: {args.method}")
        if args.method == "baseline":
            data = add_baseline_edge_weights(data)
        elif args.method == "smp":
            data = compute_smp_edge_weights(data, timestamp_dict, t_min, t_max)
        elif args.method == "ump":
            data = apply_ump_edge_filter(data, timestamp_dict)
        elif args.method == "gsmp":
            data = compute_gsmp_edge_weights(data, timestamp_dict)
        
        print_method_info(args.method, data, timestamp_dict)
        
        # Create model
        logger.info("\nCreating model...")
        model = create_model(
            data=data,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            device=device
        )
        
        # Move split indices to device
        split_idx_device = {}
        for key in split_idx:
            split_idx_device[key] = split_idx[key].to(device)
        
        # Train model
        run_label = f"Method: {METHOD_LABELS.get(args.method, args.method)} | Seed: {current_seed}"
        logger.info(f"\nTraining model ({run_label})...")
        best_valid_acc, best_test_acc, best_epoch = train_model(
            model=model,
            data=data,
            split_idx=split_idx_device,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            patience=args.patience,
            eval_steps=args.eval_steps,
            run_label=run_label
        )
        
        # Store results
        run_result = {
            "run": run + 1,
            "seed": current_seed,
            "method": args.method,
            "model": args.model,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "best_valid_acc": best_valid_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "non_paper_time_strategy": args.non_paper_time_strategy,
            "use_reverse_edges": args.use_reverse_edges
        }
        all_results.append(run_result)
        
        logger.info(f"\nRun {run + 1} Results:")
        logger.info(f"  Best Valid Accuracy: {best_valid_acc:.4f}")
        logger.info(f"  Best Test Accuracy:  {best_test_acc:.4f}")
        logger.info(f"  Best Epoch: {best_epoch}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    
    df_results = pd.DataFrame(all_results)
    
    logger.info("\nPerformance Summary:")
    logger.info(f"Method: {args.method}")
    logger.info(f"Mean Valid Accuracy: {df_results['best_valid_acc'].mean():.4f} ± {df_results['best_valid_acc'].std():.4f}")
    logger.info(f"Mean Test Accuracy:  {df_results['best_test_acc'].mean():.4f} ± {df_results['best_test_acc'].std():.4f}")
    
    # Save results to CSV
    csv_path = os.path.join(args.save_dir, "ogbn_mag_hgamLP_hope_temporal_mp.csv")
    
    # Append to existing CSV if it exists
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)
    
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(args.save_dir, f"results_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    results_data = {
        "config": vars(args),
        "results": all_results,
        "summary": {
            "mean_valid_acc": float(df_results['best_valid_acc'].mean()),
            "std_valid_acc": float(df_results['best_valid_acc'].std()),
            "mean_test_acc": float(df_results['best_test_acc'].mean()),
            "std_test_acc": float(df_results['best_test_acc'].std())
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"JSON results saved to {json_path}")
    
    # Print final table
    logger.info("\n" + "="*70)
    logger.info("RESULTS TABLE")
    logger.info("="*70)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
