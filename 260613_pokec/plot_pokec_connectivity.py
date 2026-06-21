#!/usr/bin/env python3
"""
Plot Pokec timestamp connectivity from precomputed CSV matrices.

Colab/Linux example:
python plot_pokec_connectivity.py \
  --input-dir pokec_temporal_outputs \
  --out-dir pokec_connectivity_plots

This script does not read the raw Pokec graph. It only reads small CSV matrices
created by analyze_pokec_temporal.py, so it is CPU/memory cheap and GPU-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_year_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def read_split_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)


def row_normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total == 0:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / total


def save_all_year_overlay_plot(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    legend_title: str,
    yscale: str,
) -> None:
    plt = import_matplotlib()
    years = matrix.columns.astype(int).to_numpy()
    source_years = matrix.index.astype(int).to_numpy()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(source_years)))

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for color, source_year in zip(colors, source_years):
        values = matrix.loc[source_year].to_numpy(dtype=np.float64)
        ax.plot(
            years,
            values,
            marker="o",
            markersize=4.2,
            linewidth=1.9,
            color=color,
            label=str(source_year),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if yscale == "linear" else f"{ylabel}, log scale")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.set_yscale(yscale)
    ax.legend(
        title=legend_title,
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.savefig(path, dpi=240)
    plt.close(fig)
    log(f"[plot] wrote {path}")


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    log_counts: bool = False,
    annotate: bool = False,
    probability: bool = False,
) -> None:
    plt = import_matplotlib()
    data = matrix.to_numpy(dtype=np.float64)
    shown = np.log1p(data) if log_counts else data
    n_rows, n_cols = shown.shape

    fig_width = max(7.0, 0.48 * n_cols)
    fig_height = max(5.5, 0.42 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(shown, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([str(x) for x in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(x) for x in matrix.index])
    colorbar = fig.colorbar(image, ax=ax)
    if log_counts:
        colorbar.set_label("log(1 + edge count)")
    elif probability:
        colorbar.set_label("row-normalized probability")
    else:
        colorbar.set_label("edge count")

    if annotate and n_rows <= 6 and n_cols <= 6:
        max_value = np.max(shown) if shown.size else 0.0
        threshold = max_value / 2.0
        for row in range(n_rows):
            for col in range(n_cols):
                value = data[row, col]
                text = f"{value:.2f}" if probability else f"{int(value):,}"
                color = "white" if shown[row, col] > threshold else "black"
                ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    fig.savefig(path, dpi=240)
    plt.close(fig)
    log(f"[plot] wrote {path}")


def save_node_count_plot(input_dir: Path, out_dir: Path) -> None:
    path = input_dir / "pokec_node_counts_by_registration_year.csv"
    if not path.exists():
        log(f"[plot] skip node counts; missing {path}")
        return

    plt = import_matplotlib()
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    ax.bar(df["registration_year"], df["num_nodes"], color="#4c78a8")
    ax.set_title("Pokec users by registration year")
    ax.set_xlabel("Registration year")
    ax.set_ylabel("Number of users")
    ax.set_xticks(df["registration_year"])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    output = out_dir / "pokec_node_counts_by_registration_year.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    log(f"[plot] wrote {output}")


def save_target_year_summary_and_plots(
    directed_counts: pd.DataFrame,
    undirected_counts: pd.DataFrame,
    out_dir: Path,
    target_years: Sequence[int],
) -> None:
    plt = import_matplotlib()
    years = directed_counts.columns.astype(int).to_numpy()

    for target_year in target_years:
        if target_year not in directed_counts.index:
            log(f"[target] skip {target_year}; year not present in directed matrix")
            continue
        if target_year not in undirected_counts.index:
            log(f"[target] skip {target_year}; year not present in undirected matrix")
            continue

        directed_out = directed_counts.loc[target_year].to_numpy(dtype=np.int64)
        directed_in = directed_counts[target_year].to_numpy(dtype=np.int64)
        undirected = undirected_counts.loc[target_year].to_numpy(dtype=np.int64)

        summary = pd.DataFrame(
            {
                "target_year": target_year,
                "neighbor_year": years,
                "directed_out_count": directed_out,
                "directed_in_count": directed_in,
                "undirected_count": undirected,
                "directed_out_probability": row_normalize(directed_out),
                "directed_in_probability": row_normalize(directed_in),
                "undirected_probability": row_normalize(undirected),
            }
        )
        summary_path = out_dir / f"pokec_connectivity_year_{target_year}_summary.csv"
        summary.to_csv(summary_path, index=False)
        log(f"[plot] wrote {summary_path}")

        for log_y in (False, True):
            fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
            ax.plot(years, directed_out, marker="o", linewidth=2.0, label="directed out")
            ax.plot(years, directed_in, marker="s", linewidth=2.0, label="directed in")
            ax.plot(years, undirected, marker="^", linewidth=2.0, label="undirected")
            ax.set_title(f"Pokec connectivity for registration year {target_year}")
            ax.set_xlabel("Neighbor registration year")
            ax.set_ylabel("Edge count")
            ax.set_xticks(years)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            suffix = "log" if log_y else "linear"
            if log_y:
                ax.set_yscale("log")
                ax.set_ylabel("Edge count, log scale")
            output = out_dir / f"pokec_connectivity_year_{target_year}_{suffix}.png"
            fig.savefig(output, dpi=240)
            plt.close(fig)
            log(f"[plot] wrote {output}")


def save_all_heatmaps(input_dir: Path, out_dir: Path) -> None:
    directed_counts = read_year_matrix(input_dir / "pokec_edge_counts_by_year_directed.csv")
    undirected_counts = read_year_matrix(input_dir / "pokec_edge_counts_by_year_undirected.csv")
    directed_probs = read_year_matrix(input_dir / "pokec_edge_probs_by_year_directed.csv")
    undirected_probs = read_year_matrix(input_dir / "pokec_edge_probs_by_year_undirected.csv")

    save_heatmap(
        directed_counts,
        "Directed timestamp connectivity, edge counts",
        out_dir / "pokec_year_connectivity_directed_counts_log.png",
        xlabel="Destination registration year",
        ylabel="Source registration year",
        log_counts=True,
    )
    save_heatmap(
        undirected_counts,
        "Undirected timestamp connectivity, edge counts",
        out_dir / "pokec_year_connectivity_undirected_counts_log.png",
        xlabel="Neighbor registration year",
        ylabel="Registration year",
        log_counts=True,
    )
    save_heatmap(
        directed_probs,
        "Directed timestamp connectivity, row-normalized",
        out_dir / "pokec_year_connectivity_directed_probs.png",
        xlabel="Destination registration year",
        ylabel="Source registration year",
        probability=True,
    )
    save_heatmap(
        undirected_probs,
        "Undirected timestamp connectivity, row-normalized",
        out_dir / "pokec_year_connectivity_undirected_probs.png",
        xlabel="Neighbor registration year",
        ylabel="Registration year",
        probability=True,
    )

    directed_split_counts = read_split_matrix(
        input_dir / "pokec_edge_counts_by_chronological_split_directed.csv"
    )
    undirected_split_counts = read_split_matrix(
        input_dir / "pokec_edge_counts_by_chronological_split_undirected.csv"
    )
    directed_split_probs = read_split_matrix(
        input_dir / "pokec_edge_probs_by_chronological_split_directed.csv"
    )
    undirected_split_probs = read_split_matrix(
        input_dir / "pokec_edge_probs_by_chronological_split_undirected.csv"
    )

    save_heatmap(
        directed_split_counts,
        "Directed split connectivity, edge counts",
        out_dir / "pokec_split_connectivity_directed_counts_log.png",
        xlabel="Destination split",
        ylabel="Source split",
        log_counts=True,
        annotate=True,
    )
    save_heatmap(
        undirected_split_counts,
        "Undirected split connectivity, edge counts",
        out_dir / "pokec_split_connectivity_undirected_counts_log.png",
        xlabel="Neighbor split",
        ylabel="Split",
        log_counts=True,
        annotate=True,
    )
    save_heatmap(
        directed_split_probs,
        "Directed split connectivity, row-normalized",
        out_dir / "pokec_split_connectivity_directed_probs.png",
        xlabel="Destination split",
        ylabel="Source split",
        probability=True,
        annotate=True,
    )
    save_heatmap(
        undirected_split_probs,
        "Undirected split connectivity, row-normalized",
        out_dir / "pokec_split_connectivity_undirected_probs.png",
        xlabel="Neighbor split",
        ylabel="Split",
        probability=True,
        annotate=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all-year overlaid directed and undirected Pokec timestamp "
            "connectivity from precomputed CSV matrices."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("pokec_temporal_outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("pokec_connectivity_plots"))
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale for the overlaid edge-count plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    directed_counts = read_year_matrix(args.input_dir / "pokec_edge_counts_by_year_directed.csv")
    undirected_counts = read_year_matrix(args.input_dir / "pokec_edge_counts_by_year_undirected.csv")

    save_all_year_overlay_plot(
        directed_counts,
        "Directed Pokec timestamp connectivity by source year",
        args.out_dir / "pokec_all_years_connectivity_directed.png",
        xlabel="Destination registration year",
        ylabel="Directed edge count",
        legend_title="Source year",
        yscale=args.yscale,
    )
    save_all_year_overlay_plot(
        undirected_counts,
        "Undirected Pokec timestamp connectivity by year",
        args.out_dir / "pokec_all_years_connectivity_undirected.png",
        xlabel="Neighbor registration year",
        ylabel="Undirected edge count",
        legend_title="Year",
        yscale=args.yscale,
    )
    log("[done] connectivity plots complete")


if __name__ == "__main__":
    main()
