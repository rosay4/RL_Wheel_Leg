# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Plot terrain-traversal trial / summary CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot terrain traversal results.")
parser.add_argument("--trials-csv", type=str, required=True, help="Trials CSV from eval_terrain_traversal.py")
parser.add_argument("--summary-csv", type=str, default=None, help="Summary CSV from eval_terrain_traversal.py")
parser.add_argument("--output-dir", type=str, default=None, help="Directory for plots.")
parser.add_argument("--title-prefix", type=str, default="Terrain Traversal", help="Figure title prefix.")
args = parser.parse_args()


def _resolve_output_dir(csv_path: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = csv_path.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    trials_path = Path(args.trials_csv).expanduser().resolve()
    trials_df = pd.read_csv(trials_path)
    summary_df = pd.read_csv(args.summary_csv) if args.summary_csv else None
    out_dir = _resolve_output_dir(trials_path)

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4.2))
    axes1[0].bar(["Success", "Failure"], [trials_df["success"].sum(), len(trials_df) - trials_df["success"].sum()], color=["#00798c", "#d1495b"])
    axes1[0].set_title(f"{args.title_prefix}: Success count")

    axes1[1].hist(trials_df["distance_y_w"], bins=min(10, len(trials_df)), color="#edae49", edgecolor="black", alpha=0.8)
    axes1[1].set_title(f"{args.title_prefix}: Distance")
    axes1[1].set_xlabel("Distance_y_w")

    axes1[2].hist(trials_df["terminated_step"], bins=min(10, len(trials_df)), color="#30638e", edgecolor="black", alpha=0.8)
    axes1[2].set_title(f"{args.title_prefix}: Termination step")
    axes1[2].set_xlabel("Step")
    fig1.tight_layout()
    fig1.savefig(out_dir / "terrain_traversal_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    colors = ["#00798c" if val > 0.5 else "#d1495b" for val in trials_df["success"]]
    ax2.bar(range(len(trials_df)), trials_df["distance_y_w"], color=colors)
    ax2.set_xlabel("Trial index")
    ax2.set_ylabel("Distance_y_w")
    ax2.set_title(f"{args.title_prefix}: Per-trial traversal distance")
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(out_dir / "terrain_traversal_per_trial_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    if summary_df is not None and len(summary_df) > 0:
        summary_df.to_csv(out_dir / "terrain_traversal_summary_copy.csv", index=False)

    print(f"[INFO] Terrain traversal plots exported to: {out_dir}")


if __name__ == "__main__":
    main()
