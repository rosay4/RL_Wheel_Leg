# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate and plot domain-randomization comparison results from multiple experiment summaries."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot domain-randomization comparison from summary CSV files.")
parser.add_argument(
    "--entry",
    action="append",
    nargs=3,
    metavar=("MODEL_LABEL", "METRIC_NAME", "CSV_PATH"),
    help="Comparison entry: model label, metric key, and CSV path. Repeat for multiple entries.",
    required=True,
)
parser.add_argument("--output-dir", type=str, default=None, help="Directory for aggregated CSV and plots.")
parser.add_argument("--title-prefix", type=str, default="Domain Randomization", help="Figure title prefix.")
args = parser.parse_args()


def _resolve_output_dir(first_csv: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = first_csv.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    records = []
    first_csv = None
    for model_label, metric_name, csv_path_str in args.entry:
        csv_path = Path(csv_path_str).expanduser().resolve()
        if first_csv is None:
            first_csv = csv_path
        df = pd.read_csv(csv_path)
        if metric_name not in df.columns:
            raise KeyError(f"Metric '{metric_name}' not found in {csv_path}")
        value = float(df.iloc[0][metric_name])
        records.append({"model_label": model_label, "metric_name": metric_name, "value": value, "csv_path": str(csv_path)})

    out_dir = _resolve_output_dir(first_csv)
    summary_df = pd.DataFrame(records)
    summary_csv = out_dir / "domain_randomization_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    for metric_name, metric_df in summary_df.groupby("metric_name"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(metric_df["model_label"], metric_df["value"], color=["#00798c", "#edae49", "#d1495b"][: len(metric_df)])
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{args.title_prefix}: {metric_name}")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / f"domain_randomization_{metric_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Domain-randomization comparison exported to: {out_dir}")


if __name__ == "__main__":
    main()
