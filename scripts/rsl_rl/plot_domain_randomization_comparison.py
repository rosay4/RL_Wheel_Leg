# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Plot experiment-5 domain-randomization comparison figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot domain-randomization comparison figures for experiment 5.")
parser.add_argument(
    "--main-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    required=True,
    help="Per-trial parameter-shift CSV exported by eval_parameter_shift.py. Repeat for multiple models.",
)
parser.add_argument(
    "--phase-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    default=[],
    help="Optional phase CSV exported by eval_parameter_shift.py with --export-phase-csv.",
)
parser.add_argument(
    "--phase-shift-set",
    type=str,
    default=None,
    help="Shift set to use for the phase portrait. Defaults to the first common shift across phase CSVs.",
)
parser.add_argument(
    "--phase-axis",
    type=str,
    choices=("pitch", "roll"),
    default="pitch",
    help="State pair used for the phase portrait.",
)
parser.add_argument("--output-dir", type=str, default=None, help="Directory for aggregated CSV and plots.")
parser.add_argument("--title-prefix", type=str, default="Domain Randomization", help="Figure title prefix.")
args = parser.parse_args()


BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _resolve_output_dir(first_csv: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = first_csv.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _aggregate_main_csvs() -> tuple[pd.DataFrame, Path]:
    rows = []
    first_csv = None
    for model_label, csv_path_str in args.main_entry:
        csv_path = Path(csv_path_str).expanduser().resolve()
        if first_csv is None:
            first_csv = csv_path
        df = pd.read_csv(csv_path)
        required_cols = {
            "shift_set",
            "survived",
            "survival_time_s",
            "rmse_vy",
            "rmse_wz",
            "mean_abs_pitch_deg",
            "max_abs_pitch_deg",
            "mean_abs_roll_deg",
            "max_abs_roll_deg",
        }
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise KeyError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

        grouped = df.groupby("shift_set", as_index=False).agg(
            survival_rate=("survived", "mean"),
            mean_survival_time_s=("survival_time_s", "mean"),
            mean_rmse_vy=("rmse_vy", "mean"),
            mean_rmse_wz=("rmse_wz", "mean"),
            mean_abs_pitch_deg=("mean_abs_pitch_deg", "mean"),
            max_abs_pitch_deg=("max_abs_pitch_deg", "mean"),
            mean_abs_roll_deg=("mean_abs_roll_deg", "mean"),
            max_abs_roll_deg=("max_abs_roll_deg", "mean"),
            num_trials=("survived", "count"),
        )
        grouped.insert(0, "model_label", model_label)
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True), first_csv


def _plot_grouped_bars(df: pd.DataFrame, metric_name: str, ylabel: str, out_dir: Path):
    pivot = df.pivot(index="shift_set", columns="model_label", values=metric_name)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", ax=ax, color=BAR_COLORS[: len(pivot.columns)], width=0.8)
    ax.set_xlabel("Shift Set")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{args.title_prefix}: {ylabel}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(out_dir / f"domain_randomization_{metric_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _select_phase_shift(phase_frames: list[tuple[str, pd.DataFrame]]) -> str:
    if args.phase_shift_set:
        return args.phase_shift_set

    common_shifts = None
    for _, frame in phase_frames:
        shifts = set(frame["shift_set"].unique())
        common_shifts = shifts if common_shifts is None else common_shifts.intersection(shifts)

    if not common_shifts:
        raise ValueError("No common shift_set found across provided --phase-entry CSVs.")
    return sorted(common_shifts)[0]


def _plot_phase_portrait(out_dir: Path):
    if not args.phase_entry:
        return

    phase_frames: list[tuple[str, pd.DataFrame]] = []
    for model_label, csv_path_str in args.phase_entry:
        csv_path = Path(csv_path_str).expanduser().resolve()
        frame = pd.read_csv(csv_path)
        required_cols = {"shift_set", "pitch_deg", "pitch_rate_rad_s", "roll_deg", "roll_rate_rad_s"}
        missing_cols = required_cols.difference(frame.columns)
        if missing_cols:
            raise KeyError(f"Missing columns in phase CSV {csv_path}: {sorted(missing_cols)}")
        phase_frames.append((model_label, frame))

    selected_shift = _select_phase_shift(phase_frames)
    state_col = "pitch_deg" if args.phase_axis == "pitch" else "roll_deg"
    rate_col = "pitch_rate_rad_s" if args.phase_axis == "pitch" else "roll_rate_rad_s"

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    for idx, (model_label, frame) in enumerate(phase_frames):
        frame = frame[frame["shift_set"] == selected_shift].copy()
        if frame.empty:
            continue
        ax.plot(
            frame[state_col],
            frame[rate_col],
            label=model_label,
            linewidth=1.6,
            alpha=0.9,
            color=BAR_COLORS[idx % len(BAR_COLORS)],
        )

    axis_label = "Pitch" if args.phase_axis == "pitch" else "Roll"
    ax.set_xlabel(f"{axis_label} Angle (deg)")
    ax.set_ylabel(f"{axis_label} Rate (rad/s)")
    ax.set_title(f"{args.title_prefix}: {axis_label} Phase Portrait ({selected_shift})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"domain_randomization_{args.phase_axis}_phase_{selected_shift}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    summary_df, first_csv = _aggregate_main_csvs()
    out_dir = _resolve_output_dir(first_csv)

    summary_csv = out_dir / "domain_randomization_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    _plot_grouped_bars(summary_df, "survival_rate", "Survival Rate", out_dir)
    _plot_grouped_bars(summary_df, "mean_survival_time_s", "Mean Survival Time (s)", out_dir)
    _plot_grouped_bars(summary_df, "mean_rmse_vy", "Mean RMSE Vy", out_dir)
    _plot_grouped_bars(summary_df, "mean_abs_pitch_deg", "Mean |Pitch| (deg)", out_dir)
    _plot_phase_portrait(out_dir)

    print(f"[INFO] Domain-randomization comparison exported to: {out_dir}")


if __name__ == "__main__":
    main()
