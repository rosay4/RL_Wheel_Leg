# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Plot velocity-tracking performance for one or multiple policies."""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot velocity-tracking and turning performance.")
parser.add_argument(
    "--entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    default=[],
    help="Tracking summary CSV exported by eval_tracking_performance.py. Can be repeated.",
)
parser.add_argument("--input-csv", type=str, default=None, help="Single CSV input, kept for backward compatibility.")
parser.add_argument("--model-label", type=str, default="本文方法", help="Model label used with --input-csv.")
parser.add_argument("--output-dir", type=str, default=None, help="Figure output directory.")
parser.add_argument("--title-prefix", type=str, default="速度跟踪与转向性能", help="Figure title prefix.")
args = parser.parse_args()


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _setup_chinese_font():
    plt.rcParams["axes.unicode_minus"] = False
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    selected_fonts = [font_name for font_name in preferred_fonts if font_name in available_fonts]
    if selected_fonts:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]


def _entries() -> list[tuple[str, Path]]:
    entries = [(label, Path(path).expanduser().resolve()) for label, path in args.entry]
    if args.input_csv:
        entries.append((args.model_label, Path(args.input_csv).expanduser().resolve()))
    if not entries:
        raise ValueError("Please provide at least one --entry MODEL_LABEL CSV_PATH or --input-csv.")
    return entries


def _resolve_output_dir(first_csv: Path) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else first_csv.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_heatmap(df: pd.DataFrame, value_col: str, title: str, output_path: Path, cbar_label: str):
    pivot = df.pivot(index="command_vy", columns="command_wz", values=value_col).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("角速度指令 wz")
    ax.set_ylabel("线速度指令 vy")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_single_model_details(df: pd.DataFrame, out_dir: Path):
    _plot_heatmap(df, "rmse_vy", f"{args.title_prefix}：vy 跟踪均方根误差", out_dir / "tracking_rmse_vy.png", "RMSE vy")
    _plot_heatmap(df, "rmse_wz", f"{args.title_prefix}：wz 跟踪均方根误差", out_dir / "tracking_rmse_wz.png", "RMSE wz")
    _plot_heatmap(df, "steady_err_vy", f"{args.title_prefix}：vy 稳态误差", out_dir / "tracking_steady_err_vy.png", "稳态误差 vy")
    _plot_heatmap(df, "steady_err_wz", f"{args.title_prefix}：wz 稳态误差", out_dir / "tracking_steady_err_wz.png", "稳态误差 wz")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(df["command_vy"], df["mean_vy_actual"], s=45, color="#00798c")
    axes[0].plot(
        [df["command_vy"].min(), df["command_vy"].max()],
        [df["command_vy"].min(), df["command_vy"].max()],
        "--",
        color="black",
        linewidth=1.0,
    )
    axes[0].set_xlabel("速度指令 vy")
    axes[0].set_ylabel("平均实际 vy")
    axes[0].set_title(f"{args.title_prefix}：vy 跟踪散点图")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df["command_wz"], df["mean_wz_actual"], s=45, color="#d1495b")
    axes[1].plot(
        [df["command_wz"].min(), df["command_wz"].max()],
        [df["command_wz"].min(), df["command_wz"].max()],
        "--",
        color="black",
        linewidth=1.0,
    )
    axes[1].set_xlabel("角速度指令 wz")
    axes[1].set_ylabel("平均实际 wz")
    axes[1].set_title(f"{args.title_prefix}：wz 跟踪散点图")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _summarize_entries(entries: list[tuple[str, Path]]) -> pd.DataFrame:
    records = []
    required_cols = {"rmse_vy", "rmse_wz", "steady_err_vy", "steady_err_wz", "peak_err_vy", "peak_err_wz"}
    for model_label, csv_path in entries:
        df = pd.read_csv(csv_path)
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"{csv_path} is missing columns: {sorted(missing)}")
        records.append(
            {
                "model_label": model_label,
                "mean_rmse_vy": df["rmse_vy"].mean(),
                "mean_rmse_wz": df["rmse_wz"].mean(),
                "mean_abs_steady_err_vy": df["steady_err_vy"].abs().mean(),
                "mean_abs_steady_err_wz": df["steady_err_wz"].abs().mean(),
                "mean_peak_err_vy": df["peak_err_vy"].mean(),
                "mean_peak_err_wz": df["peak_err_wz"].mean(),
            }
        )
    return pd.DataFrame(records)


def _plot_comparison(summary_df: pd.DataFrame, out_dir: Path):
    summary_df.to_csv(out_dir / "tracking_performance_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    metrics = [
        ("mean_rmse_vy", "平均线速度 RMSE"),
        ("mean_rmse_wz", "平均角速度 RMSE"),
        ("mean_abs_steady_err_vy", "平均线速度稳态误差绝对值"),
        ("mean_abs_steady_err_wz", "平均角速度稳态误差绝对值"),
    ]
    for ax, (metric, ylabel) in zip(axes.flat, metrics, strict=True):
        ax.bar(summary_df["model_label"], summary_df[metric], color=COLORS[: len(summary_df)])
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("模型")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{args.title_prefix}：奖励权重消融对比")
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    _setup_chinese_font()
    entries = _entries()
    out_dir = _resolve_output_dir(entries[0][1])
    summary_df = _summarize_entries(entries)
    _plot_comparison(summary_df, out_dir)

    if len(entries) == 1:
        _plot_single_model_details(pd.read_csv(entries[0][1]), out_dir)

    print(f"[INFO] Tracking performance figures exported to: {out_dir}")


if __name__ == "__main__":
    main()
