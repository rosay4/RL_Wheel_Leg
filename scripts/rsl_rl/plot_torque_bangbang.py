"""Plot wheel-torque behavior for reward-ablation policies."""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot wheel-torque time series and ablation summary.")
parser.add_argument(
    "--timeseries-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    required=True,
    help="Torque time-series CSV. Can be repeated for multiple models.",
)
parser.add_argument(
    "--summary-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    default=[],
    help="Torque summary CSV. Can be repeated for multiple models.",
)
parser.add_argument("--output-dir", type=str, default=None, help="Figure output directory.")
parser.add_argument("--title-prefix", type=str, default="奖励权重消融实验", help="Figure title prefix.")
parser.add_argument("--max-points", type=int, default=400, help="Maximum points plotted in the time-series figure.")
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


def _resolve_output_dir(first_csv: Path) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else first_csv.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_timeseries(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, (model_label, csv_path_str) in enumerate(args.timeseries_entry):
        csv_path = Path(csv_path_str).expanduser().resolve()
        df = pd.read_csv(csv_path)
        torque_cols = [col for col in df.columns if col.startswith("torque_")]
        if not torque_cols:
            raise KeyError(f"{csv_path} does not contain torque_* columns.")

        df["mean_wheel_torque"] = df[torque_cols].mean(axis=1)
        plot_df = df.iloc[: args.max_points]
        ax.plot(
            plot_df["time_s"],
            plot_df["mean_wheel_torque"],
            linewidth=1.6,
            label=model_label,
            color=COLORS[idx % len(COLORS)],
        )

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("平均轮毂扭矩")
    ax.set_title(f"{args.title_prefix}：轮毂扭矩时序")
    ax.grid(True, alpha=0.3)
    ax.legend(title="模型")
    fig.tight_layout()
    fig.savefig(out_dir / "torque_bangbang_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(out_dir: Path):
    if not args.summary_entry:
        return

    records = []
    for model_label, csv_path_str in args.summary_entry:
        csv_path = Path(csv_path_str).expanduser().resolve()
        df = pd.read_csv(csv_path)
        required_cols = {"torque_diff_rms", "mean_abs_torque"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"{csv_path} is missing columns: {sorted(missing)}")
        records.append(
            {
                "model_label": model_label,
                "torque_diff_rms": float(df.iloc[0]["torque_diff_rms"]),
                "mean_abs_torque": float(df.iloc[0]["mean_abs_torque"]),
            }
        )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(out_dir / "torque_bangbang_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    metrics = [
        ("torque_diff_rms", "扭矩差分 RMS"),
        ("mean_abs_torque", "平均扭矩绝对值"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        ax.bar(summary_df["model_label"], summary_df[metric], color=COLORS[: len(summary_df)])
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("模型")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{args.title_prefix}：轮毂扭矩波动指标")
    fig.tight_layout()
    fig.savefig(out_dir / "torque_bangbang_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    _setup_chinese_font()
    first_csv = Path(args.timeseries_entry[0][1]).expanduser().resolve()
    out_dir = _resolve_output_dir(first_csv)
    _plot_timeseries(out_dir)
    _plot_summary(out_dir)
    print(f"[INFO] Torque figures exported to: {out_dir}")


if __name__ == "__main__":
    main()
