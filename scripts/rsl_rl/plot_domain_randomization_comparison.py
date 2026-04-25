# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""绘制实验五的域随机化对比图。"""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="绘制实验五的域随机化对比图。")
parser.add_argument(
    "--main-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    required=False,
    help="由 eval_parameter_shift.py 导出的主 trial CSV，可重复填写多个模型。",
)
parser.add_argument(
    "--phase-entry",
    action="append",
    nargs=2,
    metavar=("MODEL_LABEL", "CSV_PATH"),
    default=[],
    help="可选，相位图 CSV；由 eval_parameter_shift.py 配合 --export-phase-csv 导出。",
)
parser.add_argument(
    "--phase-shift-set",
    type=str,
    default=None,
    help="相位图使用的参数偏移场景；默认自动选择各 CSV 共有的第一个场景。",
)
parser.add_argument(
    "--phase-axis",
    type=str,
    choices=("pitch", "roll"),
    default="pitch",
    help="相位图使用俯仰相位或横滚相位。",
)
parser.add_argument("--output-dir", type=str, default=None, help="汇总 CSV 与图片的输出目录。")
parser.add_argument("--title-prefix", type=str, default="域随机化消融实验", help="图标题前缀。")
parser.add_argument(
    "--phase-only",
    action="store_true",
    default=False,
    help="仅绘制相位图，不生成柱状图，也不要求提供 --main-entry。",
)
args = parser.parse_args()


BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SHIFT_LABELS = {
    "nominal": "标称参数",
    "low_friction": "低摩擦",
    "high_friction": "高摩擦",
    "mass_plus": "质量增加",
    "mass_minus": "质量减小",
    "actuator_weak": "驱动削弱",
    "actuator_strong": "驱动增强",
}


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
    else:
        print("[WARN] 未检测到常见中文字体，Matplotlib 可能仍会显示方框。")


def _resolve_output_dir(first_csv: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = first_csv.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_phase_output_dir() -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        if not args.phase_entry:
            raise ValueError("未提供 --phase-entry，无法确定输出目录。")
        first_phase_csv = Path(args.phase_entry[0][1]).expanduser().resolve()
        out_dir = first_phase_csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _aggregate_main_csvs() -> tuple[pd.DataFrame, Path]:
    if not args.main_entry:
        raise ValueError("未提供 --main-entry，无法生成柱状图。")
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
            raise KeyError(f"文件 {csv_path} 缺少字段: {sorted(missing_cols)}")

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
    pivot = pivot.rename(index=lambda idx: SHIFT_LABELS.get(idx, idx))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot.plot(kind="bar", ax=ax, color=BAR_COLORS[: len(pivot.columns)], width=0.8, rot=0)
    ax.set_xlabel("参数偏移场景")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{args.title_prefix}：{ylabel}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="模型")
    ax.tick_params(axis="x", labelrotation=0)
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
        raise ValueError("提供的相位图 CSV 中不存在共同的 shift_set。")
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
            raise KeyError(f"相位图 CSV {csv_path} 缺少字段: {sorted(missing_cols)}")
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

    axis_label = "俯仰" if args.phase_axis == "pitch" else "横滚"
    shift_label = SHIFT_LABELS.get(selected_shift, selected_shift)
    ax.set_xlabel(f"{axis_label}角 (deg)")
    ax.set_ylabel(f"{axis_label}角速度 (rad/s)")
    ax.set_title(f"{args.title_prefix}：{axis_label}相位图（{shift_label}）")
    ax.grid(True, alpha=0.3)
    ax.legend(title="模型")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"domain_randomization_{args.phase_axis}_phase_{selected_shift}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    _setup_chinese_font()
    if args.phase_only:
        out_dir = _resolve_phase_output_dir()
        _plot_phase_portrait(out_dir)
        print(f"[INFO] 相位图已导出到: {out_dir}")
        return

    summary_df, first_csv = _aggregate_main_csvs()
    out_dir = _resolve_output_dir(first_csv)

    summary_csv = out_dir / "domain_randomization_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    _plot_grouped_bars(summary_df, "survival_rate", "存活率", out_dir)
    _plot_grouped_bars(summary_df, "mean_survival_time_s", "平均存活时间 (s)", out_dir)
    _plot_grouped_bars(summary_df, "mean_rmse_vy", "平均纵向速度跟踪误差", out_dir)
    _plot_grouped_bars(summary_df, "mean_abs_pitch_deg", "平均俯仰角绝对值 (deg)", out_dir)
    _plot_phase_portrait(out_dir)

    print(f"[INFO] 域随机化对比图已导出到: {out_dir}")


if __name__ == "__main__":
    main()
