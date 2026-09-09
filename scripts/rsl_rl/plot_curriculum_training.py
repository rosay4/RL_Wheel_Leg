"""Plot stitched curriculum-training curves from exported TensorBoard scalar CSV."""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


RUN_ORDER = [
    "cmd_flat_v1.2",
    "robust_v2_finetune",
    "rough_light_v1",
    "rough_light_v1.1",
    "slope_v1",
    "mixed_v1",
    "unstructured_v1",
]

RUN_LABELS = {
    "cmd_flat_v1.2": "平地基础",
    "robust_v2_finetune": "DR微调",
    "rough_light_v1": "轻度崎岖",
    "rough_light_v1.1": "崎岖强化",
    "slope_v1": "连续坡地",
    "mixed_v1": "混合地形",
    "unstructured_v1": "非结构地形",
}

TAG_LABELS = {
    "Train/mean_reward": "平均累积奖励",
    "Train/mean_episode_length": "平均回合长度",
    "Episode_Reward/alive": "生存奖励",
    "Episode_Reward/track_lin_vel_xy": "线速度跟踪奖励",
    "Episode_Reward/track_ang_vel_z": "角速度跟踪奖励",
    "Episode_Reward/action_rate_l2": "动作变化惩罚",
    "Episode_Reward/wheel_vel_l2_penalty": "轮速惩罚",
    "Episode_Reward/joint_vel_l2": "关节速度惩罚",
    "Metrics/base_velocity/error_vel_xy": "线速度跟踪误差",
    "Metrics/base_velocity/error_vel_yaw": "角速度跟踪误差",
}

COLORS = {
    "Train/mean_reward": "#1f77b4",
    "Train/mean_episode_length": "#ff7f0e",
    "Episode_Reward/alive": "#2ca02c",
    "Episode_Reward/track_lin_vel_xy": "#1f77b4",
    "Episode_Reward/track_ang_vel_z": "#9467bd",
    "Episode_Reward/action_rate_l2": "#d62728",
    "Episode_Reward/wheel_vel_l2_penalty": "#8c564b",
    "Episode_Reward/joint_vel_l2": "#e377c2",
    "Metrics/base_velocity/error_vel_xy": "#1f77b4",
    "Metrics/base_velocity/error_vel_yaw": "#ff7f0e",
}


parser = argparse.ArgumentParser(description="Plot paper section 5.1.2 curriculum-training curves.")
parser.add_argument("--csv", type=str, required=True, help="CSV exported by plot_training_curves.py.")
parser.add_argument("--output-dir", type=str, default=None, help="Output directory for PNG figures.")
parser.add_argument("--smooth-window", type=int, default=15, help="Per-stage rolling average window.")
parser.add_argument(
    "--drop-head-ratio",
    type=float,
    default=0.08,
    help="Drop this ratio from the beginning of each run/tag to remove TensorBoard reset transients.",
)
parser.add_argument(
    "--drop-head-points",
    type=int,
    default=5,
    help="Minimum points dropped from the beginning of each run/tag.",
)
parser.add_argument(
    "--trend-tail-ratio",
    type=float,
    default=0.25,
    help="Use the final ratio of each stage to estimate its convergence level.",
)
parser.add_argument(
    "--connect-runs",
    action="store_true",
    default=False,
    help="Connect adjacent curriculum stages with one continuous raw curve.",
)
parser.add_argument(
    "--hide-stage-curves",
    action="store_true",
    default=False,
    help="Only plot stage-level trend lines, without within-stage curves.",
)
parser.add_argument("--dpi", type=int, default=300, help="PNG export DPI.")
args = parser.parse_args()


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
        print("[WARN] 未检测到常见中文字体，图片中文字可能显示为方框。")


def _resolve_output_dir(csv_path: Path) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent / "plots_5_1_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_csv(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {"run", "tag", "step", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if "smoothed" not in data.columns:
        data["smoothed"] = data["value"]
    return data


def _ordered_runs(data: pd.DataFrame) -> list[str]:
    existing = list(dict.fromkeys(data["run"].astype(str).tolist()))
    ordered = [run for run in RUN_ORDER if run in existing]
    ordered.extend([run for run in existing if run not in ordered])
    return ordered


def _drop_stage_head(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("step").copy()
    if len(group) <= args.drop_head_points + 2:
        return group
    drop_count = max(args.drop_head_points, int(len(group) * args.drop_head_ratio))
    drop_count = min(drop_count, len(group) - 2)
    return group.iloc[drop_count:].copy()


def _preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    processed_groups = []
    for _, group in data.groupby(["run", "tag"], sort=False):
        group = _drop_stage_head(group)
        if args.smooth_window > 1:
            group["plot_value"] = group["smoothed"].rolling(args.smooth_window, min_periods=1).mean()
        else:
            group["plot_value"] = group["smoothed"]
        processed_groups.append(group)
    return pd.concat(processed_groups, ignore_index=True)


def _stitch_steps(data: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[float, str]], list[float]]:
    data = data.copy()
    ordered_runs = _ordered_runs(data)
    current_offset = 0.0
    stage_ticks = []
    stage_edges = []

    for run in ordered_runs:
        mask = data["run"] == run
        run_steps = data.loc[mask, "step"]
        min_step = float(run_steps.min())
        max_step = float(run_steps.max())
        run_length = max_step - min_step

        data.loc[mask, "global_step"] = data.loc[mask, "step"] - min_step + current_offset
        stage_ticks.append((current_offset + run_length * 0.5, RUN_LABELS.get(run, run)))
        current_offset += run_length + 1.0
        stage_edges.append(current_offset - 0.5)

    return data, stage_ticks, stage_edges[:-1]


def _series(data: pd.DataFrame, tag: str) -> pd.DataFrame:
    return data[data["tag"] == tag].sort_values(["global_step", "step"]).copy()


def _stage_trend(series: pd.DataFrame) -> pd.DataFrame:
    records = []
    for run, stage_data in series.groupby("run", sort=False):
        stage_data = stage_data.sort_values("global_step")
        if stage_data.empty:
            continue
        tail_count = max(1, int(len(stage_data) * args.trend_tail_ratio))
        tail_data = stage_data.tail(tail_count)
        records.append(
            {
                "run": run,
                "stage": RUN_LABELS.get(run, run),
                "global_step": float(stage_data["global_step"].median()),
                "plot_value": float(tail_data["plot_value"].mean()),
            }
        )
    return pd.DataFrame(records)


def _decorate_stage_axis(ax, stage_ticks: list[tuple[float, str]], stage_edges: list[float]):
    if not stage_ticks:
        return
    ax.set_xticks([pos for pos, _ in stage_ticks])
    ax.set_xticklabels([label for _, label in stage_ticks], rotation=0)
    for edge_x in stage_edges:
        ax.axvline(edge_x, color="#999999", linewidth=0.8, linestyle="--", alpha=0.45)


def _plot_stage_lines(ax, series: pd.DataFrame, tag: str, label: str | None = None, linewidth: float = 2.0):
    if args.hide_stage_curves:
        return

    if args.connect_runs:
        ax.plot(series["global_step"], series["plot_value"], color=COLORS[tag], linewidth=linewidth, label=label)
        return

    for idx, (_, stage_data) in enumerate(series.groupby("run", sort=False)):
        ax.plot(
            stage_data["global_step"],
            stage_data["plot_value"],
            color=COLORS[tag],
            linewidth=linewidth,
            alpha=0.52,
            label=label if idx == 0 else None,
        )


def _plot_stage_trend(ax, series: pd.DataFrame, tag: str, label: str | None = None, linewidth: float = 2.6):
    trend = _stage_trend(series)
    if trend.empty:
        return
    ax.plot(
        trend["global_step"],
        trend["plot_value"],
        color=COLORS[tag],
        linewidth=linewidth,
        linestyle="-",
        marker="o",
        markersize=5,
        label=label or f"{TAG_LABELS[tag]}阶段收敛水平",
    )


def _plot_reward_and_length(data: pd.DataFrame, stage_ticks: list[tuple[float, str]], stage_edges: list[float], out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    specs = [
        ("Train/mean_reward", "平均累积奖励收敛曲线"),
        ("Train/mean_episode_length", "平均回合长度收敛曲线"),
    ]

    for ax, (tag, title) in zip(axes, specs, strict=True):
        series = _series(data, tag)
        if series.empty:
            ax.text(0.5, 0.5, f"缺少数据：{tag}", ha="center", va="center", transform=ax.transAxes)
            continue
        _plot_stage_lines(ax, series, tag, linewidth=1.8)
        _plot_stage_trend(ax, series, tag)
        ax.set_ylabel(TAG_LABELS[tag])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    _decorate_stage_axis(axes[-1], stage_ticks, stage_edges)
    axes[-1].set_xlabel("课程训练阶段")
    fig.suptitle("渐进式课程训练过程中的总体性能收敛", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig_5_1_2_reward_episode_length.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_reward_terms(data: pd.DataFrame, stage_ticks: list[tuple[float, str]], stage_edges: list[float], out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    groups = [
        (
            axes[0],
            ["Episode_Reward/alive", "Episode_Reward/track_lin_vel_xy", "Episode_Reward/track_ang_vel_z"],
            "生存与速度跟踪奖励项",
        ),
        (
            axes[1],
            ["Episode_Reward/action_rate_l2", "Episode_Reward/wheel_vel_l2_penalty", "Episode_Reward/joint_vel_l2"],
            "控制平滑性与能耗相关惩罚项",
        ),
    ]

    for ax, tags, title in groups:
        plotted = False
        for tag in tags:
            series = _series(data, tag)
            if series.empty:
                continue
            _plot_stage_lines(ax, series, tag, label=TAG_LABELS[tag], linewidth=1.5)
            _plot_stage_trend(ax, series, tag, label=f"{TAG_LABELS[tag]}阶段收敛水平", linewidth=2.0)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "缺少对应奖励项数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("奖励/惩罚值")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    _decorate_stage_axis(axes[-1], stage_ticks, stage_edges)
    axes[-1].set_xlabel("课程训练阶段")
    fig.suptitle("渐进式课程训练中的子奖励项变化", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "fig_5_1_2_reward_terms.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_tracking_errors(data: pd.DataFrame, stage_ticks: list[tuple[float, str]], stage_edges: list[float], out_dir: Path):
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    tags = ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw"]

    plotted = False
    for tag in tags:
        series = _series(data, tag)
        if series.empty:
            continue
        _plot_stage_lines(ax, series, tag, label=TAG_LABELS[tag], linewidth=1.5)
        _plot_stage_trend(ax, series, tag, label=f"{TAG_LABELS[tag]}阶段收敛水平", linewidth=2.0)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "缺少速度跟踪误差数据", ha="center", va="center", transform=ax.transAxes)

    _decorate_stage_axis(ax, stage_ticks, stage_edges)
    ax.set_xlabel("课程训练阶段")
    ax.set_ylabel("跟踪误差")
    ax.set_title("渐进式课程训练中的速度跟踪误差变化")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_5_1_2_tracking_errors.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _export_stage_trends(data: pd.DataFrame, out_dir: Path):
    trend_frames = []
    for tag in sorted(data["tag"].unique()):
        trend = _stage_trend(_series(data, tag))
        if trend.empty:
            continue
        trend["tag"] = tag
        trend["label"] = TAG_LABELS.get(tag, tag)
        trend_frames.append(trend)
    if trend_frames:
        pd.concat(trend_frames, ignore_index=True).to_csv(out_dir / "curriculum_training_stage_trends.csv", index=False)


def main():
    _setup_chinese_font()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = _resolve_output_dir(csv_path)
    data = _load_csv(csv_path)
    processed_data = _preprocess_data(data)
    stitched_data, stage_ticks, stage_edges = _stitch_steps(processed_data)
    stitched_data.to_csv(out_dir / "curriculum_training_scalars_stitched.csv", index=False)
    _export_stage_trends(stitched_data, out_dir)

    _plot_reward_and_length(stitched_data, stage_ticks, stage_edges, out_dir)
    _plot_reward_terms(stitched_data, stage_ticks, stage_edges, out_dir)
    _plot_tracking_errors(stitched_data, stage_ticks, stage_edges, out_dir)
    print(f"[INFO] Exported paper 5.1.2 figures to: {out_dir}")


if __name__ == "__main__":
    main()
