"""Export and plot TensorBoard scalar curves for paper section 5.1.2."""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/alive",
    "Episode_Reward/track_lin_vel_xy",
    "Episode_Reward/track_ang_vel_z",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/wheel_vel_l2_penalty",
    "Episode_Reward/joint_vel_l2",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
]

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
    "Loss/value_function": "价值函数损失",
    "Loss/surrogate": "策略代理损失",
    "Loss/entropy": "策略熵",
}


parser = argparse.ArgumentParser(description="Export TensorBoard scalars from one or multiple RSL-RL training runs.")
parser.add_argument("--log-dir", type=str, default=None, help="Single TensorBoard run directory.")
parser.add_argument(
    "--logdir-spec",
    type=str,
    default=None,
    help="Comma-separated run specs, for example: name1:/path/run1,name2:/path/run2.",
)
parser.add_argument("--output-dir", type=str, default=None, help="Directory for CSV and PNG outputs.")
parser.add_argument("--output-csv", type=str, default="curriculum_training_scalars.csv", help="CSV file name or path.")
parser.add_argument("--list-tags", action="store_true", default=False, help="List scalar tags and exit.")
parser.add_argument("--tags", nargs="+", default=None, help="Scalar tags to export. Defaults to paper 5.1.2 tags.")
parser.add_argument("--smooth-window", type=int, default=25, help="Moving average window. Use 1 to disable smoothing.")
parser.add_argument("--plot", action="store_true", default=False, help="Also export simple PNG plots.")
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


def _parse_logdir_spec() -> list[tuple[str, Path]]:
    if args.logdir_spec:
        runs = []
        for item in args.logdir_spec.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid --logdir-spec item: {item}")
            run_name, run_path = item.split(":", 1)
            runs.append((run_name.strip(), Path(run_path.strip().strip('"')).expanduser().resolve()))
        return runs

    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
        return [(log_dir.name, log_dir)]

    raise ValueError("Please provide either --log-dir or --logdir-spec.")


def _load_accumulator(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def _resolve_output_dir(runs: list[tuple[str, Path]]) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    elif len(runs) == 1:
        out_dir = runs[0][1] / "analysis" / "training_curves"
    else:
        out_dir = Path.cwd() / "plots" / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _output_csv_path(out_dir: Path) -> Path:
    csv_path = Path(args.output_csv).expanduser()
    if csv_path.is_absolute():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return csv_path
    return out_dir / csv_path


def _selected_tags() -> list[str]:
    return args.tags if args.tags else DEFAULT_TAGS


def _tag_to_frame(accumulator: EventAccumulator, run_name: str, tag: str) -> pd.DataFrame:
    events = accumulator.Scalars(tag)
    frame = pd.DataFrame(
        {
            "run": run_name,
            "tag": tag,
            "step": [event.step for event in events],
            "value": [event.value for event in events],
            "wall_time": [event.wall_time for event in events],
        }
    )
    if args.smooth_window > 1:
        frame["smoothed"] = frame["value"].rolling(args.smooth_window, min_periods=1).mean()
    else:
        frame["smoothed"] = frame["value"]
    return frame


def _export_scalars(runs: list[tuple[str, Path]], tags: list[str]) -> pd.DataFrame:
    frames = []
    for run_name, log_dir in runs:
        accumulator = _load_accumulator(log_dir)
        available_tags = set(accumulator.Tags().get("scalars", []))

        if args.list_tags:
            print(f"[INFO] Scalar tags for {run_name}:")
            for tag in sorted(available_tags):
                print(f"  {tag}")
            continue

        missing = [tag for tag in tags if tag not in available_tags]
        if missing:
            print(f"[WARN] {run_name} missing tags: {missing}")

        for tag in tags:
            if tag in available_tags:
                frames.append(_tag_to_frame(accumulator, run_name, tag))

    if args.list_tags:
        return pd.DataFrame()
    if not frames:
        raise RuntimeError("No scalar data was exported. Check --logdir-spec and --tags.")
    return pd.concat(frames, ignore_index=True)


def _label_for_tag(tag: str) -> str:
    return TAG_LABELS.get(tag, tag.split("/")[-1])


def _plot_tag_groups(data: pd.DataFrame, out_dir: Path):
    for tag, tag_data in data.groupby("tag"):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for run_name, run_data in tag_data.groupby("run"):
            ax.plot(run_data["step"], run_data["smoothed"], linewidth=1.8, label=run_name)
        ax.set_xlabel("训练迭代步")
        ax.set_ylabel(_label_for_tag(tag))
        ax.set_title(f"课程训练过程：{_label_for_tag(tag)}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"training_curve_{safe_tag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    _setup_chinese_font()
    runs = _parse_logdir_spec()
    out_dir = _resolve_output_dir(runs)
    data = _export_scalars(runs, _selected_tags())

    if args.list_tags:
        return

    csv_path = _output_csv_path(out_dir)
    data.to_csv(csv_path, index=False)
    if args.plot:
        _plot_tag_groups(data, out_dir)
    print(f"[INFO] Exported curriculum training scalars to: {csv_path}")


if __name__ == "__main__":
    main()
