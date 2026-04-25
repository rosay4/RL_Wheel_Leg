"""Plot TensorBoard training curves for paper section 5.1.2."""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser(description="绘制 PPO 训练过程中的奖励与损失收敛曲线。")
parser.add_argument("--log-dir", type=str, required=True, help="包含 events.out.tfevents 文件的训练日志目录。")
parser.add_argument("--output-dir", type=str, default=None, help="图片与 CSV 输出目录。")
parser.add_argument("--list-tags", action="store_true", default=False, help="只列出 TensorBoard 中已有的 scalar tags。")
parser.add_argument(
    "--tags",
    nargs="+",
    default=None,
    help="需要绘制的 scalar tag。若不填写，脚本会自动选择常见训练指标。",
)
parser.add_argument(
    "--smooth-window",
    type=int,
    default=15,
    help="移动平均窗口大小；设为 1 表示不平滑。",
)
parser.add_argument("--title-prefix", type=str, default="PPO训练过程", help="图标题前缀。")
args = parser.parse_args()


TAG_LABELS = {
    "Train/mean_reward": "平均累积奖励",
    "Train/mean_episode_length": "平均回合长度",
    "Loss/value_function": "价值函数损失",
    "Loss/surrogate": "策略代理损失",
    "Loss/entropy": "策略熵",
}

AUTO_TAG_KEYWORDS = [
    "mean_reward",
    "episode_reward",
    "mean_episode_length",
    "Episode Reward",
    "Rewards/",
    "reward",
    "alive",
    "track",
    "energy",
    "action_rate",
    "joint_vel",
    "wheel_vel",
]


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


def _load_scalars(log_dir: Path):
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator, accumulator.Tags().get("scalars", [])


def _resolve_output_dir(log_dir: Path) -> Path:
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else log_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _select_tags(all_tags: list[str]) -> list[str]:
    if args.tags:
        missing = [tag for tag in args.tags if tag not in all_tags]
        if missing:
            raise KeyError(f"日志中找不到这些 tag: {missing}")
        return args.tags

    selected = []
    for tag in all_tags:
        if any(keyword.lower() in tag.lower() for keyword in AUTO_TAG_KEYWORDS):
            selected.append(tag)
    return selected


def _tag_to_frame(accumulator: EventAccumulator, tag: str) -> pd.DataFrame:
    events = accumulator.Scalars(tag)
    frame = pd.DataFrame(
        {
            "step": [event.step for event in events],
            "value": [event.value for event in events],
            "wall_time": [event.wall_time for event in events],
        }
    )
    frame["tag"] = tag
    if args.smooth_window > 1:
        frame["smoothed"] = frame["value"].rolling(args.smooth_window, min_periods=1).mean()
    else:
        frame["smoothed"] = frame["value"]
    return frame


def _label_for_tag(tag: str) -> str:
    if tag in TAG_LABELS:
        return TAG_LABELS[tag]
    short = tag.split("/")[-1]
    replacements = {
        "mean_reward": "平均累积奖励",
        "mean_episode_length": "平均回合长度",
        "track_lin_vel_xy": "线速度跟踪奖励",
        "track_ang_vel_z": "角速度跟踪奖励",
        "alive": "生存奖励",
        "action_rate_l2": "动作平滑惩罚",
        "joint_vel_l2": "关节速度惩罚",
        "wheel_vel_l2_penalty": "轮速惩罚",
        "flat_orientation": "姿态水平惩罚",
    }
    return replacements.get(short, short)


def _plot_individual_curves(frames: list[pd.DataFrame], out_dir: Path):
    for frame in frames:
        tag = frame["tag"].iloc[0]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(frame["step"], frame["value"], color="#9ecae1", linewidth=1.0, alpha=0.45, label="原始曲线")
        ax.plot(frame["step"], frame["smoothed"], color="#1f77b4", linewidth=2.0, label="平滑曲线")
        ax.set_xlabel("训练迭代步")
        ax.set_ylabel(_label_for_tag(tag))
        ax.set_title(f"{args.title_prefix}：{_label_for_tag(tag)}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        safe_name = tag.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"training_curve_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_combined_rewards(frames: list[pd.DataFrame], out_dir: Path):
    reward_frames = [
        frame
        for frame in frames
        if any(key in frame["tag"].iloc[0].lower() for key in ["reward", "alive", "track", "action_rate", "joint_vel", "wheel_vel"])
    ]
    if not reward_frames:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for frame in reward_frames:
        tag = frame["tag"].iloc[0]
        ax.plot(frame["step"], frame["smoothed"], linewidth=1.8, label=_label_for_tag(tag))
    ax.set_xlabel("训练迭代步")
    ax.set_ylabel("奖励值")
    ax.set_title(f"{args.title_prefix}：奖励项收敛曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_rewards_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    _setup_chinese_font()
    log_dir = Path(args.log_dir).expanduser().resolve()
    accumulator, scalar_tags = _load_scalars(log_dir)

    if args.list_tags:
        print("[INFO] Scalar tags:")
        for tag in scalar_tags:
            print(f"  {tag}")
        return

    selected_tags = _select_tags(scalar_tags)
    if not selected_tags:
        print("[WARN] 没有自动匹配到训练曲线 tag。请先使用 --list-tags 查看可用 tag，再通过 --tags 指定。")
        return

    out_dir = _resolve_output_dir(log_dir)
    frames = [_tag_to_frame(accumulator, tag) for tag in selected_tags]
    all_scalars = pd.concat(frames, ignore_index=True)
    all_scalars.to_csv(out_dir / "training_curves_export.csv", index=False)

    _plot_individual_curves(frames, out_dir)
    _plot_combined_rewards(frames, out_dir)
    print(f"[INFO] 训练曲线已导出到: {out_dir}")


if __name__ == "__main__":
    main()
