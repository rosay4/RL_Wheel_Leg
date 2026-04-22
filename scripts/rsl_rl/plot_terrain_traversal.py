# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""绘制地形通过能力实验结果。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="绘制离散台阶与崎岖路面通过能力结果。")
parser.add_argument("--trials-csv", type=str, required=True, help="eval_terrain_traversal.py 导出的 trials CSV。")
parser.add_argument("--summary-csv", type=str, default=None, help="eval_terrain_traversal.py 导出的 summary CSV。")
parser.add_argument("--output-dir", type=str, default=None, help="图片输出目录。")
parser.add_argument("--title-prefix", type=str, default="地形通过能力", help="图标题前缀。")
args = parser.parse_args()


def _setup_chinese_font():
    plt.rcParams["axes.unicode_minus"] = False
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except Exception:
            continue


def _resolve_output_dir(csv_path: Path) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = csv_path.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    _setup_chinese_font()
    trials_path = Path(args.trials_csv).expanduser().resolve()
    trials_df = pd.read_csv(trials_path)
    summary_df = pd.read_csv(args.summary_csv) if args.summary_csv else None
    out_dir = _resolve_output_dir(trials_path)

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4.2))
    axes1[0].bar(["成功", "失败"], [trials_df["success"].sum(), len(trials_df) - trials_df["success"].sum()], color=["#00798c", "#d1495b"])
    axes1[0].set_title(f"{args.title_prefix}：成功与失败次数")

    axes1[1].hist(trials_df["distance_y_w"], bins=min(10, len(trials_df)), color="#edae49", edgecolor="black", alpha=0.8)
    axes1[1].set_title(f"{args.title_prefix}：前进距离分布")
    axes1[1].set_xlabel("世界坐标前进距离 y (m)")

    axes1[2].hist(trials_df["terminated_step"], bins=min(10, len(trials_df)), color="#30638e", edgecolor="black", alpha=0.8)
    axes1[2].set_title(f"{args.title_prefix}：终止步数分布")
    axes1[2].set_xlabel("终止步数")
    fig1.tight_layout()
    fig1.savefig(out_dir / "terrain_traversal_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    colors = ["#00798c" if val > 0.5 else "#d1495b" for val in trials_df["success"]]
    ax2.bar(range(len(trials_df)), trials_df["distance_y_w"], color=colors)
    ax2.set_xlabel("试验编号")
    ax2.set_ylabel("世界坐标前进距离 y (m)")
    ax2.set_title(f"{args.title_prefix}：各次试验前进距离")
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(out_dir / "terrain_traversal_per_trial_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    if summary_df is not None and len(summary_df) > 0:
        summary_df.to_csv(out_dir / "terrain_traversal_summary_copy.csv", index=False)

    print(f"[INFO] 地形通过能力图片已导出到: {out_dir}")


if __name__ == "__main__":
    main()
