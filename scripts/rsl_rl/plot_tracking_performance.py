# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""绘制速度跟踪与转向性能实验结果。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="绘制速度跟踪与转向性能汇总图。")
parser.add_argument("--input-csv", type=str, required=True, help="eval_tracking_performance.py 导出的 CSV。")
parser.add_argument("--output-dir", type=str, default=None, help="图片输出目录。")
parser.add_argument("--title-prefix", type=str, default="速度跟踪与转向性能", help="图标题前缀。")
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


def main():
    _setup_chinese_font()
    csv_path = Path(args.input_csv).expanduser().resolve()
    df = pd.read_csv(csv_path)
    out_dir = _resolve_output_dir(csv_path)

    _plot_heatmap(df, "rmse_vy", f"{args.title_prefix}：vy 跟踪均方根误差", out_dir / "tracking_rmse_vy.png", "RMSE vy")
    _plot_heatmap(df, "rmse_wz", f"{args.title_prefix}：wz 跟踪均方根误差", out_dir / "tracking_rmse_wz.png", "RMSE wz")
    _plot_heatmap(df, "steady_err_vy", f"{args.title_prefix}：vy 稳态误差", out_dir / "tracking_steady_err_vy.png", "稳态误差 vy")
    _plot_heatmap(df, "steady_err_wz", f"{args.title_prefix}：wz 稳态误差", out_dir / "tracking_steady_err_wz.png", "稳态误差 wz")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(df["command_vy"], df["mean_vy_actual"], s=45, color="#00798c")
    axes[0].plot([df["command_vy"].min(), df["command_vy"].max()], [df["command_vy"].min(), df["command_vy"].max()], "--", color="black", linewidth=1.0)
    axes[0].set_xlabel("速度指令 vy")
    axes[0].set_ylabel("平均实际 vy")
    axes[0].set_title(f"{args.title_prefix}：vy 跟踪散点图")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df["command_wz"], df["mean_wz_actual"], s=45, color="#d1495b")
    axes[1].plot([df["command_wz"].min(), df["command_wz"].max()], [df["command_wz"].min(), df["command_wz"].max()], "--", color="black", linewidth=1.0)
    axes[1].set_xlabel("角速度指令 wz")
    axes[1].set_ylabel("平均实际 wz")
    axes[1].set_title(f"{args.title_prefix}：wz 跟踪散点图")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] 速度跟踪图片已导出到: {out_dir}")


if __name__ == "__main__":
    main()
