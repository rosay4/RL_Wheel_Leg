# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""绘制坡道性能实验结果。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="绘制坡道性能实验 CSV 图表。")
parser.add_argument("--trajectory-csv", type=str, required=True, help="eval_slope_performance.py 导出的轨迹 CSV。")
parser.add_argument("--snapshots-csv", type=str, default=None, help="eval_slope_performance.py 导出的快照 CSV。")
parser.add_argument("--summary-csv", type=str, default=None, help="eval_slope_performance.py 导出的汇总 CSV。")
parser.add_argument("--output-dir", type=str, default=None, help="图片输出目录。")
parser.add_argument("--title-prefix", type=str, default="坡道通过性能", help="图标题前缀。")
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
    traj_path = Path(args.trajectory_csv).expanduser().resolve()
    traj_df = pd.read_csv(traj_path)
    snap_df = pd.read_csv(args.snapshots_csv) if args.snapshots_csv else None
    summary_df = pd.read_csv(args.summary_csv) if args.summary_csv else None
    out_dir = _resolve_output_dir(traj_path)

    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes1[0].plot(traj_df["time_s"], traj_df["pitch_deg"], linewidth=2.0, color="#d1495b", label="俯仰角")
    axes1[0].plot(traj_df["time_s"], traj_df["roll_deg"], linewidth=1.8, color="#00798c", label="横滚角")
    axes1[0].set_ylabel("姿态角 (deg)")
    axes1[0].set_title(f"{args.title_prefix}：机身姿态变化")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].plot(traj_df["time_s"], traj_df["com_height_w"], linewidth=2.0, color="#edae49", label="质心高度")
    axes1[1].plot(traj_df["time_s"], traj_df["root_z_w"], linewidth=1.8, color="#30638e", label="机身高度")
    axes1[1].set_xlabel("时间 (s)")
    axes1[1].set_ylabel("高度 (m)")
    axes1[1].set_title(f"{args.title_prefix}：高度轨迹")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "slope_attitude_height.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    axes2[0].plot(traj_df["root_y_w"], traj_df["root_z_w"], linewidth=2.0, color="#00798c")
    axes2[0].set_xlabel("世界坐标 y")
    axes2[0].set_ylabel("机身高度 z")
    axes2[0].set_title(f"{args.title_prefix}：机身轨迹")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(traj_df["time_s"], traj_df["lin_vel_y_b"], linewidth=2.0, color="#d1495b", label="vy")
    axes2[1].plot(traj_df["time_s"], traj_df["ang_vel_z_b"], linewidth=1.8, color="#edae49", label="wz")
    axes2[1].set_xlabel("时间 (s)")
    axes2[1].set_ylabel("速度")
    axes2[1].set_title(f"{args.title_prefix}：速度响应")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "slope_trajectory_velocity.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    if snap_df is not None and len(snap_df) > 0:
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        ax3.plot(traj_df["time_s"], traj_df["pitch_deg"], linewidth=1.5, color="#cccccc", label="俯仰角轨迹")
        ax3.scatter(snap_df["time_s"], snap_df["pitch_deg"], s=55, color="#d1495b", zorder=3, label="快照时刻")
        for _, row in snap_df.iterrows():
            ax3.annotate(f"{int(row['step'])}", (row["time_s"], row["pitch_deg"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax3.set_xlabel("时间 (s)")
        ax3.set_ylabel("俯仰角 (deg)")
        ax3.set_title(f"{args.title_prefix}：快照时刻标记")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(out_dir / "slope_snapshots_pitch.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    if summary_df is not None and len(summary_df) > 0:
        summary_df.to_csv(out_dir / "slope_summary_copy.csv", index=False)

    print(f"[INFO] 坡道实验图片已导出到: {out_dir}")


if __name__ == "__main__":
    main()
