import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


L_THIGH = 0.06005
L_CALF = 0.09166


def parse_args():
    parser = argparse.ArgumentParser(description="绘制横滚补偿与腿长变化分析图。")
    parser.add_argument("--timeseries", type=str, required=True, help="*_timeseries.csv 文件路径。")
    parser.add_argument("--output", type=str, default="plots/leg_compensation_analysis.png", help="输出图片路径。")
    return parser.parse_args()


def calculate_leg_length(knee_angle_rad):
    """根据膝关节角度估算髋关节到轮心的直线距离，单位转换为厘米。"""
    length = np.sqrt(L_THIGH**2 + L_CALF**2 + 2 * L_THIGH * L_CALF * np.cos(knee_angle_rad))
    return length * 100.0


def setup_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams["axes.unicode_minus"] = False
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except Exception:
            continue


def main():
    args = parse_args()

    if not os.path.exists(args.timeseries):
        print(f"[ERROR] 未找到文件: {args.timeseries}")
        return

    df = pd.read_csv(args.timeseries)
    setup_plot_style()

    left_knee = "joint_knee_front_L_joint"
    right_knee = "joint_knee_front_R_joint"

    if left_knee not in df.columns or right_knee not in df.columns:
        print(f"[ERROR] CSV 中未找到列: {left_knee} 或 {right_knee}")
        print(f"现有列名: {df.columns.tolist()}")
        return

    df["left_leg_len"] = calculate_leg_length(df[left_knee])
    df["right_leg_len"] = calculate_leg_length(df[right_knee])

    print(f"[INFO] 正在分析数据: {args.timeseries}")
    print(f"[INFO] 使用关节进行腿长换算: {left_knee}, {right_knee}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    time_s = df["time_s"]

    ax1.plot(time_s, df["roll_deg"], color="#D62728", linewidth=2, label="机身横滚角")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.fill_between(time_s, df["roll_deg"], 0, color="#D62728", alpha=0.1)
    ax1.set_ylabel("横滚角度 (deg)")
    ax1.set_title("崎岖路面机身自稳与腿部伸缩补偿分析", fontsize=18, pad=20)
    ax1.set_ylim(-15, 15)
    ax1.legend(loc="upper right")

    ax2.plot(time_s, df["left_leg_len"], color="#1F77B4", linewidth=2.5, label="左前腿长度 (cm)")
    ax2.plot(time_s, df["right_leg_len"], color="#FF7F0E", linewidth=2.5, linestyle="--", label="右前腿长度 (cm)")
    ax2.fill_between(time_s, df["left_leg_len"], df["right_leg_len"], color="gray", alpha=0.2, label="伸缩补偿差值")
    ax2.set_xlabel("时间 (s)", fontsize=14)
    ax2.set_ylabel("估算腿长 (cm)", fontsize=14)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)

    out_file = args.output
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[INFO] 分析图表已保存至: {out_file}")


if __name__ == "__main__":
    main()
