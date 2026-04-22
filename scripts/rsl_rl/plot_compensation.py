import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# ⚙️ 机器人物理参数 (根据你的模型实际长度修改)
# ==========================================
L_THIGH = 0.06005  # 大腿长度 (m)
L_CALF = 0.09166   # 小腿长度 (m)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Roll compensation and leg length variation.")
    parser.add_argument("--timeseries", type=str, required=True, help="Path to the *_timeseries.csv file")
    parser.add_argument("--output", type=str, default="plots/leg_compensation_analysis.png", help="Path to save the output png")
    return parser.parse_args()

def calculate_leg_length(knee_angle_rad):
    """
    正运动学公式：将膝关节角度换算为髋关节到轮心的直线距离 (cm)
    公式假设：膝关节为0度时腿部适度弯曲，具体根据URDF偏移可能需微调
    """
    # 简单的余弦定律模型
    length = np.sqrt(L_THIGH**2 + L_CALF**2 + 2 * L_THIGH * L_CALF * np.cos(knee_angle_rad))
    return length * 100  # 转为厘米

def main():
    args = parse_args()
    
    if not os.path.exists(args.timeseries):
        print(f"[ERROR] File not found: {args.timeseries}")
        return

    df = pd.read_csv(args.timeseries)

    # --- 字体设置 ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['axes.unicode_minus'] = False
    # 尝试加载中文字体
    import matplotlib.font_manager as fm
    for font in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            break

    # --- 数据提取 ---
    # 根据你的 CSV 列名，提取左前腿和右前腿的膝盖关节来计算腿长
    # 因为在起伏路面，膝关节的角度变化最能体现腿部的伸缩补偿
    left_knee = "joint_knee_front_L_joint"
    right_knee = "joint_knee_front_R_joint"

    if left_knee not in df.columns or right_knee not in df.columns:
        print(f"[ERROR] CSV中未找到列: {left_knee} 或 {right_knee}")
        print(f"现有列名: {df.columns.tolist()}")
        return

    # 计算腿长 (单位: cm)
    df["left_leg_len"] = calculate_leg_length(df[left_knee])
    df["right_leg_len"] = calculate_leg_length(df[right_knee])

    print(f"[INFO] 正在分析数据: {args.timeseries}")
    print(f"[INFO] 使用关节进行腿长换算: {left_knee}, {right_knee}")

    # --- 开始绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    time_s = df["time_s"]

    # 1. 上图：机身 Roll 角稳定性
    ax1.plot(time_s, df["roll_deg"], color="#D62728", linewidth=2, label="机身横滚角 (Roll)")
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    # 填充颜色带体现波动范围
    ax1.fill_between(time_s, df["roll_deg"], 0, color="#D62728", alpha=0.1)
    
    ax1.set_ylabel("滚转角度 (deg)")
    ax1.set_title("崎岖路面机身自稳与腿部伸缩补偿分析", fontsize=18, pad=20)
    ax1.set_ylim(-15, 15) # 根据实际波动调整范围
    ax1.legend(loc="upper right")

    # 2. 下图：左右腿长度补偿曲线
    ax2.plot(time_s, df["left_leg_len"], color="#1F77B4", linewidth=2.5, label="左前腿长度 (cm)")
    ax2.plot(time_s, df["right_leg_len"], color="#FF7F0E", linewidth=2.5, linestyle="--", label="右前腿长度 (cm)")
    
    # 填充两条曲线之间的部分，视觉化体现“长度差”
    ax2.fill_between(time_s, df["left_leg_len"], df["right_leg_len"], color="gray", alpha=0.2, label="伸缩补偿差")

    ax2.set_xlabel("时间 (s)", fontsize=14)
    ax2.set_ylabel("估算腿长 (cm)", fontsize=14)
    ax2.legend(loc="upper right", frameon=True)

    # 优化布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    
    # 保存结果
    out_file = args.output
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[INFO] 分析图表已保存至: {out_file}")

if __name__ == "__main__":
    main()