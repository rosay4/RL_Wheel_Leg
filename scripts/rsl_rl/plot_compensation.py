import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Roll compensation and leg variation.")
    parser.add_argument("--timeseries", type=str, required=True, help="Path to the *_timeseries.csv file")
    parser.add_argument("--output", type=str, default="plots/roll_compensation_curve.png", help="Path to save the output png")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.timeseries):
        print(f"[ERROR] File not found: {args.timeseries}")
        return

    df = pd.read_csv(args.timeseries)

    # 论文级画图风格设置
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # 更加健壮的中文支持逻辑
    import matplotlib.font_manager as fm
    # 常见的候选字体名 (覆盖 Windows, Linux, Mac)
    chinese_font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Songti SC', 'STHeiti', 'Arial Unicode MS']
    found_font = None
    
    # 扫描系统中可用的字体名
    installed_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_font_candidates:
        if font in installed_fonts:
            found_font = font
            break
    
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font, 'sans-serif']
        print(f"[INFO] Using font: {found_font}")
    else:
        print("[WARNING] No common Chinese font found. Titles might not display correctly.")
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    plt.rcParams['axes.unicode_minus'] = False
    
    # 寻找名字里带有左右腿特征的关节
    joint_cols = [col for col in df.columns if col.startswith("joint_")]
    
    # 修改：更细致地识别左右腿
    left_leg_col = next((col for col in joint_cols if ("left" in col.lower() or "_l" in col.lower()) and "hip" in col.lower()), None)
    right_leg_col = next((col for col in joint_cols if ("right" in col.lower() or "_r" in col.lower()) and "hip" in col.lower()), None)

    # 如果没找到 hip，退而求其次找任何带有 L/R 的关节
    if not left_leg_col:
        left_leg_col = next((col for col in joint_cols if "left" in col.lower() or "_l" in col.lower()), None)
    if not right_leg_col:
        right_leg_col = next((col for col in joint_cols if "right" in col.lower() or "_r" in col.lower()), None)

    # 如果自动识别还是失败，使用备用索引
    if not left_leg_col or not right_leg_col:
        print("[WARNING] Could not auto-detect left/right joints. Using fallback selection.")
        if len(joint_cols) >= 2:
            left_leg_col, right_leg_col = joint_cols[0], joint_cols[1]
        else:
            print("[ERROR] Not enough joint columns found.")
            return

    print(f"[INFO] Plotting data from: {args.timeseries}")
    print(f"[INFO] Using left joint: {left_leg_col}, right joint: {right_leg_col}")

    # --- 开始绘制双子图 (Top: Roll Angle, Bottom: Leg variation) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time_s = df["time_s"]

    # 1. 绘制机身 Roll 角
    ax1.plot(time_s, df["roll_deg"], color="crimson", linewidth=2.5, label="机身滚转角 (Roll Angle)")
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.fill_between(time_s, df["roll_deg"], 0, color="crimson", alpha=0.1)
    ax1.set_ylabel("滚转角 (度)")
    ax1.set_title("崎岖路面上机身稳定与腿部补偿对比")
    ax1.legend(loc="upper right")

    # 2. 绘制双腿关节伸缩量 (或腿长)
    left_leg_data = df[left_leg_col]
    right_leg_data = df[right_leg_col]

    ax2.plot(time_s, left_leg_data, color="royalblue", linewidth=2.5, label=f"左腿 ({left_leg_col.replace('joint_', '')})")
    ax2.plot(time_s, right_leg_data, color="darkorange", linewidth=2.5, label=f"右腿 ({right_leg_col.replace('joint_', '')})", linestyle="--")
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("关节位置")
    ax2.legend(loc="upper right")

    # 优化排版
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    
    # 确保保存为 PNG
    out_file = args.output
    if not out_file.lower().endswith(".png"):
        out_file = os.path.splitext(out_file)[0] + ".png"
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Beautiful compensation plot saved to: {out_file}")

if __name__ == "__main__":
    main()
