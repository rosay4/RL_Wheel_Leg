import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="绘制论文评测结果图。")
    parser.add_argument("--mode", type=str, choices=["stairs", "rough"], required=True, 
                        help="选择需要绘制的评测模式。")
    parser.add_argument("--files", nargs='+', required=True, 
                        help="*_trials.csv 文件路径列表。")
    parser.add_argument("--labels", nargs='+', required=True, 
                        help="与文件对应的标签列表。")
    parser.add_argument("--out-dir", type=str, default="plots", 
                        help="图片输出目录。")
    return parser.parse_args()

def setup_paper_style():
    """设置论文级别的画图风格"""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def plot_stairs(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 成功率 (Bar Plot)
    success_rates = df.groupby('Method')['success'].mean() * 100
    sns.barplot(x=success_rates.index, y=success_rates.values, ax=axes[0], palette="Blues_d")
    axes[0].set_title("台阶通过成功率")
    axes[0].set_ylabel("成功率 (%)")
    axes[0].set_ylim(0, 105)
    
    # 2. 跨越时间 (Box Plot, 仅统计成功的 Trial)
    df_success = df[df['success'] == 1.0]
    if not df_success.empty:
        sns.boxplot(data=df_success, x='Method', y='time_taken_s', ax=axes[1], palette="Set2", width=0.4)
        axes[1].set_title("成功试验的台阶通过时间")
        axes[1].set_ylabel("时间 (s)")
    else:
        axes[1].set_title("没有成功试验，无法统计通过时间")
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, "stairs_evaluation.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] 台阶评测图片已保存到 {save_path}")

def plot_rough(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 前进距离 (Box Plot)
    sns.boxplot(data=df, x='Method', y='distance_y', ax=axes[0], palette="Set2", width=0.5)
    axes[0].set_title("崎岖路面前进距离")
    axes[0].set_ylabel("距离 (m)")
    
    # 2. 俯仰角稳定性 - Pitch Std (Violin / Box Plot)
    sns.boxplot(data=df, x='Method', y='pitch_std_deg', ax=axes[1], palette="flare", width=0.5)
    axes[1].set_title("俯仰稳定性")
    axes[1].set_ylabel("俯仰角标准差 (deg)")
    
    # 3. 横滚角稳定性 - Roll Std
    sns.boxplot(data=df, x='Method', y='roll_std_deg', ax=axes[2], palette="crest", width=0.5)
    axes[2].set_title("横滚稳定性")
    axes[2].set_ylabel("横滚角标准差 (deg)")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "rough_evaluation.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] 崎岖路面评测图片已保存到 {save_path}")

def main():
    args = parse_args()
    
    if len(args.files) != len(args.labels):
        raise ValueError("文件数量必须与标签数量一致。")
    
    os.makedirs(args.out_dir, exist_ok=True)
    setup_paper_style()
    
    # 读取并合并所有 CSV 数据
    all_data =[]
    for file, label in zip(args.files, args.labels):
        if not os.path.exists(file):
            print(f"[WARNING] 未找到文件: {file}，已跳过。")
            continue
        df = pd.read_csv(file)
        df['Method'] = label
        all_data.append(df)
        
    if not all_data:
        print("[ERROR] 没有读取到有效数据，程序结束。")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 根据模式调用画图
    if args.mode == "stairs":
        plot_stairs(combined_df, args.out_dir)
    elif args.mode == "rough":
        plot_rough(combined_df, args.out_dir)

if __name__ == "__main__":
    main()
