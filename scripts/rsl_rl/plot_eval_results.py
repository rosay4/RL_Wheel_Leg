import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation results for paper.")
    parser.add_argument("--mode", type=str, choices=["stairs", "rough"], required=True, 
                        help="Which evaluation mode's data to plot.")
    parser.add_argument("--files", nargs='+', required=True, 
                        help="List of path to the *_trials.csv files.")
    parser.add_argument("--labels", nargs='+', required=True, 
                        help="List of labels corresponding to the files (e.g., Ours Baseline).")
    parser.add_argument("--out-dir", type=str, default="plots", 
                        help="Directory to save the output plots.")
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
    axes[0].set_title("Stair Traversal Success Rate")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 105)
    
    # 2. 跨越时间 (Box Plot, 仅统计成功的 Trial)
    df_success = df[df['success'] == 1.0]
    if not df_success.empty:
        sns.boxplot(data=df_success, x='Method', y='time_taken_s', ax=axes[1], palette="Set2", width=0.4)
        axes[1].set_title("Time to Cross Stairs (Successful Trials)")
        axes[1].set_ylabel("Time (s)")
    else:
        axes[1].set_title("No Successful Trials to Plot Time")
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, "stairs_evaluation.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Stairs plot saved to {save_path}")

def plot_rough(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 前进距离 (Box Plot)
    sns.boxplot(data=df, x='Method', y='distance_y', ax=axes[0], palette="Set2", width=0.5)
    axes[0].set_title("Forward Distance on Rough Terrain")
    axes[0].set_ylabel("Distance (m)")
    
    # 2. 俯仰角稳定性 - Pitch Std (Violin / Box Plot)
    sns.boxplot(data=df, x='Method', y='pitch_std_deg', ax=axes[1], palette="flare", width=0.5)
    axes[1].set_title("Pitch Stability")
    axes[1].set_ylabel("Pitch Std Dev (deg)")
    
    # 3. 横滚角稳定性 - Roll Std
    sns.boxplot(data=df, x='Method', y='roll_std_deg', ax=axes[2], palette="crest", width=0.5)
    axes[2].set_title("Roll Stability")
    axes[2].set_ylabel("Roll Std Dev (deg)")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "rough_evaluation.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"[INFO] Rough terrain plot saved to {save_path}")

def main():
    args = parse_args()
    
    if len(args.files) != len(args.labels):
        raise ValueError("The number of files must match the number of labels.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    setup_paper_style()
    
    # 读取并合并所有 CSV 数据
    all_data =[]
    for file, label in zip(args.files, args.labels):
        if not os.path.exists(file):
            print(f"[WARNING] File not found: {file}. Skipping...")
            continue
        df = pd.read_csv(file)
        df['Method'] = label
        all_data.append(df)
        
    if not all_data:
        print("[ERROR] No valid data loaded. Exiting.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 根据模式调用画图
    if args.mode == "stairs":
        plot_stairs(combined_df, args.out_dir)
    elif args.mode == "rough":
        plot_rough(combined_df, args.out_dir)

if __name__ == "__main__":
    main()