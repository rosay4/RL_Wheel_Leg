# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""绘制瞬态冲击实验中的俯仰角与轮毂扭矩响应曲线。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="将瞬态冲击实验 CSV 绘制成论文可用图片。")
parser.add_argument("--input-csv", type=str, required=True, help="eval_impulse_response.py 导出的 CSV 路径。")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="图片输出目录；默认保存在 CSV 同级 plots 文件夹。",
)
parser.add_argument(
    "--title-prefix",
    type=str,
    default="瞬态外力冲击响应",
    help="图标题前缀。",
)
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
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = csv_path.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    _setup_chinese_font()
    csv_path = Path(args.input_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir = _resolve_output_dir(csv_path)

    torque_cols = [col for col in df.columns if col.startswith("torque_")]
    impulse_start = df.index[df["impulse_torque_pitch"] != 0.0]
    impulse_start_t = df.loc[impulse_start[0], "time_s"] if len(impulse_start) > 0 else None

    # Figure 1: pitch angle / rate response
    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes1[0].plot(df["time_s"], df["pitch_deg"], linewidth=2.0, color="#d1495b", label="俯仰角")
    axes1[0].set_ylabel("俯仰角 (deg)")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].plot(df["time_s"], df["pitch_rate_rad_s"], linewidth=2.0, color="#00798c", label="俯仰角速度")
    axes1[1].set_xlabel("时间 (s)")
    axes1[1].set_ylabel("俯仰角速度 (rad/s)")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()

    if impulse_start_t is not None:
        for ax in axes1:
            ax.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    fig1.suptitle(f"{args.title_prefix}：俯仰响应曲线")
    fig1.tight_layout()
    fig1.savefig(out_dir / "pitch_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: wheel hub torque response
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    for col in torque_cols:
        ax2.plot(df["time_s"], df[col], linewidth=1.8, label=col.replace("torque_", ""))

    if impulse_start_t is not None:
        ax2.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("扭矩")
    ax2.set_title(f"{args.title_prefix}：轮毂扭矩响应")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / "hub_torque_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: combined overview for quick paper inspection
    fig3, axes3 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes3[0].plot(df["time_s"], df["pitch_deg"], linewidth=2.0, color="#d1495b")
    axes3[0].set_ylabel("俯仰角 (deg)")
    axes3[0].grid(True, alpha=0.3)

    for col in torque_cols:
        axes3[1].plot(df["time_s"], df[col], linewidth=1.6, label=col.replace("torque_", ""))
    axes3[1].set_xlabel("时间 (s)")
    axes3[1].set_ylabel("扭矩")
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend(ncol=2, fontsize=9)

    if impulse_start_t is not None:
        for ax in axes3:
            ax.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    fig3.suptitle(f"{args.title_prefix}：俯仰角与扭矩总览")
    fig3.tight_layout()
    fig3.savefig(out_dir / "impulse_response_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    print(f"[INFO] 图片已导出到: {out_dir}")


if __name__ == "__main__":
    main()
