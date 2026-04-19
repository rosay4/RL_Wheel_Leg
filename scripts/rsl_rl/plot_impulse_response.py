# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Plot pitch / hub torque response curves exported by eval_impulse_response.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot impulse-response CSV as paper-ready figures.")
parser.add_argument("--input-csv", type=str, required=True, help="Path to the CSV exported by eval_impulse_response.py")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for output figures. Defaults to a sibling folder next to the CSV.",
)
parser.add_argument(
    "--title-prefix",
    type=str,
    default="Impulse Response",
    help="Prefix added to the figure titles.",
)
args = parser.parse_args()


def _resolve_output_dir(csv_path: Path) -> Path:
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = csv_path.resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
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
    axes1[0].plot(df["time_s"], df["pitch_deg"], linewidth=2.0, color="#d1495b", label="Pitch angle")
    axes1[0].set_ylabel("Pitch (deg)")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].plot(df["time_s"], df["pitch_rate_rad_s"], linewidth=2.0, color="#00798c", label="Pitch rate")
    axes1[1].set_xlabel("Time (s)")
    axes1[1].set_ylabel("Pitch rate (rad/s)")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend()

    if impulse_start_t is not None:
        for ax in axes1:
            ax.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    fig1.suptitle(f"{args.title_prefix}: Pitch Response")
    fig1.tight_layout()
    fig1.savefig(out_dir / "pitch_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: wheel hub torque response
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    for col in torque_cols:
        ax2.plot(df["time_s"], df[col], linewidth=1.8, label=col.replace("torque_", ""))

    if impulse_start_t is not None:
        ax2.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque")
    ax2.set_title(f"{args.title_prefix}: Wheel Hub Torque Response")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / "hub_torque_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: combined overview for quick paper inspection
    fig3, axes3 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes3[0].plot(df["time_s"], df["pitch_deg"], linewidth=2.0, color="#d1495b")
    axes3[0].set_ylabel("Pitch (deg)")
    axes3[0].grid(True, alpha=0.3)

    for col in torque_cols:
        axes3[1].plot(df["time_s"], df[col], linewidth=1.6, label=col.replace("torque_", ""))
    axes3[1].set_xlabel("Time (s)")
    axes3[1].set_ylabel("Torque")
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend(ncol=2, fontsize=9)

    if impulse_start_t is not None:
        for ax in axes3:
            ax.axvline(impulse_start_t, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    fig3.suptitle(f"{args.title_prefix}: Pitch and Torque Overview")
    fig3.tight_layout()
    fig3.savefig(out_dir / "impulse_response_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    print(f"[INFO] Figures exported to: {out_dir}")


if __name__ == "__main__":
    main()
