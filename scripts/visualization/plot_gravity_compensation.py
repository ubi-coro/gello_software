#!/usr/bin/env python3
"""
Gravity Compensation Validation Plot (Section 4.3.1)

Demonstrates "weightless" arm behavior by showing joint positions remain
constant when the arm is held in a fixed pose and released.

Usage:
    python plot_gravity_compensation.py <logfile.csv> --output gravity_comp_validation.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scientific plotting style
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'svg',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def main():
    parser = argparse.ArgumentParser(description="Validate gravity compensation (drift test)")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    parser.add_argument("--output", type=str, default="gravity_comp_validation.svg", 
                        help="Output filename")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Time window to analyze [start, end] in seconds")
    parser.add_argument("--joints", type=int, nargs='+', default=[1, 2, 3],
                        help="Joints to plot (default: 1 2 3 = shoulder, elbow, wrist1)")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Convert timestamp to relative time
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Apply time window if specified
    if args.time_window:
        t_start, t_end = args.time_window
        df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
    
    print(f"Analyzing {len(df)} samples over {df['time'].iloc[-1]:.1f} seconds")
    
    # Joint names
    joint_names = {
        0: "Joint 1 (Base)",
        1: "Joint 2 (Shoulder)",
        2: "Joint 3 (Elbow)",
        3: "Joint 4 (Wrist 1)",
        4: "Joint 5 (Wrist 2)",
        5: "Joint 6 (Wrist 3)",
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Plot each joint
    for idx, joint_idx in enumerate(args.joints):
        if joint_idx > 5:
            print(f"Warning: Joint {joint_idx} out of range, skipping")
            continue
        
        position_rad = df[f'q_leader_{joint_idx}'].values
        position_deg = np.rad2deg(position_rad)
        time = df['time'].values
        
        # Calculate drift (deviation from initial position)
        initial_pos = position_deg[0]
        drift = position_deg - initial_pos
        max_drift = np.max(np.abs(drift))
        
        # Plot
        ax.plot(time, position_deg, linewidth=2.0, alpha=0.85, 
                color=colors[idx % len(colors)],
                label=f'{joint_names[joint_idx]} (drift: {max_drift:.2f}°)')
        
        # Add horizontal line at initial position
        ax.axhline(initial_pos, color=colors[idx % len(colors)], 
                   linewidth=0.8, linestyle='--', alpha=0.4)
    
    # Styling
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Joint Position [°]', fontsize=14)
    ax.set_title('Gravity Compensation Validation: Position Hold Test', 
                 fontsize=16, pad=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    textstr = 'Ideal behavior: Horizontal lines (zero drift)\nExcessive drift indicates insufficient compensation'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved preview to {png_path}")
    
    # Print drift statistics
    print("\n=== Drift Statistics (Maximum Absolute Deviation) ===")
    for joint_idx in args.joints:
        if joint_idx <= 5:
            position_rad = df[f'q_leader_{joint_idx}'].values
            position_deg = np.rad2deg(position_rad)
            drift = position_deg - position_deg[0]
            max_drift = np.max(np.abs(drift))
            print(f"{joint_names[joint_idx]:20s}: {max_drift:.3f}° ({np.rad2deg(max_drift):.5f} rad)")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
