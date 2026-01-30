#!/usr/bin/env python3
"""
Dynamics Model Validation Plot (Section 4.2.1)

Plots residuals (measured - predicted torque) for all joints during slow motion.
This validates that the URDF inertial parameters are correct.

Usage:
    python plot_dynamics_validation.py <logfile.csv> --output dynamics_validation.svg
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
    parser = argparse.ArgumentParser(description="Plot dynamics model validation")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    parser.add_argument("--output", type=str, default="dynamics_validation.svg", help="Output filename")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Time window to plot [start, end] in seconds")
    parser.add_argument("--filter-mode", type=str, default="gravity_comp",
                        help="Only use data from this control mode")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Filter by control mode
    if args.filter_mode:
        df = df[df['control_mode'] == args.filter_mode]
    
    if len(df) == 0:
        print(f"Error: No data found for mode '{args.filter_mode}'")
        return 1
    
    # Convert timestamp to relative time
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Apply time window if specified
    if args.time_window:
        t_start, t_end = args.time_window
        df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
    
    print(f"Analyzing {len(df)} samples over {df['time'].iloc[-1]:.1f} seconds")
    
    # Detect number of joints
    num_joints = 6
    
    # Create subplots (2 rows x 3 columns for 6 joints)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    joint_names = ["Joint 1 (Base)", "Joint 2 (Shoulder)", "Joint 3 (Elbow)",
                   "Joint 4 (Wrist 1)", "Joint 5 (Wrist 2)", "Joint 6 (Wrist 3)"]
    
    # Calculate residuals for each joint
    for i in range(num_joints):
        ax = axes[i]
        
        # Calculate residual = measured - modeled
        # For quasi-static motion: tau_measured ≈ tau_gravity
        # Residual shows unmodeled effects (friction, errors in inertial params)
        tau_measured = df[f'tau_total_{i}'].values
        tau_modeled = df[f'tau_gravity_{i}'].values
        residual = tau_measured - tau_modeled
        time = df['time'].values
        
        # Calculate statistics
        mean_residual = np.mean(residual)
        std_residual = np.std(residual)
        rmse = np.sqrt(np.mean(residual**2))
        
        # Plot residual
        ax.plot(time, residual, linewidth=0.8, color='#2c3e50', alpha=0.7)
        
        # Add zero line
        ax.axhline(0, color='r', linewidth=1.2, linestyle='--', alpha=0.6, label='Zero (Perfect Model)')
        
        # Add mean ± std band
        ax.axhspan(mean_residual - std_residual, mean_residual + std_residual,
                   alpha=0.2, color='#3498db', label=f'Mean ± σ')
        ax.axhline(mean_residual, color='#3498db', linewidth=1.0, linestyle=':', alpha=0.8)
        
        # Styling
        ax.set_title(f'{joint_names[i]}\nRMSE={rmse:.4f} Nm', fontsize=11)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel('Residual [Nm]', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Legend only on first plot
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        # Add statistics text
        stats_text = f'μ={mean_residual:.3f}\nσ={std_residual:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Dynamics Model Validation: Torque Residuals (τ_meas - τ_RNEA)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved preview to {png_path}")
    
    # Print summary statistics
    print("\n=== Residual Statistics ===")
    for i in range(num_joints):
        tau_measured = df[f'tau_total_{i}'].values
        tau_modeled = df[f'tau_gravity_{i}'].values
        residual = tau_measured - tau_modeled
        rmse = np.sqrt(np.mean(residual**2))
        print(f"{joint_names[i]:20s}: RMSE={rmse:.4f} Nm, Mean={np.mean(residual):+.4f} Nm")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
