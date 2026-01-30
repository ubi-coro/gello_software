#!/usr/bin/env python3
"""
Force-Torque Correlation Plot (Section 4.3.2)

Demonstrates the causal relationship between external forces at the follower
robot and feedback torques at the leader robot. Shows that haptic feedback
is correctly transmitted.

Usage:
    python plot_force_torque_correlation.py <logfile.csv> --output force_torque_corr.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal

# Scientific plotting style
sns.set_context("paper", font_scale=1.4)
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
    parser = argparse.ArgumentParser(description="Plot force-torque correlation")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    parser.add_argument("--output", type=str, default="force_torque_correlation.svg", 
                        help="Output filename")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Time window to plot [start, end] in seconds")
    parser.add_argument("--joint", type=int, default=2, 
                        help="Joint to analyze (default: 2 = shoulder)")
    parser.add_argument("--tcp-axis", type=int, default=2,
                        help="TCP force axis to plot (0-2: Fx,Fy,Fz; 3-5: Tx,Ty,Tz; default: 2=Fz)")
    parser.add_argument("--filter-mode", type=str, default="impedance_teleop",
                        help="Control mode to analyze")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Filter by control mode
    if args.filter_mode:
        df = df[df['control_mode'] == args.filter_mode]
    
    if len(df) == 0:
        print(f"Error: No data for mode '{args.filter_mode}'")
        print("Available modes:", df['control_mode'].unique())
        return 1
    
    # Convert timestamp to relative time
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Apply time window if specified
    if args.time_window:
        t_start, t_end = args.time_window
        df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
    
    print(f"Analyzing {len(df)} samples over {df['time'].iloc[-1]:.1f} seconds")
    
    # Extract data
    time = df['time'].values
    tcp_force_raw = df[f'tcp_force_{args.tcp_axis}'].values
    tau_feedback = df[f'tau_feedback_{args.joint}'].values
    
    # Smooth for better visualization (optional)
    window = 51  # Must be odd
    if len(tcp_force_raw) > window:
        tcp_force = signal.savgol_filter(tcp_force_raw, window, 3)
    else:
        tcp_force = tcp_force_raw
    
    # Calculate cross-correlation to show causality
    if len(tcp_force) > 100:
        # Normalize signals
        tcp_norm = (tcp_force - np.mean(tcp_force)) / (np.std(tcp_force) + 1e-9)
        tau_norm = (tau_feedback - np.mean(tau_feedback)) / (np.std(tau_feedback) + 1e-9)
        
        # Compute cross-correlation
        correlation = np.correlate(tcp_norm, tau_norm, mode='full')
        lags = np.arange(-len(tcp_norm) + 1, len(tau_norm))
        
        # Find peak correlation and corresponding lag
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_corr = correlation[peak_idx]
        
        # Convert lag to time (assuming constant sample rate)
        dt = np.mean(np.diff(time))
        lag_time_ms = peak_lag * dt * 1000
        
        print(f"\n=== Correlation Analysis ===")
        print(f"Peak correlation: {peak_corr:.3f}")
        print(f"Lag: {peak_lag} samples ({lag_time_ms:.1f} ms)")
    else:
        peak_corr = 0
        lag_time_ms = 0
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Axis labels
    tcp_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    tcp_units = ["N", "N", "N", "Nm", "Nm", "Nm"]
    
    # Plot TCP force on left axis (blue)
    color1 = '#3498db'
    ax1.set_xlabel('Time [s]', fontsize=14)
    ax1.set_ylabel(f'TCP Force {tcp_labels[args.tcp_axis]} [{tcp_units[args.tcp_axis]}]', 
                   fontsize=14, color=color1)
    line1 = ax1.plot(time, tcp_force, linewidth=2.0, alpha=0.85, color=color1,
                     label=f'Follower TCP {tcp_labels[args.tcp_axis]}')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot feedback torque on right axis (red)
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel(f'Leader Feedback Torque Joint {args.joint} [Nm]', 
                   fontsize=14, color=color2)
    line2 = ax2.plot(time, tau_feedback, linewidth=2.0, alpha=0.85, color=color2,
                     label=f'Leader Torque J{args.joint}')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add correlation info to title
    title = f'Force-Torque Correlation: Haptic Feedback Validation\n'
    title += f'(Correlation: {peak_corr:.3f}, Lag: {lag_time_ms:.1f} ms)'
    ax1.set_title(title, fontsize=15, pad=15, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.95)
    
    # Add annotation box
    textstr = f'Joint {args.joint} responds to {tcp_labels[args.tcp_axis]} forces\n'
    textstr += 'Curves should be similar in shape (causal relationship)'
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved preview to {png_path}")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
