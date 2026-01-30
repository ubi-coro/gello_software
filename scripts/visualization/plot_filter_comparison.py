#!/usr/bin/env python3
"""
Filter Comparison Plot (Section 4.2.2)

Compares raw, EMA-filtered, and 1€-filtered velocity signals to demonstrate
the trade-off between noise reduction and latency.

Note: This script assumes you've recorded data with different filter settings
or can apply filters post-hoc to demonstrate the comparison.

Usage:
    python plot_filter_comparison.py <logfile.csv> --joint 4 --output filter_comparison.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


class OneEuroFilter:
    """Simple 1€ Filter implementation for comparison."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            dt = 0.001
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Smooth derivative with low-pass filter
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_smooth = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Adaptive cutoff based on derivative magnitude
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        
        # Smooth value
        alpha = self._alpha(dt, cutoff)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_smooth
        self.t_prev = t
        
        return x_filtered
    
    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


def apply_ema(signal, alpha=0.1):
    """Apply exponential moving average filter."""
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
    return filtered


def apply_one_euro(signal, time):
    """Apply 1€ filter to signal."""
    filter_obj = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    filtered = np.zeros_like(signal)
    for i in range(len(signal)):
        filtered[i] = filter_obj(signal[i], time[i])
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Compare velocity filters")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    parser.add_argument("--joint", type=int, default=4, help="Joint index (0-5), default: 4 (wrist)")
    parser.add_argument("--output", type=str, default="filter_comparison.svg", help="Output filename")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Time window to plot [start, end] in seconds")
    parser.add_argument("--ema-alpha", type=float, default=0.1, help="EMA filter alpha (default: 0.1)")
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
    
    joint_idx = args.joint
    print(f"Analyzing Joint {joint_idx} ({len(df)} samples)")
    
    # Extract data
    time = df['time'].values
    velocity = df[f'q_dot_leader_{joint_idx}'].values
    
    # Compute raw velocity from position (for comparison)
    position = df[f'q_leader_{joint_idx}'].values
    dt = np.diff(time, prepend=time[0])
    velocity_raw = np.diff(position, prepend=position[0]) / np.where(dt > 0, dt, 0.001)
    
    # Apply filters
    print("Applying filters...")
    velocity_ema = apply_ema(velocity_raw, alpha=args.ema_alpha)
    velocity_one_euro = apply_one_euro(velocity_raw, time)
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # === Upper plot: Full comparison ===
    ax1.plot(time, velocity_raw, linewidth=0.5, alpha=0.4, color='#95a5a6', 
             label='Raw (Finite Difference)', rasterized=True)
    ax1.plot(time, velocity_ema, linewidth=1.5, alpha=0.85, color='#3498db', 
             label=f'EMA (α={args.ema_alpha})')
    ax1.plot(time, velocity_one_euro, linewidth=1.5, alpha=0.85, color='#e74c3c', 
             label='1€ Filter')
    
    ax1.set_ylabel('Velocity [rad/s]', fontsize=13)
    ax1.set_title(f'Filter Comparison - Joint {joint_idx}', fontsize=15, pad=10)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # === Lower plot: Zoom on transition (stillstand → motion) ===
    # Find a good transition region (where velocity changes rapidly)
    velocity_abs = np.abs(velocity_raw)
    # Find where velocity exceeds threshold after being low
    threshold = 0.5
    low_vel_idx = np.where(velocity_abs < threshold)[0]
    high_vel_idx = np.where(velocity_abs > threshold)[0]
    
    if len(low_vel_idx) > 0 and len(high_vel_idx) > 0:
        # Find first transition
        transition_idx = None
        for idx in high_vel_idx:
            if idx > 100 and any(low_vel_idx < idx):
                transition_idx = idx
                break
        
        if transition_idx is not None:
            # Zoom window: ±1 second around transition
            zoom_start = max(0, transition_idx - 500)  # Assuming ~500Hz
            zoom_end = min(len(time), transition_idx + 500)
            
            ax2.plot(time[zoom_start:zoom_end], velocity_raw[zoom_start:zoom_end], 
                     linewidth=0.8, alpha=0.5, color='#95a5a6', label='Raw')
            ax2.plot(time[zoom_start:zoom_end], velocity_ema[zoom_start:zoom_end], 
                     linewidth=2.0, alpha=0.9, color='#3498db', label='EMA')
            ax2.plot(time[zoom_start:zoom_end], velocity_one_euro[zoom_start:zoom_end], 
                     linewidth=2.0, alpha=0.9, color='#e74c3c', label='1€ Filter')
            
            # Mark transition point
            ax2.axvline(time[transition_idx], color='k', linewidth=1.5, linestyle='--', 
                       alpha=0.6, label='Motion Onset')
            
            ax2.set_xlabel('Time [s]', fontsize=13)
            ax2.set_ylabel('Velocity [rad/s]', fontsize=13)
            ax2.set_title('Zoomed View: Motion Onset Detail', fontsize=13, pad=8)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)
    else:
        # No clear transition found, just zoom middle section
        mid = len(time) // 2
        zoom_start = max(0, mid - 500)
        zoom_end = min(len(time), mid + 500)
        
        ax2.plot(time[zoom_start:zoom_end], velocity_raw[zoom_start:zoom_end], 
                 linewidth=0.8, alpha=0.5, color='#95a5a6', label='Raw')
        ax2.plot(time[zoom_start:zoom_end], velocity_ema[zoom_start:zoom_end], 
                 linewidth=2.0, alpha=0.9, color='#3498db', label='EMA')
        ax2.plot(time[zoom_start:zoom_end], velocity_one_euro[zoom_start:zoom_end], 
                 linewidth=2.0, alpha=0.9, color='#e74c3c', label='1€ Filter')
        
        ax2.set_xlabel('Time [s]', fontsize=13)
        ax2.set_ylabel('Velocity [rad/s]', fontsize=13)
        ax2.set_title('Zoomed View: Detail', fontsize=13, pad=8)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
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
