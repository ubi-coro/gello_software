#!/usr/bin/env python3
"""
Contact Behavior Comparison Plot (Section 4.3.3)

Compares position control vs. impedance control during contact.
Shows that impedance control yields safely on contact while position
control generates high force peaks.

Usage:
    python plot_contact_behavior.py <position_log.csv> <impedance_log.csv> --output contact_behavior.svg
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


def load_and_filter(logfile, mode):
    """Load and filter data by control mode."""
    df = pd.read_csv(logfile)
    
    # Filter by mode if available
    if 'control_mode' in df.columns:
        df = df[df['control_mode'] == mode]
    
    # Convert to relative time
    if len(df) > 0:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare position vs. impedance control contact")
    parser.add_argument("position_log", type=str, help="Log file from position control mode")
    parser.add_argument("impedance_log", type=str, help="Log file from impedance control mode")
    parser.add_argument("--output", type=str, default="contact_behavior.svg", help="Output filename")
    parser.add_argument("--tcp-axis", type=int, default=2, 
                        help="TCP force axis to plot (0-2 for Fx,Fy,Fz; default: 2=Fz)")
    parser.add_argument("--follower-joint", type=int, default=2, 
                        help="Follower joint for position plot (default: 2)")
    args = parser.parse_args()

    print("Loading position control data...")
    df_pos = load_and_filter(args.position_log, "position_teleop")
    
    # Fallback if mode filtering didn't work
    if len(df_pos) == 0:
        df_pos = pd.read_csv(args.position_log)
        if len(df_pos) > 0:
            df_pos['time'] = df_pos['timestamp'] - df_pos['timestamp'].iloc[0]
    
    print("Loading impedance control data...")
    df_imp = load_and_filter(args.impedance_log, "impedance_teleop")
    
    # Fallback
    if len(df_imp) == 0:
        df_imp = pd.read_csv(args.impedance_log)
        if len(df_imp) > 0:
            df_imp['time'] = df_imp['timestamp'] - df_imp['timestamp'].iloc[0]
    
    if len(df_pos) == 0 or len(df_imp) == 0:
        print("Error: Could not load data from both files")
        return 1
    
    print(f"Position control: {len(df_pos)} samples, {df_pos['time'].iloc[-1]:.1f}s")
    print(f"Impedance control: {len(df_imp)} samples, {df_imp['time'].iloc[-1]:.1f}s")
    
    # Extract data
    tcp_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    tcp_units = ["N", "N", "N", "Nm", "Nm", "Nm"]
    
    # Create subplots (2 rows: Position, Force)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    
    # === Upper Plot: Follower Position ===
    # Position control (blue)
    if f'q_follower_{args.follower_joint}' in df_pos.columns:
        pos_position = np.rad2deg(df_pos[f'q_follower_{args.follower_joint}'].values)
        pos_time = df_pos['time'].values
        ax1.plot(pos_time, pos_position, linewidth=2.0, alpha=0.85, color='#3498db',
                 label='Position Control')
    
    # Impedance control (red)
    if f'q_follower_{args.follower_joint}' in df_imp.columns:
        imp_position = np.rad2deg(df_imp[f'q_follower_{args.follower_joint}'].values)
        imp_time = df_imp['time'].values
        ax1.plot(imp_time, imp_position, linewidth=2.0, alpha=0.85, color='#e74c3c',
                 label='Impedance Control')
    
    ax1.set_ylabel(f'Follower Joint {args.follower_joint} Position [°]', fontsize=13)
    ax1.set_title('Contact Behavior Comparison: Position vs. Impedance Control',
                  fontsize=15, pad=10, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    ax1.text(0.02, 0.98, 'Impedance control yields on contact\nPosition control fights against obstacle',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
    
    # === Lower Plot: TCP Force ===
    # Position control (blue)
    if f'tcp_force_{args.tcp_axis}' in df_pos.columns:
        pos_force = df_pos[f'tcp_force_{args.tcp_axis}'].values
        pos_time = df_pos['time'].values
        ax2.plot(pos_time, pos_force, linewidth=2.0, alpha=0.85, color='#3498db',
                 label='Position Control')
        
        # Highlight max force
        max_force_pos = np.max(np.abs(pos_force))
        max_idx_pos = np.argmax(np.abs(pos_force))
        ax2.plot(pos_time[max_idx_pos], pos_force[max_idx_pos], 'D', 
                 markersize=10, color='#3498db', markeredgecolor='black', markeredgewidth=1.5,
                 label=f'Max: {max_force_pos:.1f} {tcp_units[args.tcp_axis]}')
    
    # Impedance control (red)
    if f'tcp_force_{args.tcp_axis}' in df_imp.columns:
        imp_force = df_imp[f'tcp_force_{args.tcp_axis}'].values
        imp_time = df_imp['time'].values
        ax2.plot(imp_time, imp_force, linewidth=2.0, alpha=0.85, color='#e74c3c',
                 label='Impedance Control')
        
        # Highlight max force
        max_force_imp = np.max(np.abs(imp_force))
        max_idx_imp = np.argmax(np.abs(imp_force))
        ax2.plot(imp_time[max_idx_imp], imp_force[max_idx_imp], 'D',
                 markersize=10, color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5,
                 label=f'Max: {max_force_imp:.1f} {tcp_units[args.tcp_axis]}')
        
        # Print comparison
        print(f"\n=== Force Comparison ===")
        print(f"Position Control Max Force: {max_force_pos:.2f} {tcp_units[args.tcp_axis]}")
        print(f"Impedance Control Max Force: {max_force_imp:.2f} {tcp_units[args.tcp_axis]}")
        print(f"Reduction: {(1 - max_force_imp/max_force_pos)*100:.1f}%")
    
    ax2.set_xlabel('Time [s]', fontsize=13)
    ax2.set_ylabel(f'TCP Force {tcp_labels[args.tcp_axis]} [{tcp_units[args.tcp_axis]}]', fontsize=13)
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Add annotation
    ax2.text(0.02, 0.98, 'Impedance control: Lower peak forces\nPosition control: High force spikes (unsafe)',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
    
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
