#!/usr/bin/env python3
"""
Friction Identification Plot (Section 4.2.1)

Plots the friction characteristic curve (velocity vs. friction torque)
for a single joint. This demonstrates the nonlinear friction model:
    τ_friction = τ_coulomb * sign(dq) + τ_viscous * dq + stiction @ dq≈0

Usage:
    python plot_friction_identification.py <logfile.csv> --joint 2 --output friction.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit

# Scientific plotting style
sns.set_context("paper", font_scale=1.4)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'svg',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def coulomb_viscous_model(velocity, tau_coulomb, tau_viscous, deadband=0.01):
    """Coulomb + viscous friction model."""
    tau = np.zeros_like(velocity)
    for i, v in enumerate(velocity):
        if abs(v) < deadband:
            # In deadband: linear interpolation through zero
            tau[i] = 0.0
        else:
            # Coulomb + viscous
            tau[i] = tau_coulomb * np.sign(v) + tau_viscous * v
    return tau


def main():
    parser = argparse.ArgumentParser(description="Plot friction identification curve")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    parser.add_argument("--joint", type=int, default=2, help="Joint index (0-5), default: 2 (shoulder)")
    parser.add_argument("--output", type=str, default="friction_identification.svg", help="Output filename")
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
    
    joint_idx = args.joint
    print(f"Analyzing Joint {joint_idx} ({len(df)} samples)")
    
    # Extract velocity and measured friction
    velocity = df[f'q_dot_leader_{joint_idx}'].values
    
    # Use logged friction torque directly (includes friction comp output)
    if f'tau_friction_{joint_idx}' in df.columns:
        tau_friction_measured = df[f'tau_friction_{joint_idx}'].values
        print(f"Using logged tau_friction column (includes compensation if enabled)")
    else:
        # Fallback: compute from total - gravity
        tau_gravity = df[f'tau_gravity_{joint_idx}'].values
        tau_total = df[f'tau_total_{joint_idx}'].values
        tau_friction_measured = tau_total - tau_gravity
        print(f"Warning: Computing friction as total - gravity (may be inaccurate)")
    
    # Filter outliers (beyond 99th percentile)
    valid_mask = np.abs(tau_friction_measured) < np.percentile(np.abs(tau_friction_measured), 99)
    velocity = velocity[valid_mask]
    tau_friction_measured = tau_friction_measured[valid_mask]
    
    # Additional filtering: remove near-zero velocity samples (deadband noise)
    motion_mask = np.abs(velocity) > 0.005  # 0.005 rad/s threshold
    velocity = velocity[motion_mask]
    tau_friction_measured = tau_friction_measured[motion_mask]
    
    print(f"After filtering: {len(velocity)} samples")
    print(f"Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}] rad/s")
    print(f"Friction range: [{tau_friction_measured.min():.4f}, {tau_friction_measured.max():.4f}] Nm")
    
    # Fit friction model
    try:
        # Check if there's enough variation in the data
        if np.std(tau_friction_measured) < 0.001:
            print(f"Warning: Very low friction variation (std={np.std(tau_friction_measured):.6f})")
            print("  → Friction compensation might already be active, or no friction present")
            raise ValueError("Insufficient data variation for fitting")
        
        # Initial guess based on data range
        tau_range = np.abs(tau_friction_measured).max()
        vel_range = np.abs(velocity).max()
        p0_coulomb = tau_range * 0.3  # Guess ~30% is Coulomb
        p0_viscous = (tau_range * 0.7) / vel_range if vel_range > 0 else 0.1
        
        print(f"Initial guess: τ_c={p0_coulomb:.4f}, τ_v={p0_viscous:.4f}")
        
        popt, _ = curve_fit(
            coulomb_viscous_model, 
            velocity, 
            tau_friction_measured,
            p0=[p0_coulomb, p0_viscous],
            bounds=([0, 0], [5.0, 2.0]),  # Reasonable bounds
            maxfev=10000
        )
        tau_coulomb_fit, tau_viscous_fit = popt
        print(f"\n✓ Fitted parameters:")
        print(f"  Coulomb friction: {tau_coulomb_fit:.4f} Nm")
        print(f"  Viscous friction: {tau_viscous_fit:.4f} Nm·s/rad")
        print(f"\n→ Add to config yaml:")
        print(f"   friction_feedforward[{joint_idx}] = {tau_coulomb_fit:.4f}")
        print(f"   viscous_friction[{joint_idx}] = {tau_viscous_fit:.4f}")
        
        # Generate smooth model curve
        vel_model = np.linspace(velocity.min(), velocity.max(), 500)
        tau_model = coulomb_viscous_model(vel_model, tau_coulomb_fit, tau_viscous_fit)
    except Exception as e:
        print(f"Warning: Model fitting failed: {e}")
        tau_coulomb_fit, tau_viscous_fit = None, None
        vel_model, tau_model = None, None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot of measurements
    ax.scatter(velocity, tau_friction_measured, 
               s=2, alpha=0.3, c='#3498db', label='Measured', rasterized=True)
    
    # Plot fitted model
    if tau_coulomb_fit is not None:
        ax.plot(vel_model, tau_model, 'r-', linewidth=2.5, 
                label=f'Model ($\\tau_C$={tau_coulomb_fit:.3f}, $\\tau_v$={tau_viscous_fit:.3f})')
    
    # Styling
    ax.set_xlabel('Joint Velocity [rad/s]', fontsize=14)
    ax.set_ylabel('Friction Torque [Nm]', fontsize=14)
    ax.set_title(f'Friction Characteristic Curve - Joint {joint_idx}', fontsize=16, pad=15)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    
    # Also save as PNG for quick preview
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved preview to {png_path}")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
