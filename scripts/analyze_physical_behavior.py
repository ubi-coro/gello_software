#!/usr/bin/env python3
"""
Analyze Recorded Physical Behavior vs URDF Model

This script loads recorded physical behavior data and provides analysis/visualization
to compare real robot behavior with URDF model predictions.

Usage:
    python scripts/analyze_physical_behavior.py recordings/physical_behavior_*.npz

Features:
  - Joint position/velocity time series plots
  - Gravity torque comparison (measured vs URDF)
  - Static vs dynamic behavior analysis
  - Joint-by-joint comparison reports
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")
    print("Install with: pip install matplotlib")

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


def load_recording(filepath: str) -> dict:
    """Load a recording from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def find_static_segments(
    velocities: np.ndarray,
    threshold: float = 0.05,
    min_duration: int = 10
) -> List[tuple]:
    """
    Find segments where the robot is stationary (low velocity).
    
    Returns list of (start_idx, end_idx) tuples.
    """
    # Check if all joints are below threshold
    is_static = np.all(np.abs(velocities) < threshold, axis=1)
    
    segments = []
    start = None
    
    for i, static in enumerate(is_static):
        if static and start is None:
            start = i
        elif not static and start is not None:
            if i - start >= min_duration:
                segments.append((start, i))
            start = None
    
    if start is not None and len(is_static) - start >= min_duration:
        segments.append((start, len(is_static)))
    
    return segments


def compute_velocity_from_position(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity by differentiating positions."""
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) / dt
    return vel


def verify_calibration(data: dict, urdf_path: Optional[str] = None) -> None:
    """
    Verify joint calibration by checking positions against expected ranges.
    
    Checks:
    1. Are positions within typical robot joint limits (±2π or ±π)?
    2. If URDF provided, check against URDF joint limits
    3. Check for potential sign errors (asymmetric ranges)
    4. Check for potential offset errors (unexpected mean positions)
    """
    positions = data['joint_positions']
    n_joints = positions.shape[1]
    
    print("\n" + "="*70)
    print("CALIBRATION VERIFICATION")
    print("="*70)
    
    # Get saved calibration info if available
    if 'joint_offsets' in data:
        offsets = data['joint_offsets']
        print(f"\nRecorded joint_offsets: {offsets}")
    if 'joint_signs' in data:
        signs = data['joint_signs']
        print(f"Recorded joint_signs:   {signs}")
    
    # Try to load URDF joint limits if path provided
    urdf_limits = None
    if urdf_path and HAS_PINOCCHIO:
        try:
            model = pin.buildModelFromUrdf(urdf_path)
            urdf_limits = {
                'lower': model.lowerPositionLimit[:n_joints],
                'upper': model.upperPositionLimit[:n_joints],
            }
            print(f"\nURDF joint limits loaded from: {urdf_path}")
        except Exception as e:
            print(f"\nCould not load URDF limits: {e}")
    
    print(f"\n{'Joint':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'Status':<30}")
    print("-"*70)
    
    issues = []
    
    for i in range(n_joints):
        j_min = positions[:, i].min()
        j_max = positions[:, i].max()
        j_mean = positions[:, i].mean()
        
        status = []
        
        # Check 1: Positions outside typical limits
        if j_min < -2*np.pi or j_max > 2*np.pi:
            # Multi-turn - might be normal for some joints
            if j_max - j_min > 2*np.pi:
                status.append("⚠️ Multi-turn (>360°)")
            else:
                status.append("⚠️ Outside ±2π")
        
        # Check 2: Against URDF limits if available
        if urdf_limits is not None:
            lower = urdf_limits['lower'][i]
            upper = urdf_limits['upper'][i]
            
            # Allow for multi-turn by checking modulo
            j_min_mod = ((j_min + np.pi) % (2*np.pi)) - np.pi
            j_max_mod = ((j_max + np.pi) % (2*np.pi)) - np.pi
            
            if lower > -1e10 and upper < 1e10:  # Finite limits
                if j_min_mod < lower - 0.1 or j_max_mod > upper + 0.1:
                    status.append(f"⚠️ Outside URDF [{lower:.2f}, {upper:.2f}]")
        
        # Check 3: Potential sign error
        # If range is very asymmetric around 0 or π, might indicate sign issue
        range_center = (j_max + j_min) / 2
        if abs(range_center) > np.pi and (j_max - j_min) < np.pi:
            status.append("⚠️ Asymmetric - check sign?")
        
        # Check 4: Unexpected offset
        # Mean position very far from typical poses might indicate offset error
        if abs(j_mean) > 3*np.pi:
            status.append("⚠️ Large offset - check calibration")
        
        if not status:
            status.append("✅ OK")
        
        status_str = " | ".join(status)
        print(f"Joint {i+1:<2} {j_min:>10.3f} {j_max:>10.3f} {j_mean:>10.3f} {status_str}")
        
        if "⚠️" in status_str:
            issues.append((i+1, status_str))
    
    # Additional checks
    print("\n" + "-"*70)
    print("Additional Calibration Checks:")
    print("-"*70)
    
    # Check for sign errors by looking at velocity vs position correlation
    # If sign is wrong, moving "positive" on the motor would be "negative" in joint space
    velocities = data['joint_velocities']
    
    print(f"\n{'Joint':<8} {'Pos-Vel Corr':>12} {'Interpretation':<40}")
    print("-"*70)
    
    for i in range(n_joints):
        # When position increases, velocity should be positive
        # Compute correlation between position changes and velocity
        pos_diff = np.diff(positions[:, i])
        vel_mid = velocities[:-1, i]
        
        if np.std(pos_diff) > 0.001 and np.std(vel_mid) > 0.001:
            corr = np.corrcoef(pos_diff, vel_mid)[0, 1]
            
            if corr > 0.8:
                interp = "✅ Position and velocity agree"
            elif corr < -0.8:
                interp = "❌ SIGN ERROR - pos/vel disagree!"
                issues.append((i+1, "Sign error detected"))
            else:
                interp = f"⚠️ Weak correlation ({corr:.2f})"
            
            print(f"Joint {i+1:<2} {corr:>12.3f} {interp}")
        else:
            print(f"Joint {i+1:<2} {'N/A':>12} (insufficient motion)")
    
    # Check gravity torque consistency with position
    if 'urdf_gravity_torques' in data:
        print(f"\n{'Joint':<8} {'Gravity Consistency':<50}")
        print("-"*70)
        gravity = data['urdf_gravity_torques']
        
        for i in range(min(n_joints, gravity.shape[1])):
            # For joints that should have gravity (not base rotation),
            # check if gravity changes appropriately with position
            if i == 0:  # Base joint - should have ~0 gravity
                g_range = gravity[:, i].max() - gravity[:, i].min()
                if g_range < 0.01:
                    print(f"Joint {i+1:<2} ✅ Base axis - negligible gravity as expected")
                else:
                    print(f"Joint {i+1:<2} ⚠️ Base axis has gravity variation ({g_range:.3f} Nm) - check axis?")
            else:
                # Other joints should have gravity that varies with position
                g_range = gravity[:, i].max() - gravity[:, i].min()
                if g_range < 0.001:
                    print(f"Joint {i+1:<2} ⚠️ No gravity variation - check URDF or insufficient motion")
                else:
                    # Check if gravity correlates sensibly with position (should be sinusoidal)
                    print(f"Joint {i+1:<2} ✅ Gravity varies with position (range: {g_range:.3f} Nm)")
    
    # Summary
    print("\n" + "="*70)
    if issues:
        print("⚠️  CALIBRATION ISSUES DETECTED:")
        for joint, issue in issues:
            print(f"   Joint {joint}: {issue}")
        print("\nSuggested actions:")
        print("  1. Verify joint_signs match physical motor direction vs URDF convention")
        print("  2. Re-run calibration if offsets seem wrong")
        print("  3. Check URDF joint axis definitions match physical robot")
    else:
        print("✅ CALIBRATION LOOKS GOOD - No obvious issues detected")
    print("="*70)


def analyze_joint_ranges(data: dict) -> None:
    """Print analysis of joint ranges covered during recording."""
    positions = data['joint_positions']
    n_joints = positions.shape[1]
    
    print("\n" + "="*70)
    print("JOINT RANGE ANALYSIS")
    print("="*70)
    print(f"\n{'Joint':<8} {'Min (rad)':<12} {'Max (rad)':<12} {'Range (rad)':<12} {'Range (deg)':<12}")
    print("-"*70)
    
    for i in range(n_joints):
        j_min = positions[:, i].min()
        j_max = positions[:, i].max()
        j_range = j_max - j_min
        print(f"Joint {i+1:<2} {j_min:<12.4f} {j_max:<12.4f} {j_range:<12.4f} {np.rad2deg(j_range):<12.1f}")


def analyze_static_positions(data: dict) -> None:
    """Analyze gravity torques at static positions."""
    if 'urdf_gravity_torques' not in data:
        print("\nNo URDF gravity data available for static analysis.")
        return
    
    positions = data['joint_positions']
    velocities = data['joint_velocities']
    gravity = data['urdf_gravity_torques']
    
    # Find static segments
    static_segments = find_static_segments(velocities, threshold=0.05, min_duration=10)
    
    print("\n" + "="*70)
    print("STATIC POSITION ANALYSIS")
    print("="*70)
    print(f"\nFound {len(static_segments)} static segments")
    
    if not static_segments:
        print("No static segments found. Try moving more slowly or pausing between movements.")
        return
    
    # Collect static positions and their gravity torques
    static_data = []
    for start, end in static_segments:
        mid = (start + end) // 2
        static_data.append({
            'position': positions[mid],
            'gravity': gravity[mid],
            'duration': end - start,
        })
    
    print(f"\n{'Segment':<10} {'Duration':<10} {'Position (rad)':<40} {'Gravity (Nm)':<40}")
    print("-"*100)
    
    for i, seg in enumerate(static_data[:10]):  # Show first 10
        pos_str = ' '.join([f'{p:+.2f}' for p in seg['position']])
        grav_str = ' '.join([f'{g:+.3f}' for g in seg['gravity']])
        print(f"{i+1:<10} {seg['duration']:<10} [{pos_str}] [{grav_str}]")
    
    if len(static_data) > 10:
        print(f"... and {len(static_data) - 10} more segments")


def analyze_dynamic_behavior(data: dict) -> None:
    """Analyze dynamic behavior - velocity patterns, acceleration."""
    velocities = data['joint_velocities']
    timestamps = data['timestamps']
    n_joints = velocities.shape[1]
    
    dt = np.mean(np.diff(timestamps))
    
    # Compute accelerations
    accelerations = np.zeros_like(velocities)
    accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
    
    print("\n" + "="*70)
    print("DYNAMIC BEHAVIOR ANALYSIS")
    print("="*70)
    
    print(f"\n{'Joint':<8} {'Peak Vel':<12} {'Peak Accel':<12} {'Vel RMS':<12} {'Accel RMS':<12}")
    print("-"*70)
    
    for i in range(n_joints):
        peak_vel = np.abs(velocities[:, i]).max()
        peak_acc = np.abs(accelerations[:, i]).max()
        rms_vel = np.sqrt(np.mean(velocities[:, i]**2))
        rms_acc = np.sqrt(np.mean(accelerations[:, i]**2))
        print(f"Joint {i+1:<2} {peak_vel:<12.4f} {peak_acc:<12.4f} {rms_vel:<12.4f} {rms_acc:<12.4f}")


def analyze_gravity_consistency(data: dict, urdf_path: Optional[str] = None) -> None:
    """Check consistency of gravity torques at similar positions."""
    if 'urdf_gravity_torques' not in data:
        print("\nNo URDF gravity data for consistency analysis.")
        return
    
    positions = data['joint_positions']
    gravity = data['urdf_gravity_torques']
    
    print("\n" + "="*70)
    print("GRAVITY TORQUE CONSISTENCY CHECK")
    print("="*70)
    
    # This checks if similar positions give similar gravity torques
    # (they should if the URDF model is correct)
    
    n_samples = len(positions)
    n_joints = positions.shape[1]
    
    # Sample some pairs and check consistency
    n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
    
    position_diffs = []
    gravity_diffs = []
    
    np.random.seed(42)
    for _ in range(n_pairs):
        i, j = np.random.choice(n_samples, 2, replace=False)
        pos_diff = np.linalg.norm(positions[i] - positions[j])
        grav_diff = np.linalg.norm(gravity[i] - gravity[j])
        position_diffs.append(pos_diff)
        gravity_diffs.append(grav_diff)
    
    position_diffs = np.array(position_diffs)
    gravity_diffs = np.array(gravity_diffs)
    
    # For similar positions, gravity should be similar
    close_mask = position_diffs < 0.1  # Within 0.1 rad
    if np.sum(close_mask) > 10:
        avg_grav_diff_close = np.mean(gravity_diffs[close_mask])
        print(f"\nFor positions within 0.1 rad of each other:")
        print(f"  Average gravity torque difference: {avg_grav_diff_close:.4f} Nm")
        print(f"  Number of pairs: {np.sum(close_mask)}")
    
    # Correlation between position difference and gravity difference
    correlation = np.corrcoef(position_diffs, gravity_diffs)[0, 1]
    print(f"\nCorrelation between position and gravity differences: {correlation:.3f}")
    print("(Should be positive - similar positions should have similar gravity)")


def analyze_rnea_torques(data: dict) -> None:
    """Analyze RNEA inverse dynamics torques: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)"""
    
    has_rnea = 'urdf_rnea_torques' in data
    has_gravity = 'urdf_gravity_torques' in data
    has_coriolis = 'urdf_coriolis_torques' in data
    
    if not has_rnea and not has_gravity:
        print("\nNo URDF dynamics data available.")
        return
    
    print("\n" + "="*70)
    print("RNEA INVERSE DYNAMICS ANALYSIS")
    print("="*70)
    print("\nτ_rnea = M(q)q̈ + C(q,q̇)q̇ + g(q)")
    print("This is the torque required to produce the observed motion.\n")
    
    n_joints = data['joint_positions'].shape[1]
    
    if has_gravity:
        gravity = data['urdf_gravity_torques']
        print(f"{'Gravity Torques g(q) [Nm]':^70}")
        print("-"*70)
        print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'RMS':>10}")
        print("-"*70)
        for i in range(min(n_joints, gravity.shape[1])):
            g = gravity[:, i]
            print(f"Joint {i+1:<2} {g.min():>10.4f} {g.max():>10.4f} {g.mean():>10.4f} {np.sqrt(np.mean(g**2)):>10.4f}")
    
    if has_coriolis:
        coriolis = data['urdf_coriolis_torques']
        print(f"\n{'Coriolis/Centrifugal Torques C(q,dq)*dq [Nm]':^70}")
        print("-"*70)
        print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'RMS':>10}")
        print("-"*70)
        for i in range(min(n_joints, coriolis.shape[1])):
            c = coriolis[:, i]
            print(f"Joint {i+1:<2} {c.min():>10.4f} {c.max():>10.4f} {c.mean():>10.4f} {np.sqrt(np.mean(c**2)):>10.4f}")
    
    if has_rnea:
        rnea = data['urdf_rnea_torques']
        print(f"\n{'Full RNEA Torques τ = M*q̈ + C*q̇ + g [Nm]':^70}")
        print("-"*70)
        print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'RMS':>10}")
        print("-"*70)
        for i in range(min(n_joints, rnea.shape[1])):
            t = rnea[:, i]
            print(f"Joint {i+1:<2} {t.min():>10.4f} {t.max():>10.4f} {t.mean():>10.4f} {np.sqrt(np.mean(t**2)):>10.4f}")
        
        # Compare RNEA to gravity-only
        if has_gravity:
            print(f"\n{'Comparison: RNEA vs Gravity-only':^70}")
            print("-"*70)
            print("(Difference shows inertial + Coriolis contribution)")
            print(f"{'Joint':<8} {'RNEA RMS':>12} {'Gravity RMS':>12} {'Diff RMS':>12} {'Ratio':>10}")
            print("-"*70)
            for i in range(min(n_joints, rnea.shape[1], gravity.shape[1])):
                rnea_rms = np.sqrt(np.mean(rnea[:, i]**2))
                grav_rms = np.sqrt(np.mean(gravity[:, i]**2))
                diff_rms = np.sqrt(np.mean((rnea[:, i] - gravity[:, i])**2))
                ratio = diff_rms / grav_rms if grav_rms > 0.001 else 0
                print(f"Joint {i+1:<2} {rnea_rms:>12.4f} {grav_rms:>12.4f} {diff_rms:>12.4f} {ratio:>10.2f}")
            
            print("\n  Interpretation:")
            print("  - Ratio ≈ 0: Motion is quasi-static, gravity dominates")
            print("  - Ratio > 0.5: Significant dynamic effects (fast motion)")
            print("  - High ratio + low gravity: Inertia/Coriolis dominant")


def plot_time_series(data: dict, output_path: Optional[str] = None):
    """Plot joint positions and velocities over time."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting.")
        return
    
    timestamps = data['timestamps']
    t = timestamps - timestamps[0]  # Start from 0
    positions = data['joint_positions']
    velocities = data['joint_velocities']
    n_joints = positions.shape[1]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Positions
    ax1 = axes[0]
    for i in range(n_joints):
        ax1.plot(t, positions[:, i], label=f'Joint {i+1}', alpha=0.8)
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Joint Positions Over Time')
    ax1.legend(loc='upper right', ncol=n_joints)
    ax1.grid(True, alpha=0.3)
    
    # Velocities
    ax2 = axes[1]
    for i in range(n_joints):
        ax2.plot(t, velocities[:, i], label=f'Joint {i+1}', alpha=0.8)
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Joint Velocities Over Time')
    ax2.legend(loc='upper right', ncol=n_joints)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series plot to: {output_path}")
    else:
        plt.show()


def plot_gravity_torques(data: dict, output_path: Optional[str] = None):
    """Plot gravity torques over time and vs position."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting.")
        return
    
    if 'urdf_gravity_torques' not in data:
        print("No URDF gravity data for plotting.")
        return
    
    timestamps = data['timestamps']
    t = timestamps - timestamps[0]
    positions = data['joint_positions']
    gravity = data['urdf_gravity_torques']
    n_joints = min(positions.shape[1], gravity.shape[1])
    
    fig, axes = plt.subplots(2, n_joints, figsize=(3*n_joints, 8))
    
    for i in range(n_joints):
        # Gravity over time
        axes[0, i].plot(t, gravity[:, i], 'b-', alpha=0.7)
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Gravity (Nm)')
        axes[0, i].set_title(f'Joint {i+1} Gravity vs Time')
        axes[0, i].grid(True, alpha=0.3)
        
        # Gravity vs position
        axes[1, i].scatter(positions[:, i], gravity[:, i], s=1, alpha=0.3)
        axes[1, i].set_xlabel('Position (rad)')
        axes[1, i].set_ylabel('Gravity (Nm)')
        axes[1, i].set_title(f'Joint {i+1} Gravity vs Position')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved gravity plot to: {output_path}")
    else:
        plt.show()


def plot_rnea_torques(data: dict, output_path: Optional[str] = None):
    """Plot RNEA torques breakdown: gravity, Coriolis, and total."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting.")
        return
    
    has_rnea = 'urdf_rnea_torques' in data
    has_gravity = 'urdf_gravity_torques' in data
    has_coriolis = 'urdf_coriolis_torques' in data
    
    if not has_rnea and not has_gravity:
        print("No URDF dynamics data for plotting.")
        return
    
    timestamps = data['timestamps']
    t = timestamps - timestamps[0]
    n_joints = data['joint_positions'].shape[1]
    
    # Determine number of rows based on available data
    n_rows = sum([has_gravity, has_coriolis, has_rnea])
    if n_rows == 0:
        return
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3*n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    
    row = 0
    
    if has_gravity:
        gravity = data['urdf_gravity_torques']
        ax = axes[row]
        for i in range(min(n_joints, gravity.shape[1])):
            ax.plot(t, gravity[:, i], label=f'Joint {i+1}', alpha=0.8)
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Gravity Torques g(q)')
        ax.legend(loc='upper right', ncol=n_joints)
        ax.grid(True, alpha=0.3)
        row += 1
    
    if has_coriolis:
        coriolis = data['urdf_coriolis_torques']
        ax = axes[row]
        for i in range(min(n_joints, coriolis.shape[1])):
            ax.plot(t, coriolis[:, i], label=f'Joint {i+1}', alpha=0.8)
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Coriolis/Centrifugal Torques C(q,q̇)q̇')
        ax.legend(loc='upper right', ncol=n_joints)
        ax.grid(True, alpha=0.3)
        row += 1
    
    if has_rnea:
        rnea = data['urdf_rnea_torques']
        ax = axes[row]
        for i in range(min(n_joints, rnea.shape[1])):
            ax.plot(t, rnea[:, i], label=f'Joint {i+1}', alpha=0.8)
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Full RNEA: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)')
        ax.legend(loc='upper right', ncol=n_joints)
        ax.grid(True, alpha=0.3)
        row += 1
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved RNEA torques plot to: {output_path}")
    else:
        plt.show()


def plot_phase_portraits(data: dict, output_path: Optional[str] = None):
    """Plot position vs velocity phase portraits for each joint."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting.")
        return
    
    positions = data['joint_positions']
    velocities = data['joint_velocities']
    n_joints = positions.shape[1]
    
    cols = 3
    rows = (n_joints + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for i in range(n_joints):
        ax = axes[i]
        # Color by time
        colors = np.arange(len(positions))
        scatter = ax.scatter(positions[:, i], velocities[:, i], c=colors, s=1, alpha=0.5, cmap='viridis')
        ax.set_xlabel('Position (rad)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title(f'Joint {i+1} Phase Portrait')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase portrait to: {output_path}")
    else:
        plt.show()


def generate_comparison_report(data: dict, urdf_path: Optional[str] = None) -> str:
    """Generate a text report comparing physical behavior with URDF model."""
    report = []
    report.append("="*70)
    report.append("PHYSICAL BEHAVIOR vs URDF COMPARISON REPORT")
    report.append("="*70)
    
    positions = data['joint_positions']
    velocities = data['joint_velocities']
    timestamps = data['timestamps']
    n_joints = positions.shape[1]
    duration = timestamps[-1] - timestamps[0]
    
    report.append(f"\nRecording Summary:")
    report.append(f"  - Duration: {duration:.2f} seconds")
    report.append(f"  - Samples: {len(timestamps)}")
    report.append(f"  - Sample rate: {len(timestamps)/duration:.1f} Hz")
    report.append(f"  - Number of joints: {n_joints}")
    
    # Joint range coverage
    report.append(f"\nJoint Range Coverage:")
    for i in range(n_joints):
        j_range = positions[:, i].max() - positions[:, i].min()
        report.append(f"  Joint {i+1}: {np.rad2deg(j_range):.1f}° range covered")
    
    # Velocity analysis - can indicate friction issues
    report.append(f"\nVelocity Analysis (friction indicators):")
    for i in range(n_joints):
        vel = velocities[:, i]
        # Asymmetry between positive and negative velocities might indicate friction
        pos_vel_mean = np.mean(vel[vel > 0.01]) if np.sum(vel > 0.01) > 0 else 0
        neg_vel_mean = np.mean(vel[vel < -0.01]) if np.sum(vel < -0.01) > 0 else 0
        asymmetry = abs(pos_vel_mean + neg_vel_mean) / max(abs(pos_vel_mean), abs(neg_vel_mean), 0.001)
        report.append(f"  Joint {i+1}: velocity asymmetry = {asymmetry:.3f} (>0.2 may indicate friction)")
    
    # Gravity torque analysis
    if 'urdf_gravity_torques' in data:
        gravity = data['urdf_gravity_torques']
        report.append(f"\nURDF Gravity Torque Analysis:")
        for i in range(min(n_joints, gravity.shape[1])):
            g_min, g_max = gravity[:, i].min(), gravity[:, i].max()
            g_range = g_max - g_min
            report.append(f"  Joint {i+1}: gravity range = {g_range:.3f} Nm [{g_min:.3f} to {g_max:.3f}]")
        
        report.append(f"\nObservations for URDF validation:")
        report.append(f"  1. If robot 'drifts' when stationary, URDF gravity may be inaccurate")
        report.append(f"  2. If robot feels 'heavy' to move, friction model may be missing")
        report.append(f"  3. Velocity asymmetry > 0.2 suggests friction effects not in URDF")
    else:
        report.append(f"\nNo URDF data available for comparison.")
        report.append(f"Re-run recording with --urdf flag to enable URDF comparison.")
    
    report.append("\n" + "="*70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze recorded physical behavior data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'recording', type=str,
        help='Path to recording file (.npz)'
    )
    parser.add_argument(
        '--urdf', type=str, default=None,
        help='Path to URDF file for additional analysis'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save plots (shows interactively if not specified)'
    )
    parser.add_argument(
        '--report', type=str, default=None,
        help='Save text report to file'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading recording: {args.recording}")
    data = load_recording(args.recording)
    
    print(f"Loaded {len(data['timestamps'])} samples")
    print(f"Available fields: {list(data.keys())}")
    
    # Resolve URDF path if provided
    urdf_path = args.urdf
    if urdf_path and not os.path.isabs(urdf_path):
        # Try relative to current dir, then project root
        if not os.path.exists(urdf_path):
            project_root = Path(__file__).parent.parent
            urdf_path = str(project_root / urdf_path)
    
    # Run analyses
    verify_calibration(data, urdf_path)  # NEW: Calibration verification
    analyze_joint_ranges(data)
    analyze_static_positions(data)
    analyze_dynamic_behavior(data)
    analyze_rnea_torques(data)
    analyze_gravity_consistency(data, urdf_path)
    
    # Generate report
    report = generate_comparison_report(data, args.urdf)
    print(report)
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Saved report to: {args.report}")
    
    # Generate plots
    if args.plot and HAS_MATPLOTLIB:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            base_name = Path(args.recording).stem
            
            plot_time_series(data, str(output_dir / f"{base_name}_time_series.png"))
            plot_gravity_torques(data, str(output_dir / f"{base_name}_gravity.png"))
            plot_rnea_torques(data, str(output_dir / f"{base_name}_rnea_torques.png"))
            plot_phase_portraits(data, str(output_dir / f"{base_name}_phase.png"))
        else:
            plot_time_series(data)
            plot_gravity_torques(data)
            plot_rnea_torques(data)
            plot_phase_portraits(data)


if __name__ == '__main__':
    main()
