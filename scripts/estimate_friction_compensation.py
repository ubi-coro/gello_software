#!/usr/bin/env python3
"""
Estimate Friction and Improve Gravity Compensation Parameters

This script analyzes recorded physical behavior data to:
1. Estimate Coulomb (static) and viscous (velocity-dependent) friction
2. Validate URDF gravity model against observed behavior
3. Generate improved compensation parameters

The friction model used is:
    τ_friction = τ_coulomb * sign(dq) + τ_viscous * dq
    
Where:
    τ_coulomb = static/Coulomb friction (constant opposing motion)
    τ_viscous = viscous friction coefficient (proportional to velocity)

Usage:
    python scripts/estimate_friction_compensation.py recordings/physical_behavior_*.npz
    
    # With URDF for gravity validation:
    python scripts/estimate_friction_compensation.py recordings/*.npz \\
        --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for saving to file
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


def load_recording(filepath: str) -> dict:
    """Load a recording from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def estimate_coulomb_friction(velocities: np.ndarray, accelerations: np.ndarray, 
                               gravity_torques: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Coulomb (static) friction from velocity sign changes.
    
    At the moment of direction change (velocity crosses zero), the friction
    torque must overcome both gravity and change direction. This gives us
    a lower bound on Coulomb friction.
    
    Returns:
        (friction_estimates, confidence): Arrays of friction estimates per joint
    """
    n_joints = velocities.shape[1]
    friction_pos = [[] for _ in range(n_joints)]  # Friction when moving positive
    friction_neg = [[] for _ in range(n_joints)]  # Friction when moving negative
    
    for i in range(n_joints):
        vel = velocities[:, i]
        acc = accelerations[:, i] if accelerations is not None else np.zeros_like(vel)
        grav = gravity_torques[:, i] if gravity_torques is not None else np.zeros_like(vel)
        
        # Find samples with clear motion direction
        moving_pos = vel > 0.1  # Moving in positive direction
        moving_neg = vel < -0.1  # Moving in negative direction
        
        # At constant velocity (low acceleration), friction ≈ gravity (quasi-static)
        # τ_friction ≈ -τ_gravity when dq≈const
        low_acc = np.abs(acc) < 1.0  # Low acceleration
        
        # For positive motion with low acceleration
        mask_pos = moving_pos & low_acc
        if np.sum(mask_pos) > 10:
            # Friction opposes motion, so τ_friction ≈ -τ_gravity - τ_inertia
            # At quasi-static: τ_friction ≈ -τ_gravity
            friction_pos[i] = -grav[mask_pos]
        
        # For negative motion
        mask_neg = moving_neg & low_acc
        if np.sum(mask_neg) > 10:
            friction_neg[i] = -grav[mask_neg]
    
    # Estimate Coulomb friction as the bias between positive and negative motion
    coulomb_friction = np.zeros(n_joints)
    confidence = np.zeros(n_joints)
    
    for i in range(n_joints):
        if len(friction_pos[i]) > 10 and len(friction_neg[i]) > 10:
            # Coulomb friction ≈ (mean_positive - mean_negative) / 2
            mean_pos = np.mean(friction_pos[i])
            mean_neg = np.mean(friction_neg[i])
            coulomb_friction[i] = (mean_pos - mean_neg) / 2
            
            # Confidence based on consistency
            std_pos = np.std(friction_pos[i])
            std_neg = np.std(friction_neg[i])
            confidence[i] = 1.0 / (1.0 + std_pos + std_neg)
    
    return coulomb_friction, confidence


def estimate_viscous_friction(velocities: np.ndarray, accelerations: np.ndarray,
                               gravity_torques: np.ndarray, 
                               rnea_torques: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate viscous friction coefficient from velocity-torque relationship.
    
    τ_viscous = B * dq
    
    We estimate B by looking at the relationship between velocity and
    the "unexplained" torque (RNEA - gravity).
    
    Returns:
        (viscous_coefficients, r_squared): Friction coefficients and fit quality
    """
    n_joints = velocities.shape[1]
    viscous_coeff = np.zeros(n_joints)
    r_squared = np.zeros(n_joints)
    
    for i in range(n_joints):
        vel = velocities[:, i]
        
        if rnea_torques is not None and gravity_torques is not None:
            # Unexplained torque = RNEA - gravity (should be inertia + friction)
            unexplained = rnea_torques[:, i] - gravity_torques[:, i]
        else:
            # Without RNEA, we can't estimate viscous friction well
            continue
        
        # Filter to samples with significant velocity
        mask = np.abs(vel) > 0.05
        if np.sum(mask) < 20:
            continue
        
        vel_filtered = vel[mask]
        unexplained_filtered = unexplained[mask]
        
        # Linear regression: unexplained ≈ B * vel + offset
        # The offset captures inertia effects, B is viscous friction
        try:
            coeffs = np.polyfit(vel_filtered, unexplained_filtered, 1)
            viscous_coeff[i] = coeffs[0]  # Slope = viscous coefficient
            
            # Compute R²
            predicted = np.polyval(coeffs, vel_filtered)
            ss_res = np.sum((unexplained_filtered - predicted) ** 2)
            ss_tot = np.sum((unexplained_filtered - np.mean(unexplained_filtered)) ** 2)
            r_squared[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except Exception:
            pass
    
    return viscous_coeff, r_squared


def estimate_static_friction_threshold(velocities: np.ndarray, 
                                        gravity_torques: np.ndarray) -> np.ndarray:
    """
    Estimate static friction (stiction) threshold.
    
    This is the minimum torque needed to start motion. We estimate it by
    looking at the gravity torque at positions where the arm is NOT moving
    despite having gravity load.
    
    Returns:
        static_friction: Array of static friction thresholds per joint
    """
    n_joints = velocities.shape[1]
    static_friction = np.zeros(n_joints)
    
    for i in range(n_joints):
        vel = velocities[:, i]
        grav = gravity_torques[:, i]
        
        # Find samples where arm is stationary (low velocity)
        stationary = np.abs(vel) < 0.02
        
        if np.sum(stationary) > 10:
            # The gravity torque at stationary positions is balanced by friction
            # Static friction ≈ max |gravity| at stationary positions
            static_friction[i] = np.percentile(np.abs(grav[stationary]), 95)
    
    return static_friction


def analyze_gravity_model_accuracy(data: dict, urdf_path: Optional[str] = None) -> Dict:
    """
    Analyze how well the URDF gravity model matches observed behavior.
    
    At truly static positions, the arm should not drift if gravity comp is perfect.
    Any drift indicates model error.
    """
    positions = data['joint_positions']
    velocities = data['joint_velocities']
    gravity = data.get('urdf_gravity_torques')
    
    if gravity is None:
        return {'error': 'No gravity data available'}
    
    n_joints = positions.shape[1]
    
    # Find static segments
    is_static = np.all(np.abs(velocities) < 0.05, axis=1)
    
    results = {
        'joints': [],
        'summary': {}
    }
    
    for i in range(n_joints):
        joint_result = {
            'joint': i + 1,
            'gravity_at_static': [],
            'drift_tendency': 0,
        }
        
        # Analyze gravity at static positions
        static_gravity = gravity[is_static, i]
        if len(static_gravity) > 10:
            joint_result['gravity_at_static_mean'] = float(np.mean(static_gravity))
            joint_result['gravity_at_static_std'] = float(np.std(static_gravity))
            
            # If gravity is non-zero at static positions, the arm should drift
            # unless friction is holding it. This is expected.
            joint_result['max_static_gravity'] = float(np.max(np.abs(static_gravity)))
        
        results['joints'].append(joint_result)
    
    # Overall assessment
    total_static_gravity = np.sum(np.abs(gravity[is_static]), axis=1)
    results['summary'] = {
        'total_samples': len(positions),
        'static_samples': int(np.sum(is_static)),
        'avg_total_gravity_at_static': float(np.mean(total_static_gravity)) if len(total_static_gravity) > 0 else 0,
    }
    
    return results


def generate_compensation_parameters(data: dict) -> Dict:
    """
    Generate improved compensation parameters from recorded data.
    
    Returns a dictionary suitable for use in gravity_compensation.py
    """
    velocities = data['joint_velocities']
    accelerations = data.get('joint_accelerations', np.zeros_like(velocities))
    gravity = data.get('urdf_gravity_torques')
    rnea = data.get('urdf_rnea_torques')
    
    n_joints = velocities.shape[1]
    
    # Estimate friction parameters
    coulomb, coulomb_conf = estimate_coulomb_friction(velocities, accelerations, gravity)
    viscous, viscous_r2 = estimate_viscous_friction(velocities, accelerations, gravity, rnea)
    static = estimate_static_friction_threshold(velocities, gravity) if gravity is not None else np.zeros(n_joints)
    
    params = {
        'friction_compensation': {
            'coulomb_friction': coulomb.tolist(),
            'coulomb_confidence': coulomb_conf.tolist(),
            'viscous_friction': viscous.tolist(),
            'viscous_r_squared': viscous_r2.tolist(),
            'static_friction_threshold': static.tolist(),
        },
        'recommended_settings': {
            'friction_feedforward': [],
            'velocity_deadband': [],
        }
    }
    
    # Generate recommendations
    for i in range(n_joints):
        # Friction feedforward = Coulomb friction (if confident)
        if coulomb_conf[i] > 0.3:
            ff = abs(coulomb[i])
        else:
            ff = static[i] * 0.5  # Conservative estimate
        params['recommended_settings']['friction_feedforward'].append(round(ff, 4))
        
        # Velocity deadband based on static friction
        # Don't apply friction comp until velocity exceeds this
        deadband = 0.02 if static[i] < 0.01 else 0.05
        params['recommended_settings']['velocity_deadband'].append(deadband)
    
    return params


def print_results(data: dict, params: Dict, gravity_analysis: Dict):
    """Print analysis results."""
    n_joints = data['joint_velocities'].shape[1]
    
    print("\n" + "="*70)
    print("FRICTION ESTIMATION RESULTS")
    print("="*70)
    
    fc = params['friction_compensation']
    
    print(f"\n{'Joint':<8} {'Coulomb':>10} {'Conf':>8} {'Viscous':>10} {'R²':>8} {'Static':>10}")
    print("-"*70)
    
    for i in range(n_joints):
        print(f"Joint {i+1:<2} {fc['coulomb_friction'][i]:>10.4f} "
              f"{fc['coulomb_confidence'][i]:>8.2f} "
              f"{fc['viscous_friction'][i]:>10.4f} "
              f"{fc['viscous_r_squared'][i]:>8.2f} "
              f"{fc['static_friction_threshold'][i]:>10.4f}")
    
    print("\n" + "-"*70)
    print("Interpretation:")
    print("  Coulomb: Constant friction opposing motion (Nm)")
    print("  Viscous: Friction proportional to velocity (Nm·s/rad)")
    print("  Static:  Minimum torque to start motion (Nm)")
    print("-"*70)
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDED COMPENSATION PARAMETERS")
    print("="*70)
    
    rec = params['recommended_settings']
    print(f"\nFriction feedforward torques (add to gravity comp):")
    ff_values = [round(float(x), 4) for x in rec['friction_feedforward']]
    print(f"  {ff_values}")
    
    print(f"\nVelocity deadbands (don't apply friction comp below this):")
    print(f"  {rec['velocity_deadband']}")
    
    # Gravity model analysis
    if 'error' not in gravity_analysis:
        print("\n" + "="*70)
        print("GRAVITY MODEL VALIDATION")
        print("="*70)
        
        print(f"\nStatic samples analyzed: {gravity_analysis['summary']['static_samples']}")
        print(f"Average total gravity at static positions: "
              f"{gravity_analysis['summary']['avg_total_gravity_at_static']:.4f} Nm")
        
        print(f"\n{'Joint':<8} {'Mean Gravity':>14} {'Std':>10} {'Max |Gravity|':>14}")
        print("-"*50)
        for j in gravity_analysis['joints']:
            if 'gravity_at_static_mean' in j:
                print(f"Joint {j['joint']:<2} {j['gravity_at_static_mean']:>14.4f} "
                      f"{j['gravity_at_static_std']:>10.4f} "
                      f"{j['max_static_gravity']:>14.4f}")
    
    # Suggested improvements
    print("\n" + "="*70)
    print("SUGGESTED IMPROVEMENTS FOR GRAVITY COMPENSATION")
    print("="*70)
    
    print("""
1. ADD FRICTION FEEDFORWARD:
   When the arm is commanded to move, add the Coulomb friction torque
   in the direction of motion to help overcome static friction:
   
   τ_cmd = τ_gravity + τ_friction_feedforward * sign(dq_desired)
   
2. ADD VISCOUS DAMPING COMPENSATION:
   For smoother motion, compensate for velocity-dependent friction:
   
   τ_cmd = τ_gravity + τ_viscous * dq_measured
   
3. IMPLEMENT FRICTION MODEL:
   Full friction compensation:
   
   τ_friction = τ_coulomb * sign(dq) + τ_viscous * dq
   
   Apply this when |dq| > velocity_deadband

4. CONSIDER ADDING TO YOUR CONFIG:
""")
    
    print("   # Friction compensation parameters (estimated from physical data)")
    ff_values = [round(float(x), 4) for x in rec['friction_feedforward']]
    print(f"   friction_feedforward: {ff_values}")
    print(f"   viscous_friction: {[round(v, 4) for v in fc['viscous_friction']]}")
    print(f"   velocity_deadband: {rec['velocity_deadband']}")
    

def plot_friction_analysis(data: dict, params: Dict, output_path: Optional[str] = None):
    """Plot friction analysis visualizations."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting.")
        return
    
    velocities = data['joint_velocities']
    gravity = data.get('urdf_gravity_torques')
    rnea = data.get('urdf_rnea_torques')
    
    if gravity is None:
        print("No gravity data for plotting.")
        return
    
    n_joints = velocities.shape[1]
    
    fig, axes = plt.subplots(2, n_joints, figsize=(3*n_joints, 8))
    
    for i in range(n_joints):
        vel = velocities[:, i]
        grav = gravity[:, i]
        
        # Top row: Velocity vs Gravity (shows friction asymmetry)
        ax1 = axes[0, i]
        ax1.scatter(vel, grav, s=1, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Velocity (rad/s)')
        ax1.set_ylabel('Gravity Torque (Nm)')
        ax1.set_title(f'Joint {i+1}: Vel vs Gravity')
        ax1.grid(True, alpha=0.3)
        
        # Bottom row: Velocity vs (RNEA - Gravity) shows friction + inertia
        ax2 = axes[1, i]
        if rnea is not None:
            unexplained = rnea[:, i] - grav
            ax2.scatter(vel, unexplained, s=1, alpha=0.3, c='orange')
            
            # Add linear fit line
            mask = np.abs(vel) > 0.05
            if np.sum(mask) > 20:
                coeffs = np.polyfit(vel[mask], unexplained[mask], 1)
                vel_range = np.linspace(vel.min(), vel.max(), 100)
                ax2.plot(vel_range, np.polyval(coeffs, vel_range), 'r-', 
                        label=f'Viscous: {coeffs[0]:.3f} Nm·s/rad')
                ax2.legend(fontsize=8)
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Velocity (rad/s)')
        ax2.set_ylabel('RNEA - Gravity (Nm)')
        ax2.set_title(f'Joint {i+1}: Friction + Inertia')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved friction analysis plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate friction compensation parameters from recorded data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'recording', type=str,
        help='Path to recording file (.npz)'
    )
    parser.add_argument(
        '--urdf', type=str, default=None,
        help='Path to URDF file for gravity model validation'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for compensation parameters'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading recording: {args.recording}")
    data = load_recording(args.recording)
    
    print(f"Loaded {len(data['timestamps'])} samples")
    print(f"Available fields: {list(data.keys())}")
    
    # Check required fields
    if 'urdf_gravity_torques' not in data:
        print("\n⚠️  Warning: No URDF gravity data in recording.")
        print("   Re-record with --urdf flag for better friction estimation.")
    
    if 'urdf_rnea_torques' not in data:
        print("\n⚠️  Warning: No RNEA data in recording.")
        print("   Viscous friction estimation will be limited.")
    
    # Estimate compensation parameters
    params = generate_compensation_parameters(data)
    
    # Analyze gravity model
    gravity_analysis = analyze_gravity_model_accuracy(data, args.urdf)
    
    # Print results
    print_results(data, params, gravity_analysis)
    
    # Save parameters
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\nSaved compensation parameters to: {args.output}")
    
    # Generate plots
    if args.plot and HAS_MATPLOTLIB:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            base_name = Path(args.recording).stem
            plot_friction_analysis(data, params, str(output_dir / f"{base_name}_friction.png"))
        else:
            plot_friction_analysis(data, params)


if __name__ == '__main__':
    main()
