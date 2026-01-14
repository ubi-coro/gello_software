#!/usr/bin/env python3
"""
GELLO Calibration Offset Finder

This script helps find the correct calibration offset for the shoulder joint
by finding where the gravity torque is actually zero (the true equilibrium point).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from gello.dynamixel.driver import DynamixelDriver


def main():
    parser = argparse.ArgumentParser(description="Find true shoulder calibration offset")
    parser.add_argument("--config", "-c", default="../configs/ur5e_gello_factr_hw.yaml")
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        script_dir = Path(__file__).parent
        config_path = str(script_dir / config_path)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load URDF
    urdf_filename = config["arm_teleop"]["leader_urdf"]
    workspace_root = Path(__file__).parent.parent.resolve()
    
    candidates = [
        Path(urdf_filename),
        Path(config_path).parent / urdf_filename,
        workspace_root / urdf_filename,
    ]
    
    urdf_path = None
    for c in candidates:
        if c.exists() and c.is_file():
            urdf_path = c.resolve()
            break
    
    if urdf_path is None:
        filename = Path(urdf_filename).name
        matches = list(workspace_root.glob(f"**/{filename}"))
        if matches:
            urdf_path = matches[0].resolve()
    
    if urdf_path is None:
        raise FileNotFoundError(f"Could not find URDF: {urdf_filename}")
    
    print(f"Loading URDF: {urdf_path}")
    model, _, _ = pin.buildModelsFromUrdf(str(urdf_path), str(urdf_path.parent))
    data = model.createData()
    
    # Parameters
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    joint_signs = np.array(config["dynamixel"]["joint_signs"], dtype=float)
    calibration_pos = np.array(config["arm_teleop"]["initialization"]["calibration_joint_pos"])
    
    gc_cfg = config.get("controller", {}).get("gravity_comp", {})
    gc_gain = gc_cfg.get("gain", 1.0)
    gc_per_joint = np.array(gc_cfg.get("gain_per_joint", [1.0] * num_arm_joints))
    
    # Connect to hardware
    print("\nConnecting to Dynamixel...")
    servo_types = config["dynamixel"]["servo_types"]
    port = config["dynamixel"]["dynamixel_port"]
    if not port.startswith("/"):
        port = "/dev/serial/by-id/" + port
    baudrate = config["dynamixel"].get("baudrate", 57600)
    
    driver = DynamixelDriver(
        list(range(1, len(servo_types) + 1)),
        servo_types,
        port,
        baudrate=baudrate
    )
    driver.set_torque_mode(False)
    
    # Standard calibration (as in gravity_compensation.py)
    print("\nRunning standard calibration...")
    for _ in range(10):
        driver.get_positions_and_velocities()
    
    raw_pos, _ = driver.get_positions_and_velocities()
    
    standard_offsets = []
    for i in range(num_arm_joints):
        best_offset = 0
        best_error = 1e9
        for offset in np.linspace(-10*np.pi, 10*np.pi, 721):
            calibrated = joint_signs[i] * (raw_pos[i] - offset)
            error = abs(calibrated - calibration_pos[i])
            if error < best_error:
                best_error = error
                best_offset = offset
        standard_offsets.append(best_offset)
    standard_offsets = np.array(standard_offsets)
    
    def get_calibrated_pos(raw, offsets):
        return (raw[:num_arm_joints] - offsets) * joint_signs[:num_arm_joints]
    
    def compute_gravity_torque(joint_pos):
        q = np.zeros(model.nq)
        q[:num_arm_joints] = joint_pos
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        tau = pin.rnea(model, data, q, v, a)
        tau_arm = np.array(tau[:num_arm_joints])
        tau_arm *= gc_gain
        tau_arm *= gc_per_joint
        return tau_arm
    
    # Current state with standard calibration
    joint_pos = get_calibrated_pos(raw_pos, standard_offsets)
    tau = compute_gravity_torque(joint_pos)
    
    print("\n" + "="*70)
    print("CURRENT CALIBRATION ANALYSIS")
    print("="*70)
    print(f"\nRaw Dynamixel positions (rad): {[f'{x:.4f}' for x in raw_pos[:num_arm_joints]]}")
    print(f"Standard offsets (rad):        {[f'{x:.4f}' for x in standard_offsets]}")
    print(f"Calibrated positions (deg):    {[f'{np.rad2deg(x):+.1f}' for x in joint_pos]}")
    print(f"Gravity torques (Nm):          {[f'{x:+.4f}' for x in tau]}")
    
    # Find where τ₂ = 0 by searching different offsets
    print("\n" + "="*70)
    print("SEARCHING FOR TRUE EQUILIBRIUM (where τ₂ = 0)")
    print("="*70)
    
    # We'll vary the shoulder offset and find where τ₂ crosses zero
    shoulder_idx = 1
    original_offset = standard_offsets[shoulder_idx]
    
    print(f"\nVarying shoulder offset around {np.rad2deg(original_offset):.1f}°...")
    
    # Search range: ±45° around current offset
    test_offsets = np.linspace(original_offset - np.deg2rad(45), 
                                original_offset + np.deg2rad(45), 
                                181)  # 0.5° resolution
    
    results = []
    for test_offset in test_offsets:
        test_offsets_full = standard_offsets.copy()
        test_offsets_full[shoulder_idx] = test_offset
        
        test_pos = get_calibrated_pos(raw_pos, test_offsets_full)
        test_tau = compute_gravity_torque(test_pos)
        
        results.append({
            'offset': test_offset,
            'theta2': test_pos[shoulder_idx],
            'tau2': test_tau[shoulder_idx],
        })
    
    # Find zero crossing
    tau_values = np.array([r['tau2'] for r in results])
    zero_crossings = []
    
    for i in range(len(tau_values) - 1):
        if tau_values[i] * tau_values[i+1] < 0:  # Sign change
            # Linear interpolation to find exact crossing
            t1, t2 = tau_values[i], tau_values[i+1]
            o1, o2 = results[i]['offset'], results[i+1]['offset']
            zero_offset = o1 - t1 * (o2 - o1) / (t2 - t1)
            
            # Calculate corresponding theta2
            test_offsets_full = standard_offsets.copy()
            test_offsets_full[shoulder_idx] = zero_offset
            test_pos = get_calibrated_pos(raw_pos, test_offsets_full)
            
            zero_crossings.append({
                'offset': zero_offset,
                'theta2': test_pos[shoulder_idx],
            })
    
    print("\nFound equilibrium points (where τ₂ ≈ 0):")
    print("-"*70)
    
    for i, zc in enumerate(zero_crossings):
        theta2_deg = np.rad2deg(zc['theta2'])
        offset_deg = np.rad2deg(zc['offset'])
        offset_diff = np.rad2deg(zc['offset'] - original_offset)
        
        # Check if this is near 180° (expected singularity)
        dist_from_180 = min(abs(theta2_deg - 180), abs(theta2_deg + 180), 
                           abs(theta2_deg - 0), abs(theta2_deg + 360))
        
        is_expected = dist_from_180 < 10
        marker = "✓ EXPECTED" if is_expected else "✗ UNEXPECTED"
        
        print(f"  Crossing {i+1}: θ₂ = {theta2_deg:+.1f}° at offset = {offset_deg:+.1f}° "
              f"(diff from current: {offset_diff:+.1f}°) [{marker}]")
    
    # Calculate the correction needed
    print("\n" + "="*70)
    print("RECOMMENDED CORRECTION")
    print("="*70)
    
    # Find the crossing closest to 180° (or 0° for the other equilibrium)
    best_correction = None
    best_target = None
    
    for zc in zero_crossings:
        theta2_deg = np.rad2deg(zc['theta2'])
        
        # Check distance to expected equilibria (180° and 0°)
        dist_to_180 = min(abs(theta2_deg - 180), abs(theta2_deg + 180))
        dist_to_0 = abs(theta2_deg)
        
        if dist_to_180 < 30:  # This should be the 180° equilibrium
            correction_needed = 180 - theta2_deg
            if best_correction is None or abs(correction_needed) < abs(best_correction):
                best_correction = correction_needed
                best_target = 180
        elif dist_to_0 < 30:  # This should be the 0° equilibrium
            correction_needed = 0 - theta2_deg
            if best_correction is None or abs(correction_needed) < abs(best_correction):
                best_correction = correction_needed
                best_target = 0
    
    if best_correction is not None:
        print(f"\nThe equilibrium that should be at {best_target}° is actually at "
              f"{best_target - best_correction:.1f}°")
        print(f"This means the calibration is off by approximately {best_correction:+.1f}°")
        
        # Calculate new calibration position
        current_calib = calibration_pos[shoulder_idx]
        new_calib = current_calib + np.deg2rad(best_correction)
        
        print(f"\n>>> RECOMMENDED FIX:")
        print(f"    In your YAML config, change calibration_joint_pos[1] from:")
        print(f"      {current_calib:.4f} rad ({np.rad2deg(current_calib):.1f}°)")
        print(f"    to:")
        print(f"      {new_calib:.4f} rad ({np.rad2deg(new_calib):.1f}°)")
        
        print(f"\n    Or adjust the URDF joint offset by {best_correction:+.1f}°")
    else:
        print("\nCould not find a clear correction. The equilibrium might be correct,")
        print("or the arm is not currently near the singularity region.")
    
    # Interactive verification
    print("\n" + "="*70)
    print("INTERACTIVE VERIFICATION")
    print("="*70)
    print("\nMove the shoulder joint to where it naturally balances (τ₂ ≈ 0).")
    print("The script will tell you what angle that corresponds to.")
    print("Press Ctrl+C when done.\n")
    
    try:
        while True:
            raw_pos, _ = driver.get_positions_and_velocities()
            joint_pos = get_calibrated_pos(raw_pos, standard_offsets)
            tau = compute_gravity_torque(joint_pos)
            
            theta2_deg = np.rad2deg(joint_pos[shoulder_idx])
            tau2 = tau[shoulder_idx]
            
            # Color code based on torque magnitude
            if abs(tau2) < 0.01:
                status = "🟢 EQUILIBRIUM"
            elif abs(tau2) < 0.05:
                status = "🟡 NEAR EQUILIBRIUM"
            else:
                status = "🔴 NOT EQUILIBRIUM"
            
            print(f"\r{status} | θ₂ = {theta2_deg:+7.1f}° | τ₂ = {tau2:+.4f} Nm | "
                  f"Expected equilibrium: ±180° or 0°", end="", flush=True)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nDone!")
    finally:
        driver.close()


if __name__ == "__main__":
    main()