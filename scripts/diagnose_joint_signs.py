#!/usr/bin/env python3
"""
GELLO Joint Sign Diagnostic Tool

This script helps determine the correct joint_signs for each motor.
It applies a small positive torque to each joint individually and shows
which direction the joint moves.

Usage:
    python scripts/diagnose_joint_signs.py --config configs/ur5e_gello_factr_sim.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver


def diagnose_joint_signs(config_path: str):
    """Test each joint to determine correct signs."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("GELLO JOINT SIGN DIAGNOSTIC")
    print("="*70)
    print("\nThis tool will apply a small POSITIVE torque to each joint")
    print("and show which direction it moves. Use this to verify joint_signs.")
    print("\nExpected behavior:")
    print("  - Positive torque should move joint in POSITIVE direction")
    print("  - If joint moves in NEGATIVE direction, the sign is WRONG")
    print("\n" + "="*70)
    
    # Setup Dynamixel
    port_config = config["dynamixel"]["dynamixel_port"]
    if port_config.startswith("/"):
        port = port_config
    else:
        port = "/dev/serial/by-id/" + port_config
    
    servo_types = config["dynamixel"]["servo_types"]
    joint_signs = np.array(config["dynamixel"]["joint_signs"], dtype=float)
    baudrate = config["dynamixel"].get("baudrate", 1000000)
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    
    print(f"\nConfig: {config['name']}")
    print(f"Port: {port}")
    print(f"Number of arm joints: {num_arm_joints}")
    print(f"Current joint_signs: {joint_signs.tolist()}")
    
    # Connect to motors
    joint_ids = list(range(1, len(servo_types) + 1))
    try:
        driver = DynamixelDriver(
            ids=joint_ids,
            servo_types=servo_types,
            port=port,
            baudrate=baudrate
        )
        print(f"✓ Connected to {len(joint_ids)} motors")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return 1
    
    # Switch to current control mode
    try:
        driver.set_torque_mode(False)
        driver.set_operating_mode(0)  # Current control
        driver.set_torque_mode(True)
        print("✓ Motors in current control mode")
    except Exception as e:
        print(f"⚠️  Warning: Could not set current control mode: {e}")
    
    input("\n⚠️  WARNING: The arm will move! Make sure it's in a safe position.")
    input("Press ENTER to start testing each joint...")
    
    # Test parameters
    test_torque_nm = 0.15  # Small test torque in Nm
    test_duration = 1.5    # seconds
    
    print("\n" + "="*70)
    print("TESTING JOINTS")
    print("="*70)
    
    recommended_signs = []
    
    for joint_idx in range(num_arm_joints):
        motor_id = joint_idx + 1
        print(f"\n{'='*70}")
        print(f"Testing Joint {motor_id} (Motor ID {motor_id})")
        print(f"  Current sign: {joint_signs[joint_idx]:+.0f}")
        print(f"{'='*70}")
        
        # Record initial position
        pos_before, _ = driver.get_positions_and_velocities()
        pos_before_joint = pos_before[joint_idx]
        
        print(f"  Initial raw motor position: {np.rad2deg(pos_before_joint):+7.2f}°")
        print(f"  Applying +{test_torque_nm:.3f} Nm for {test_duration} seconds...")
        
        # Apply positive torque
        torque_array = np.zeros(len(servo_types))
        # Apply torque with current sign to motor
        torque_array[joint_idx] = test_torque_nm * joint_signs[joint_idx]
        
        try:
            driver.set_torque(torque_array.tolist())
            time.sleep(test_duration)
            
            # Stop torque
            torque_array[:] = 0
            driver.set_torque(torque_array.tolist())
            
            # Record final position
            time.sleep(0.1)
            pos_after, _ = driver.get_positions_and_velocities()
            pos_after_joint = pos_after[joint_idx]
            
            # Calculate movement
            raw_delta = pos_after_joint - pos_before_joint
            raw_delta_deg = np.rad2deg(raw_delta)
            
            # Apply current sign to see effective movement
            effective_delta = joint_signs[joint_idx] * raw_delta
            effective_delta_deg = np.rad2deg(effective_delta)
            
            print(f"  Final raw motor position: {np.rad2deg(pos_after_joint):+7.2f}°")
            print(f"  Raw motor movement: {raw_delta_deg:+7.2f}°")
            print(f"  Effective joint movement (with current sign): {effective_delta_deg:+7.2f}°")
            
            # Determine if sign is correct
            if abs(raw_delta_deg) < 0.5:
                print(f"  ⚠️  WARNING: Joint barely moved! Check if motor is working.")
                recommended_sign = joint_signs[joint_idx]  # Keep current
            elif effective_delta > 0:
                print(f"  ✓ CORRECT: Positive torque → Positive joint movement")
                recommended_sign = joint_signs[joint_idx]
            else:
                print(f"  ❌ WRONG: Positive torque → Negative joint movement")
                print(f"     The sign should be INVERTED!")
                recommended_sign = -joint_signs[joint_idx]
            
            recommended_signs.append(recommended_sign)
            
        except Exception as e:
            print(f"  ❌ Error during test: {e}")
            recommended_signs.append(joint_signs[joint_idx])
        
        # Wait before next test
        if joint_idx < num_arm_joints - 1:
            input(f"\nPress ENTER to test next joint...")
    
    # Add gripper sign (unchanged)
    if len(joint_signs) > num_arm_joints:
        recommended_signs.append(joint_signs[-1])
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    print("\nCurrent joint_signs:")
    print(f"  {joint_signs.tolist()}")
    
    print("\nRecommended joint_signs:")
    print(f"  {recommended_signs}")
    
    changes_needed = []
    for i in range(num_arm_joints):
        if recommended_signs[i] != joint_signs[i]:
            changes_needed.append(f"Joint {i+1}: {joint_signs[i]:+.0f} → {recommended_signs[i]:+.0f}")
    
    if changes_needed:
        print("\n⚠️  CHANGES NEEDED:")
        for change in changes_needed:
            print(f"  - {change}")
        
        print("\nUpdate your config file:")
        print(f"  joint_signs: {recommended_signs}")
    else:
        print("\n✓ All joint signs are correct!")
    
    # Cleanup
    driver.set_torque_mode(False)
    driver.close()
    
    print("\n" + "="*70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Diagnose GELLO joint signs")
    parser.add_argument(
        "--config", "-c",
        default="configs/ur5e_gello_factr_sim.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    return diagnose_joint_signs(args.config)


if __name__ == "__main__":
    sys.exit(main())
