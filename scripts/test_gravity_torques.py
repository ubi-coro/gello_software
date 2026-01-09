#!/usr/bin/env python3
"""
Simple gravity torque test - shows what torques are being computed and applied.

Usage:
    python scripts/test_gravity_torques.py --config configs/ur5e_gello_factr_sim.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver


def test_gravity_torques(config_path: str):
    """Test gravity compensation in real-time."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("GRAVITY TORQUE REAL-TIME TEST")
    print("="*70)
    
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
    gravity_gain = config["controller"]["gravity_comp"]["gain"]
    
    # Load URDF
    urdf_path = config["arm_teleop"]["leader_urdf"]
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(os.path.dirname(config_path), "..", urdf_path)
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    print(f"\nConfig: {config['name']}")
    print(f"URDF: {urdf_path}")
    print(f"Gravity compensation gain: {gravity_gain}")
    print(f"Number of arm joints: {num_arm_joints}")
    
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
    
    # Get current positions for calibration
    print("\n" + "="*70)
    print("CALIBRATION")
    print("="*70)
    print("Move the arm to the ZERO position (straight up like a candle)!")
    input("Press ENTER when ready to calibrate offsets...")
    
    raw_pos, _ = driver.get_positions_and_velocities()
    
    # Simple offset calculation (round to nearest 5°)
    offsets = []
    for i in range(num_arm_joints):
        raw_deg = np.rad2deg(raw_pos[i])
        # Round to nearest 5° multiple of 180°
        offset_deg = round(raw_deg / 5.0) * 5.0
        offset_rad = np.deg2rad(offset_deg)
        offsets.append(offset_rad)
        
        result_deg = joint_signs[i] * (raw_deg - offset_deg)
        print(f"  Joint {i+1}: raw={raw_deg:+7.1f}°, offset={offset_deg:+7.1f}°, "
              f"sign={joint_signs[i]:+.0f}, result={result_deg:+6.1f}°")
    
    # Add gripper offset
    offsets.append(raw_pos[-1])
    
    print("\n✓ Calibration complete!")
    print("\n" + "="*70)
    print("REAL-TIME MONITORING")
    print("="*70)
    print("Move the arm slowly and watch the torques...")
    print("Press Ctrl+C to stop")
    print()
    
    # Enable torque mode
    driver.set_torque_mode(False)
    driver.set_operating_mode(0)  # Current control
    driver.set_torque_mode(True)
    
    try:
        iteration = 0
        while True:
            # Get current state
            raw_pos, raw_vel = driver.get_positions_and_velocities()
            
            # Apply offsets and signs
            joint_pos = np.zeros(num_arm_joints)
            joint_vel = np.zeros(num_arm_joints)
            for i in range(num_arm_joints):
                joint_pos[i] = joint_signs[i] * (raw_pos[i] - offsets[i])
                joint_vel[i] = joint_signs[i] * raw_vel[i]
            
            # Compute gravity torque using Pinocchio
            q_pin = np.zeros(model.nq)
            v_pin = np.zeros(model.nv)
            a_pin = np.zeros(model.nv)
            q_pin[:num_arm_joints] = joint_pos
            v_pin[:num_arm_joints] = joint_vel
            
            gravity_torque = pin.rnea(model, data, q_pin, v_pin, a_pin)
            gravity_torque = np.array(gravity_torque[:num_arm_joints])
            
            # Apply gain
            commanded_torque = gravity_gain * gravity_torque
            
            # Apply to motors (with signs)
            motor_torque = joint_signs[:num_arm_joints] * commanded_torque
            
            # Send to motors
            full_torque = np.zeros(len(servo_types))
            full_torque[:num_arm_joints] = motor_torque
            driver.set_torque(full_torque.tolist())
            
            # Print status every 0.5 seconds
            if iteration % 25 == 0:
                print(f"\nJoint positions (deg): {np.rad2deg(joint_pos)}")
                print(f"Gravity τ (Nm):        {gravity_torque}")
                print(f"Commanded τ (Nm):      {commanded_torque}")
                print(f"Motor τ*sign (Nm):     {motor_torque}")
            
            iteration += 1
            time.sleep(0.02)  # 50 Hz
            
    except KeyboardInterrupt:
        print("\n\n✓ Stopping...")
        # Disable torques
        driver.set_torque_mode(False)
        driver.close()
        print("Done!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test gravity torques")
    parser.add_argument(
        "--config", "-c",
        default="configs/ur5e_gello_factr_sim.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    return test_gravity_torques(args.config)


if __name__ == "__main__":
    sys.exit(main())
