#!/usr/bin/env python3
"""
GELLO Setup Validation Script

This script helps diagnose and validate the GELLO setup for gravity compensation.
It checks:
1. URDF loading and joint properties
2. Dynamixel motor connectivity and raw positions
3. Calibration offset calculation
4. Expected gravity torques at various poses

Usage:
    python scripts/validate_gello_setup.py --config configs/ur5e_gello_factr_sim.yaml
    
    # Without hardware (URDF check only):
    python scripts/validate_gello_setup.py --config configs/ur5e_gello_factr_sim.yaml --no-hardware
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_urdf(config: dict, config_path: str) -> None:
    """Validate URDF loading and show joint properties."""
    import pinocchio as pin
    
    print("\n" + "="*60)
    print("URDF VALIDATION")
    print("="*60)
    
    urdf_filename = config["arm_teleop"]["leader_urdf"]
    config_dir = Path(config_path).parent
    
    # Try to find URDF
    urdf_path = None
    search_paths = [
        config_dir / urdf_filename,
        Path(__file__).parent.parent / urdf_filename,
        Path(__file__).parent.parent / "gello" / "factr" / "urdf" / Path(urdf_filename).name,
    ]
    
    for sp in search_paths:
        if sp.exists():
            urdf_path = sp
            break
    
    if urdf_path is None:
        print(f"❌ URDF not found! Searched in:")
        for sp in search_paths:
            print(f"   - {sp}")
        return
    
    print(f"✓ URDF found: {urdf_path}")
    
    # Load with Pinocchio
    try:
        model, _, _ = pin.buildModelsFromUrdf(
            filename=str(urdf_path),
            package_dirs=str(urdf_path.parent)
        )
        data = model.createData()
        print(f"✓ Pinocchio model loaded successfully")
        print(f"  - Number of joints (nq): {model.nq}")
        print(f"  - Number of velocities (nv): {model.nv}")
        print(f"  - Number of actuators (nu): {model.njoints - 1}")  # -1 for universe
        
        print("\nJoint Details:")
        print("-" * 80)
        print(f"{'#':<3} {'Name':<25} {'Type':<12} {'Axis':<15} {'Mass (child)':<12}")
        print("-" * 80)
        
        for i in range(1, model.njoints):  # Skip universe joint
            joint = model.joints[i]
            joint_name = model.names[i]
            
            # Get joint axis from model
            # Different joint types have different ways to get axis
            try:
                # For revolute joints
                if hasattr(joint, 'axis'):
                    axis = joint.axis
                else:
                    # Try to infer from joint placement
                    axis = "N/A"
            except Exception:
                axis = "N/A"
            
            # Get mass of child link
            inertia = model.inertias[i]
            mass = inertia.mass
            
            joint_type = str(type(joint).__name__).replace("JointModel", "")
            
            print(f"{i:<3} {joint_name:<25} {joint_type:<12} {str(axis):<15} {mass:.4f} kg")
        
        # Test gravity torques at zero position
        print("\n" + "="*60)
        print("GRAVITY TORQUE TEST (at q=0)")
        print("="*60)
        
        q = np.zeros(model.nq)
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        
        tau = pin.rnea(model, data, q, v, a)
        
        print(f"Joint positions (rad): {[f'{x:.3f}' for x in q]}")
        print(f"Gravity torques (Nm):  {[f'{x:.4f}' for x in tau]}")
        
        # Calculate total mass
        total_mass = sum(model.inertias[i].mass for i in range(1, model.njoints))
        print(f"\nTotal arm mass: {total_mass:.4f} kg")
        
        # Test at different poses
        test_poses = {
            "All zeros": np.zeros(model.nq),
            "Joint 2 at -90°": np.array([0, -np.pi/2, 0, 0, 0, 0, 0][:model.nq]),
            "Joint 2 at +90°": np.array([0, np.pi/2, 0, 0, 0, 0, 0][:model.nq]),
            "Joint 3 at -90°": np.array([0, 0, -np.pi/2, 0, 0, 0, 0][:model.nq]),
        }
        
        print("\nGravity torques at different poses:")
        print("-" * 80)
        max_torque_overall = 0.0
        for pose_name, q in test_poses.items():
            tau = pin.rnea(model, data, q, v, a)
            max_tau = np.max(np.abs(tau[:6]))
            max_torque_overall = max(max_torque_overall, max_tau)
            print(f"{pose_name}:")
            print(f"  τ = {[f'{x:+.4f}' for x in tau[:6]]} Nm (max: {max_tau:.4f} Nm)")
        
        # Analysis and recommendations
        print("\n" + "="*60)
        print("GRAVITY COMPENSATION ANALYSIS")
        print("="*60)
        print(f"Maximum gravity torque across all test poses: {max_torque_overall:.4f} Nm")
        
        # Estimate required gain based on typical friction
        # Dynamixel XC330 has ~0.1-0.2 Nm static friction at the output
        # XM430 has ~0.2-0.4 Nm
        estimated_friction = 0.15  # Conservative estimate per joint
        recommended_gain = max(1.0, (max_torque_overall + 3 * estimated_friction) / max_torque_overall) if max_torque_overall > 0.01 else 5.0
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        if max_torque_overall < 0.5:
            print(f"   - Your GELLO is very lightweight ({total_mass:.2f} kg)")
            print(f"   - Gravity torques are small (max {max_torque_overall:.3f} Nm)")
            print(f"   - Friction may dominate over gravity!")
            print(f"   - Recommended gravity_comp.gain: {recommended_gain:.1f} - {recommended_gain * 1.5:.1f}")
        else:
            print(f"   - Gravity torques look reasonable (max {max_torque_overall:.3f} Nm)")
            print(f"   - Recommended gravity_comp.gain: {recommended_gain:.1f}")
        
        print(f"\n   Current config has gain: 2.3")
        print(f"   → Max applied torque: {max_torque_overall * 2.3:.3f} Nm")
        
    except Exception as e:
        print(f"❌ Failed to load URDF with Pinocchio: {e}")
        import traceback
        traceback.print_exc()


def validate_dynamixel(config: dict) -> Optional[np.ndarray]:
    """Validate Dynamixel connection and read positions."""
    from gello.dynamixel.driver import DynamixelDriver
    
    print("\n" + "="*60)
    print("DYNAMIXEL VALIDATION")
    print("="*60)
    
    port_config = config["dynamixel"]["dynamixel_port"]
    if port_config.startswith("/"):
        port = port_config
    else:
        port = "/dev/serial/by-id/" + port_config
    
    servo_types = config["dynamixel"]["servo_types"]
    joint_signs = config["dynamixel"]["joint_signs"]
    baudrate = config["dynamixel"].get("baudrate", 57600)
    
    print(f"Port: {port}")
    print(f"Baudrate: {baudrate}")
    print(f"Servo types: {servo_types}")
    print(f"Joint signs: {joint_signs}")
    
    if not os.path.exists(port):
        print(f"❌ Port {port} does not exist!")
        print("   Available ports:")
        try:
            for p in Path("/dev/serial/by-id/").iterdir():
                print(f"   - {p.name}")
        except Exception:
            print("   - Could not list available ports")
        return None
    
    try:
        joint_ids = list(range(1, len(servo_types) + 1))
        driver = DynamixelDriver(
            ids=joint_ids,
            servo_types=servo_types,
            port=port,
            baudrate=baudrate
        )
        print(f"✓ Connected to {len(joint_ids)} Dynamixel servos")
        
        # Read positions multiple times for stability
        positions = []
        for _ in range(10):
            pos, vel = driver.get_positions_and_velocities()
            positions.append(pos)
        
        avg_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        print("\nRaw Dynamixel Readings:")
        print("-" * 60)
        for i, (pos, std, sign) in enumerate(zip(avg_pos, std_pos, joint_signs)):
            print(f"  Motor {i+1}: {np.rad2deg(pos):+8.2f}° (±{np.rad2deg(std):.2f}°), sign={sign:+.0f}")
        
        driver.close()
        return avg_pos
        
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_calibration(config: dict, raw_positions: np.ndarray) -> None:
    """Validate calibration offset calculation."""
    print("\n" + "="*60)
    print("CALIBRATION VALIDATION")
    print("="*60)
    
    joint_signs = np.array(config["dynamixel"]["joint_signs"], dtype=float)
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    calibration_joint_pos = np.array(
        config["arm_teleop"]["initialization"]["calibration_joint_pos"]
    )
    
    print(f"Expected calibration pose: {[f'{np.rad2deg(x):.1f}°' for x in calibration_joint_pos]}")
    print(f"Raw motor positions: {[f'{np.rad2deg(x):.1f}°' for x in raw_positions[:num_arm_joints]]}")
    
    # Calculate offsets with fine resolution
    print("\nCalculating offsets (searching every 5°)...")
    offsets = []
    
    for i in range(num_arm_joints):
        best_offset = 0
        best_error = 1e9
        target = calibration_joint_pos[i]
        sign = joint_signs[i]
        raw = raw_positions[i]
        
        for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
            calibrated = sign * (raw - offset)
            error = abs(calibrated - target)
            if error < best_error:
                best_error = error
                best_offset = offset
        
        offsets.append(best_offset)
        calibrated_pos = sign * (raw - best_offset)
        
        print(f"  Joint {i+1}: offset={np.rad2deg(best_offset):+7.1f}° ({best_offset/np.pi:.2f}π), "
              f"result={np.rad2deg(calibrated_pos):+7.1f}° (error={np.rad2deg(best_error):.2f}°)")
    
    # Show final calibrated pose
    calibrated_pose = joint_signs[:num_arm_joints] * (raw_positions[:num_arm_joints] - np.array(offsets))
    print(f"\nFinal calibrated pose: {[f'{np.rad2deg(x):+.1f}°' for x in calibrated_pose]}")
    print(f"Target pose:           {[f'{np.rad2deg(x):+.1f}°' for x in calibration_joint_pos]}")
    
    max_error = np.max(np.abs(calibrated_pose - calibration_joint_pos))
    print(f"Max error: {np.rad2deg(max_error):.2f}°")
    
    if max_error > np.deg2rad(10):
        print("\n⚠️  WARNING: Large calibration error!")
        print("   Possible causes:")
        print("   1. The GELLO is not in the expected calibration_joint_pos")
        print("   2. Joint signs in config don't match the actual motor directions")
        print("   3. The URDF joint directions don't match the physical setup")


def validate_torque_mapping(config: dict) -> None:
    """Validate torque-to-current mapping for motors."""
    from gello.dynamixel.driver import TORQUE_TO_CURRENT_MAPPING, SERVO_CURRENT_LIMITS
    
    print("\n" + "="*60)
    print("TORQUE-TO-CURRENT MAPPING")
    print("="*60)
    
    servo_types = config["dynamixel"]["servo_types"]
    
    print("Configured servo types and their limits:")
    print("-" * 60)
    
    for i, servo_type in enumerate(servo_types):
        if servo_type in TORQUE_TO_CURRENT_MAPPING:
            torque_const = TORQUE_TO_CURRENT_MAPPING[servo_type]
            current_limit = SERVO_CURRENT_LIMITS.get(servo_type, "N/A")
            # Calculate max torque (current_limit / torque_const gives max torque in Nm)
            if isinstance(current_limit, (int, float)):
                max_torque = current_limit / torque_const
                print(f"  Motor {i+1} ({servo_type}):")
                print(f"    Torque constant: {torque_const:.2f} (mA/Nm)")
                print(f"    Current limit: {current_limit} mA")
                print(f"    Max torque: {max_torque:.3f} Nm")
            else:
                print(f"  Motor {i+1} ({servo_type}): Mapping found, no current limit")
        else:
            print(f"  Motor {i+1} ({servo_type}): ❌ NO MAPPING FOUND!")
            print(f"    Available mappings: {list(TORQUE_TO_CURRENT_MAPPING.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Validate GELLO Setup")
    parser.add_argument(
        "--config", "-c",
        default="configs/ur5e_gello_factr_sim.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Skip hardware tests (URDF validation only)"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    print("="*60)
    print("GELLO SETUP VALIDATION")
    print("="*60)
    print(f"Config: {args.config}")
    
    config = load_config(args.config)
    print(f"Name: {config.get('name', 'N/A')}")
    
    # Always validate URDF
    validate_urdf(config, args.config)
    
    # Validate torque mapping
    validate_torque_mapping(config)
    
    # Optionally validate hardware
    if not args.no_hardware:
        raw_positions = validate_dynamixel(config)
        if raw_positions is not None:
            validate_calibration(config, raw_positions)
    else:
        print("\n⏭️  Skipping hardware tests (--no-hardware)")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
