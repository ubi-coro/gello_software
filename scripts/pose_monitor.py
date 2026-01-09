#!/usr/bin/env python3
"""
GELLO Pose Monitor - Live Anzeige der Motorpositionen OHNE Kalibrierung

Dieses Tool zeigt kontinuierlich die rohen Motorpositionen an,
sodass du verstehen kannst, welche Werte in welcher Pose zu erwarten sind.

Usage:
    python scripts/pose_monitor.py --config configs/ur5e_gello_factr_sim.yaml
    
    # Mit Referenzpose-Vergleich:
    python scripts/pose_monitor.py --config configs/ur5e_gello_factr_sim.yaml --reference candle
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver

# Vordefinierte Referenzposen (URDF-Koordinaten in Radiant)
REFERENCE_POSES = {
    "candle": {
        "name": "Candle (Kerze)",
        "description": "Arm vollständig gestreckt nach oben",
        "joints": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "diagram": """
      [EE]
       |
      [5]
       |
      [4]
       |
      [3]
       |
      [2]
       |
      [1]
    __|__
   |_____|
        """
    },
    "t_pose": {
        "name": "T-Pose (Horizontal)",
        "description": "Arm horizontal zur Seite gestreckt",
        "joints": np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0]),
        "diagram": """
   [1]---[2]---[3]---[4]---[5]---[EE]
    |
  __|__
 |_____|
        """
    },
    "l_pose": {
        "name": "L-Pose",
        "description": "Shoulder horizontal, Elbow 90° nach oben",
        "joints": np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),
        "diagram": """
      [EE]
       |
      [5]
       |
      [4]
       |
      [3]
       |
[1]---[2]
 |
_|_
        """
    },
    "home": {
        "name": "Home (Simulation Default)",
        "description": "Arm horizontal, Wrist nach unten",
        "joints": np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]),
        "diagram": """
   [1]---[2]---[3]---[4]
    |                 |
  __|__              [5]
 |_____|              |
                    [EE]
        """
    },
    "home_bent": {
        "name": "Home Bent (Ellbogen gebeugt)",
        "description": "Kompakte Ruheposition",
        "joints": np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, 0.0, 0.0]),
        "diagram": """
       [4]---[5]---[EE]
        |
       [3]
        |
[1]----[2]
 |
_|_
        """
    }
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_header():
    """Print the header with instructions."""
    print("\033[2J\033[H", end="")  # Clear screen
    print("=" * 70)
    print("GELLO POSE MONITOR - Live Motorpositionen (ohne Kalibrierung)")
    print("=" * 70)
    print("Drücke Ctrl+C zum Beenden")
    print("-" * 70)


def print_reference_pose(ref_name: str):
    """Print reference pose information."""
    if ref_name and ref_name in REFERENCE_POSES:
        ref = REFERENCE_POSES[ref_name]
        print(f"\n📍 Referenzpose: {ref['name']}")
        print(f"   {ref['description']}")
        print(f"   Erwartete Werte: {[f'{np.rad2deg(x):+.1f}°' for x in ref['joints']]}")
        print(ref['diagram'])


def format_joint_value(raw_deg: float, expected_deg: float = None) -> str:
    """Format a joint value with optional comparison."""
    if expected_deg is not None:
        diff = raw_deg - expected_deg
        # Color coding: green if close, yellow if medium, red if far
        if abs(diff) < 5:
            color = "\033[92m"  # Green
        elif abs(diff) < 15:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        return f"{color}{raw_deg:+7.1f}°\033[0m (Δ{diff:+.1f}°)"
    else:
        return f"{raw_deg:+7.1f}°"


def calculate_hypothetical_calibration(raw_pos: np.ndarray, joint_signs: np.ndarray, 
                                        target_pose: np.ndarray) -> tuple:
    """
    Berechne hypothetische Offsets, WENN der Arm gerade in target_pose wäre.
    
    Returns:
        offsets: Die Offsets die berechnet würden
        calibrated: Die kalibrierten Positionen
    """
    num_joints = min(len(raw_pos), len(target_pose), len(joint_signs))
    offsets = np.zeros(num_joints)
    
    for i in range(num_joints):
        # Offset so wählen, dass: sign * (raw - offset) = target
        # => offset = raw - target / sign
        # Aber wir suchen den nächsten Offset im Multi-Turn-Bereich
        best_offset = 0
        best_error = 1e9
        
        for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
            calibrated = joint_signs[i] * (raw_pos[i] - offset)
            error = abs(calibrated - target_pose[i])
            if error < best_error:
                best_error = error
                best_offset = offset
        
        offsets[i] = best_offset
    
    calibrated = joint_signs[:num_joints] * (raw_pos[:num_joints] - offsets)
    return offsets, calibrated


def main():
    parser = argparse.ArgumentParser(description="GELLO Pose Monitor")
    parser.add_argument("--config", "-c", required=True, help="Config YAML file")
    parser.add_argument("--reference", "-r", choices=list(REFERENCE_POSES.keys()),
                        help="Referenzpose zum Vergleich")
    parser.add_argument("--show-all", action="store_true",
                        help="Zeige Vergleich mit allen Referenzposen")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup Dynamixel
    port_config = config["dynamixel"]["dynamixel_port"]
    if port_config.startswith("/"):
        port = port_config
    else:
        port = "/dev/serial/by-id/" + port_config
    
    servo_types = config["dynamixel"]["servo_types"]
    joint_signs = np.array(config["dynamixel"]["joint_signs"], dtype=float)
    baudrate = config["dynamixel"].get("baudrate", 57600)
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    
    # Get calibration pose from config
    config_calib_pose = np.array(
        config["arm_teleop"]["initialization"]["calibration_joint_pos"]
    )
    
    print(f"Verbinde mit Dynamixels auf {port}...")
    
    try:
        joint_ids = list(range(1, len(servo_types) + 1))
        driver = DynamixelDriver(joint_ids, servo_types, port, baudrate=baudrate)
        print("✓ Verbunden!\n")
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
        return 1
    
    # Reference pose for comparison
    ref_pose = None
    if args.reference:
        ref_pose = REFERENCE_POSES[args.reference]["joints"]
    
    try:
        while True:
            print_header()
            
            # Read current positions
            raw_pos, raw_vel = driver.get_positions_and_velocities()
            raw_pos = np.array(raw_pos)
            
            # Show config info
            print(f"Config: {args.config}")
            print(f"Joint Signs: {[f'{int(s):+d}' for s in joint_signs[:num_arm_joints]]}")
            print(f"Config calibration_joint_pos: {[f'{np.rad2deg(x):.1f}°' for x in config_calib_pose]}")
            print()
            
            # Show raw motor positions
            print("ROHE MOTORPOSITIONEN (unverändert vom Dynamixel):")
            print("-" * 70)
            for i in range(num_arm_joints):
                raw_deg = np.rad2deg(raw_pos[i])
                print(f"  Motor {i+1}: {raw_deg:+8.1f}° ({raw_pos[i]:+7.4f} rad)")
            
            if len(raw_pos) > num_arm_joints:
                print(f"  Gripper:  {np.rad2deg(raw_pos[-1]):+8.1f}°")
            
            print()
            
            # Show hypothetical calibration for each reference pose
            if args.show_all:
                print("HYPOTHETISCHE KALIBRIERUNG (wenn Arm JETZT in dieser Pose wäre):")
                print("-" * 70)
                for ref_name, ref_data in REFERENCE_POSES.items():
                    ref_joints = ref_data["joints"]
                    offsets, calibrated = calculate_hypothetical_calibration(
                        raw_pos[:num_arm_joints], joint_signs[:num_arm_joints], ref_joints
                    )
                    max_error = np.max(np.abs(calibrated - ref_joints))
                    
                    status = "✓" if max_error < np.deg2rad(5) else "⚠"
                    print(f"  {status} {ref_data['name']:<25} Max-Fehler: {np.rad2deg(max_error):.1f}°")
                    print(f"      Offsets: {[f'{np.rad2deg(o):+.0f}°' for o in offsets]}")
                print()
            
            # Show comparison with selected reference
            if ref_pose is not None:
                ref_data = REFERENCE_POSES[args.reference]
                print(f"VERGLEICH MIT: {ref_data['name']}")
                print(f"  Beschreibung: {ref_data['description']}")
                print(f"  Erwartete URDF-Werte: {[f'{np.rad2deg(x):+.1f}°' for x in ref_pose]}")
                print()
                
                offsets, calibrated = calculate_hypothetical_calibration(
                    raw_pos[:num_arm_joints], joint_signs[:num_arm_joints], ref_pose
                )
                
                print("  Wenn dies die Pose ist, würde Kalibrierung ergeben:")
                print("  " + "-" * 60)
                for i in range(num_arm_joints):
                    cal_deg = np.rad2deg(calibrated[i])
                    exp_deg = np.rad2deg(ref_pose[i])
                    diff = cal_deg - exp_deg
                    
                    if abs(diff) < 3:
                        status = "✓"
                        color = "\033[92m"
                    elif abs(diff) < 10:
                        status = "~"
                        color = "\033[93m"
                    else:
                        status = "✗"
                        color = "\033[91m"
                    
                    print(f"  {status} Joint {i+1}: "
                          f"Roh={np.rad2deg(raw_pos[i]):+7.1f}° → "
                          f"Kal={color}{cal_deg:+7.1f}°\033[0m "
                          f"(Soll: {exp_deg:+.1f}°, Δ={diff:+.1f}°)")
                
                max_error = np.max(np.abs(calibrated - ref_pose))
                print()
                if max_error < np.deg2rad(5):
                    print("  \033[92m✓ Pose scheint korrekt! Kalibrierung sollte funktionieren.\033[0m")
                else:
                    print(f"  \033[93m⚠ Abweichung erkannt: {np.rad2deg(max_error):.1f}°\033[0m")
                    print("    → Entweder bist du NICHT in dieser Pose,")
                    print("    → oder die joint_signs stimmen nicht.")
            
            # If no reference, show simple info
            if not args.show_all and ref_pose is None:
                print("\n💡 Tipps:")
                print("   --reference candle   : Vergleich mit Kerzenpose (alle Gelenke 0°)")
                print("   --reference home     : Vergleich mit Home-Pose")
                print("   --show-all           : Zeige alle Referenzposen-Vergleiche")
            
            print("\n" + "-" * 70)
            print("Aktualisierung alle 1.0s... (Ctrl+C zum Beenden)")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nBeende...")
    finally:
        driver.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
