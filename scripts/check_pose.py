"""
DEPRECATED: Dieses Script wurde durch bessere Tools ersetzt!

Verwende stattdessen:
  - scripts/pose_monitor.py    : Live Anzeige der Motorpositionen
  - scripts/calibrate_gello.py : Interaktive Kalibrierung mit visueller Anleitung

Beispiel:
    python scripts/pose_monitor.py --config configs/ur5e_gello_factr_sim.yaml --reference candle
    python scripts/calibrate_gello.py --config configs/ur5e_gello_factr_sim.yaml
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import yaml

print(__doc__)
print("=" * 60)
print("Dieses Script läuft trotzdem weiter, aber erwäge die neuen Tools.")
print("=" * 60)
print()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gello.dynamixel.driver import DynamixelDriver

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = "../configs/ur5e_gello_factr_sim.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Extract params
    port = config["dynamixel"]["dynamixel_port"]
    if not port.startswith("/"):
        port = "/dev/serial/by-id/" + port
        
    baudrate = config["dynamixel"].get("baudrate", 57600)
    servo_types = config["dynamixel"]["servo_types"]
    # Slice joint_signs to match the 6 arm joints we are checking
    joint_signs = np.array(config["dynamixel"]["joint_signs"])[:6]
    
    # Define calibration options
    options = {
        1: ("Null (All Zeros)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        2: ("T-Pose (Horizontal)", np.array([0.0, -1.5708, 0.0, 0.0, 0.0, 0.0])),
        3: ("Home (Angled)", np.array([0.0, -0.7854, -1.5708, 0.0, 0.0, 0.0])),
        4: ("Candle (Vertical - from txt)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        5: ("Current Config", np.array(config["arm_teleop"]["initialization"]["calibration_joint_pos"]))
    }

    print("\nSelect Calibration Pose to Test:")
    for key, (name, pose) in options.items():
        pose_str = ", ".join([f"{x:.2f}" for x in pose])
        print(f"{key}: {name} -> [{pose_str}]")
    
    try:
        choice = int(input("\nEnter option number (1-5): "))
        if choice not in options:
            print("Invalid choice, defaulting to 5 (Current Config)")
            choice = 5
    except ValueError:
        print("Invalid input, defaulting to 5 (Current Config)")
        choice = 5
        
    selected_name, calibration_target = options[choice]
    
    print(f"\nConnecting to Dynamixels on {port}...")
    joint_ids = list(range(1, len(servo_types) + 1))
    driver = DynamixelDriver(joint_ids, servo_types, port, baudrate=baudrate)
    
    print("\n" + "="*60)
    print(f"POSE CHECKER - Mode: {selected_name}")
    print("="*60)
    print(f"Hold the robot in the {selected_name} pose.")
    print(f"Target Angles (deg): {[f'{np.rad2deg(x):.1f}' for x in calibration_target]}")
    print("-" * 60)
    
    # Calculate offsets based on current position (assuming we are AT the target)
    print("Reading current positions to calculate offsets...")
    raw_pos, _ = driver.get_positions_and_velocities()
    raw_pos = np.array(raw_pos[:6]) # Arm only
    
    # Let's simulate the calibration search
    offsets = []
    print("Calculating offsets...")
    for i in range(6):
        best_offset = 0
        best_error = 1e9
        target = calibration_target[i]
        sign = joint_signs[i]
        raw = raw_pos[i]
        
        # Search range similar to main code
        for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
            val = sign * (raw - offset)
            err = abs(val - target)
            if err < best_error:
                best_error = err
                best_offset = offset
        offsets.append(best_offset)
    
    offsets = np.array(offsets)
    print(f"Calculated Offsets: {[f'{x:.3f}' for x in offsets]}")
    
    print("\nNow showing live calibrated angles. Move the robot to verify.")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            curr_raw, _ = driver.get_positions_and_velocities()
            curr_raw = np.array(curr_raw[:6])
            
            # Apply calibration
            curr_angles = (curr_raw - offsets) * joint_signs
            
            # Print
            deg_str = [f"{np.rad2deg(x):+6.1f}" for x in curr_angles]
            print(f"\rAngles (deg): {deg_str}", end="")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDone.")

if __name__ == "__main__":
    main()
