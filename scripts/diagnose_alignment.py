#!/usr/bin/env python3
"""
GELLO Leader-Follower Alignment Diagnose

Dieses Tool hilft dabei, den Zusammenhang zwischen Leader (GELLO) 
und Follower (Simulation) zu verstehen und Mismatches zu diagnostizieren.

Es zeigt:
1. Aktuelle Leader-Position (kalibriert)
2. Aktuelle Follower-Position (Simulation)
3. Das Mapping zwischen beiden
4. Wo Diskrepanzen auftreten

Usage:
    # Erst die Simulation starten:
    python experiments/launch_nodes.py --robot sim_ur --hostname 127.0.0.1 --robot_port 6001
    
    # Dann dieses Tool:
    python scripts/diagnose_alignment.py --config configs/ur5e_gello_factr_sim.yaml
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def connect_to_follower(host: str, port: int, timeout: float = 5.0):
    """Verbinde zum Follower-Robot-Server."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
    socket.connect(f"tcp://{host}:{port}")
    
    return socket, context


def get_follower_state(socket) -> np.ndarray:
    """Hole aktuelle Follower-Position über das ZMQ-Protokoll."""
    try:
        # Verwende das korrekte pickle-basierte Protokoll
        request = {"method": "get_joint_state"}
        socket.send(pickle.dumps(request))
        result = pickle.loads(socket.recv())
        if isinstance(result, dict) and "error" in result:
            return None
        return np.array(result)
    except zmq.Again:
        return None
    except Exception as e:
        print(f"  Follower-Kommunikationsfehler: {e}")
        return None


def calculate_calibration(raw_pos: np.ndarray, joint_signs: np.ndarray, 
                           target_pose: np.ndarray) -> np.ndarray:
    """Berechne Kalibrierungs-Offsets."""
    num_joints = len(target_pose)
    offsets = np.zeros(num_joints)
    
    for i in range(num_joints):
        best_offset = 0
        best_error = 1e9
        
        for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
            calibrated = joint_signs[i] * (raw_pos[i] - offset)
            error = abs(calibrated - target_pose[i])
            if error < best_error:
                best_error = error
                best_offset = offset
        
        offsets[i] = best_offset
    
    return offsets


def main():
    parser = argparse.ArgumentParser(description="Leader-Follower Alignment Diagnose")
    parser.add_argument("--config", "-c", required=True, help="Config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # === Setup Leader (Dynamixel) ===
    port_config = config["dynamixel"]["dynamixel_port"]
    if port_config.startswith("/"):
        port = port_config
    else:
        port = "/dev/serial/by-id/" + port_config
    
    servo_types = config["dynamixel"]["servo_types"]
    joint_signs = np.array(config["dynamixel"]["joint_signs"], dtype=float)
    baudrate = config["dynamixel"].get("baudrate", 57600)
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    
    calibration_pose = np.array(
        config["arm_teleop"]["initialization"]["calibration_joint_pos"]
    )
    
    # Teleop mapping config
    teleop_cfg = config.get("teleop", {})
    mapping_cfg = teleop_cfg.get("mapping", {})
    index_map = mapping_cfg.get("index_map", list(range(num_arm_joints)))
    mapping_signs = np.array(mapping_cfg.get("signs", [1] * num_arm_joints))
    mapping_offsets = np.array(mapping_cfg.get("offsets", [0.0] * num_arm_joints))
    auto_align = mapping_cfg.get("auto_align", True)
    
    follower_host = teleop_cfg.get("robot", {}).get("host", "127.0.0.1")
    follower_port = teleop_cfg.get("robot", {}).get("port", 6001)
    
    print("=" * 70)
    print("LEADER-FOLLOWER ALIGNMENT DIAGNOSE")
    print("=" * 70)
    print()
    print("Konfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Calibration Pose: {[f'{np.rad2deg(x):.0f}°' for x in calibration_pose]}")
    print(f"  Joint Signs: {[f'{int(s):+d}' for s in joint_signs[:num_arm_joints]]}")
    print(f"  Mapping Index: {index_map}")
    print(f"  Mapping Signs: {[f'{int(s):+d}' for s in mapping_signs]}")
    print(f"  Mapping Offsets: {[f'{np.rad2deg(o):.1f}°' for o in mapping_offsets]}")
    print(f"  Auto-Align: {auto_align}")
    print()
    
    # Connect to Leader
    print(f"Verbinde mit Leader (Dynamixel) auf {port}...")
    try:
        joint_ids = list(range(1, len(servo_types) + 1))
        driver = DynamixelDriver(joint_ids, servo_types, port, baudrate=baudrate)
        print("✓ Leader verbunden")
    except Exception as e:
        print(f"❌ Leader-Verbindungsfehler: {e}")
        return 1
    
    # Connect to Follower
    print(f"Verbinde mit Follower (Simulation) auf {follower_host}:{follower_port}...")
    try:
        socket, context = connect_to_follower(follower_host, follower_port)
        follower_state = get_follower_state(socket)
        if follower_state is not None:
            print(f"✓ Follower verbunden ({len(follower_state)} Gelenke)")
        else:
            print("❌ Follower antwortet nicht")
            socket = None
    except Exception as e:
        print(f"⚠ Follower nicht erreichbar: {e}")
        print("  → Starte erst: python experiments/launch_nodes.py --robot sim_ur")
        socket = None
    
    print()
    
    # Kalibriere Leader
    print("Kalibriere Leader...")
    raw_pos, _ = driver.get_positions_and_velocities()
    raw_pos = np.array(raw_pos)
    offsets = calculate_calibration(raw_pos[:num_arm_joints], 
                                    joint_signs[:num_arm_joints], 
                                    calibration_pose)
    print("✓ Kalibrierung berechnet")
    print()
    
    print("-" * 70)
    print("Live-Diagnose (Ctrl+C zum Beenden)")
    print("-" * 70)
    print()
    
    # Berechne auto-align Offsets wenn aktiviert
    if auto_align and socket:
        print("Auto-Align ist AKTIVIERT - berechne initiale Offsets...")
        follower_init = get_follower_state(socket)
        if follower_init is not None:
            # Hole initiale Leader-Position
            raw_init, _ = driver.get_positions_and_velocities()
            raw_init = np.array(raw_init[:num_arm_joints])
            calibrated_init = joint_signs[:num_arm_joints] * (raw_init - offsets)
            
            # Berechne auto-align Offsets: follower - leader (so dass leader → follower)
            auto_offsets = np.zeros(num_arm_joints)
            for i, src_idx in enumerate(index_map):
                if i < len(follower_init) and i < len(calibrated_init):
                    mapped_leader = mapping_signs[i] * calibrated_init[src_idx] if i < len(mapping_signs) else calibrated_init[src_idx]
                    auto_offsets[i] = follower_init[i] - mapped_leader
            
            # Ersetze die Config-Offsets durch die berechneten
            mapping_offsets = auto_offsets + mapping_offsets  # Addiere zu existierenden
            print(f"  Auto-Align Offsets berechnet: {[f'{np.rad2deg(o):+.1f}°' for o in auto_offsets]}")
            print(f"  Effektive Mapping-Offsets: {[f'{np.rad2deg(o):+.1f}°' for o in mapping_offsets]}")
        print()
    
    print("Spalten:")
    print("  Raw    = Rohe Dynamixel-Position")
    print("  Cal    = Kalibrierte Leader-Position (URDF)")
    print("  Mapped = Nach Teleop-Mapping transformiert (inkl. auto-align)")
    print("  Follow = Aktuelle Follower-Position")
    print("  Δ      = Differenz (Mapped - Follow)")
    print()
    
    try:
        while True:
            # Read Leader
            raw_pos, _ = driver.get_positions_and_velocities()
            raw_pos = np.array(raw_pos[:num_arm_joints])
            
            # Apply calibration
            calibrated = joint_signs[:num_arm_joints] * (raw_pos - offsets)
            
            # Apply teleop mapping
            mapped = np.zeros(num_arm_joints)
            for i, src_idx in enumerate(index_map):
                if i < len(mapping_signs):
                    mapped[i] = mapping_signs[i] * calibrated[src_idx] + mapping_offsets[i]
                else:
                    mapped[i] = calibrated[src_idx]
            
            # Read Follower
            follower_pos = None
            if socket:
                follower_pos = get_follower_state(socket)
            
            # Display
            print("\033[2J\033[H", end="")  # Clear screen
            print("=" * 70)
            print("LEADER-FOLLOWER ALIGNMENT - Live")
            print("=" * 70)
            print()
            
            print(f"{'Joint':<8} {'Raw':>10} {'Cal':>10} {'Mapped':>10} ", end="")
            if follower_pos is not None:
                print(f"{'Follow':>10} {'Δ':>10}")
            else:
                print()
            print("-" * 70)
            
            max_delta = 0
            for i in range(num_arm_joints):
                raw_deg = np.rad2deg(raw_pos[i])
                cal_deg = np.rad2deg(calibrated[i])
                map_deg = np.rad2deg(mapped[i])
                
                line = f"Joint {i+1:<2} {raw_deg:+10.1f} {cal_deg:+10.1f} {map_deg:+10.1f} "
                
                if follower_pos is not None and i < len(follower_pos):
                    fol_deg = np.rad2deg(follower_pos[i])
                    delta = map_deg - fol_deg
                    max_delta = max(max_delta, abs(delta))
                    
                    if abs(delta) < 5:
                        color = "\033[92m"  # Grün
                    elif abs(delta) < 15:
                        color = "\033[93m"  # Gelb
                    else:
                        color = "\033[91m"  # Rot
                    
                    line += f"{fol_deg:+10.1f} {color}{delta:+10.1f}°\033[0m"
                
                print(line)
            
            print("-" * 70)
            
            if follower_pos is not None:
                if max_delta < 5:
                    print("\033[92m✓ Alignment gut! Max Δ: {:.1f}°\033[0m".format(max_delta))
                elif max_delta < 15:
                    print("\033[93m~ Alignment akzeptabel. Max Δ: {:.1f}°\033[0m".format(max_delta))
                else:
                    print("\033[91m✗ Alignment Problem! Max Δ: {:.1f}°\033[0m".format(max_delta))
                    print()
                    print("  Mögliche Ursachen:")
                    print("  1. Leader war nicht in calibration_joint_pos beim Start")
                    print("  2. Follower-Startpose passt nicht zu initial_match_joint_pos")
                    print("  3. mapping.signs oder mapping.offsets falsch")
            else:
                print("⚠ Follower nicht verbunden - nur Leader-Werte werden angezeigt")
            
            print()
            print("Kalibrierungs-Offsets: ", end="")
            print([f"{np.rad2deg(o):+.0f}°" for o in offsets])
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nBeende...")
    finally:
        driver.close()
        if socket:
            socket.close()
            context.term()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
