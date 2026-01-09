#!/usr/bin/env python3
"""
Interaktiver Torque-Signs Test für GELLO.

Dieses Tool hilft herauszufinden, ob die torque_signs korrekt sind,
indem es einzelne Gelenke mit bekanntem Vorzeichen ansteuert.

WICHTIG: Das ist ein GEFÄHRLICHER Test! Der Arm kann sich schnell bewegen.
         Halte den Arm fest und sei bereit, Ctrl+C zu drücken!

Usage:
    python scripts/diagnose_torque_signs.py --config configs/ur5e_gello_factr_sim.yaml
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    port_config = config["dynamixel"]["dynamixel_port"]
    if port_config.startswith("/"):
        port = port_config
    else:
        port = "/dev/serial/by-id/" + port_config
    
    servo_types = config["dynamixel"]["servo_types"]
    torque_signs = config["dynamixel"].get("torque_signs", [1]*7)
    baudrate = config["dynamixel"].get("baudrate", 4000000)
    num_arm_joints = config["arm_teleop"]["num_arm_joints"]
    
    print("=" * 70)
    print("TORQUE SIGNS DIAGNOSE")
    print("=" * 70)
    print()
    print("⚠️  WARNUNG: Dieses Tool bewegt einzelne Gelenke mit Torque!")
    print("    Halte den Arm fest und sei bereit, Ctrl+C zu drücken!")
    print()
    print("Aktuelle torque_signs:", torque_signs[:num_arm_joints])
    print()
    
    # Connect
    joint_ids = list(range(1, len(servo_types) + 1))
    try:
        driver = DynamixelDriver(joint_ids, servo_types, port, baudrate=baudrate)
        print("✓ Verbunden")
        
        # Enable torque mode for current control
        driver.set_torque_mode(True)
        print("✓ Torque-Modus aktiviert")
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
        return 1
    
    print()
    print("So funktioniert der Test:")
    print("  1. Ein Gelenk wird mit kleinem POSITIVEN Torque angesteuert")
    print("  2. Du beobachtest, in welche Richtung es sich bewegt/drückt")
    print("  3. Wenn es sich GEGEN die Schwerkraft drückt = RICHTIG")
    print("  4. Wenn es MIT der Schwerkraft fällt = torque_sign muss invertiert werden")
    print()
    
    # Gelenk-Beschreibungen für UR5e GELLO
    joint_descriptions = [
        "Base (Drehung um Z-Achse)",
        "Schulter (hebt/senkt Oberarm) - GEGEN SCHWERKRAFT = nach oben heben",
        "Ellbogen (hebt/senkt Unterarm) - GEGEN SCHWERKRAFT = nach oben heben",
        "Handgelenk 1 (Rotation)",
        "Handgelenk 2 (Neigung)",
        "Handgelenk 3 (Rotation)"
    ]
    
    test_torque = 0.1  # Nm - klein genug um sicher zu sein
    
    try:
        while True:
            print("-" * 70)
            print("Welches Gelenk testen? (1-6, oder 'q' zum Beenden)")
            for i in range(num_arm_joints):
                sign_str = "+" if torque_signs[i] > 0 else "-"
                print(f"  {i+1}: {joint_descriptions[i]} [sign: {sign_str}1]")
            print()
            
            choice = input("Auswahl: ").strip().lower()
            if choice == 'q':
                break
            
            try:
                joint_idx = int(choice) - 1
                if joint_idx < 0 or joint_idx >= num_arm_joints:
                    print("Ungültige Auswahl!")
                    continue
            except ValueError:
                print("Bitte eine Zahl eingeben!")
                continue
            
            print()
            print(f"Test für Joint {joint_idx + 1}: {joint_descriptions[joint_idx]}")
            print(f"  Aktueller torque_sign: {torque_signs[joint_idx]:+d}")
            print()
            print("Der Test wird jetzt starten:")
            print("  1. Zuerst 2 Sekunden KEIN Torque (zum Stabilisieren)")
            print("  2. Dann 3 Sekunden POSITIVER Torque (mit aktuellem sign)")
            print("  3. Dann 3 Sekunden NEGATIVER Torque (invertiert)")
            print()
            input("HALTE DEN ARM FEST! Drücke ENTER um zu starten...")
            
            # Phase 1: Kein Torque
            print("Phase 1: Kein Torque (2s)...")
            torques = [0.0] * len(servo_types)
            driver.set_torque(torques)
            time.sleep(2)
            
            # Phase 2: Positiver Torque (mit sign)
            print(f"Phase 2: +{test_torque} Nm × sign={torque_signs[joint_idx]:+d} (3s)...")
            torques = [0.0] * len(servo_types)
            torques[joint_idx] = test_torque * torque_signs[joint_idx]
            driver.set_torque(torques)
            time.sleep(3)
            
            # Phase 3: Negativer Torque
            print(f"Phase 3: -{test_torque} Nm × sign={torque_signs[joint_idx]:+d} (3s)...")
            torques = [0.0] * len(servo_types)
            torques[joint_idx] = -test_torque * torque_signs[joint_idx]
            driver.set_torque(torques)
            time.sleep(3)
            
            # Stopp
            print("Stopp - Torque = 0")
            torques = [0.0] * len(servo_types)
            driver.set_torque(torques)
            
            print()
            print("Frage: Hat das Gelenk in Phase 2 GEGEN die Schwerkraft gedrückt?")
            print("  (j) Ja, es hat nach oben / gegen Schwerkraft gedrückt")
            print("  (n) Nein, es hat nach unten / mit Schwerkraft gedrückt")
            answer = input("Antwort: ").strip().lower()
            
            if answer == 'j':
                print(f"✓ torque_sign[{joint_idx}] = {torque_signs[joint_idx]:+d} ist KORREKT!")
            else:
                new_sign = -torque_signs[joint_idx]
                print(f"✗ torque_sign[{joint_idx}] sollte {new_sign:+d} sein!")
                print(f"  → Ändere in der Config: torque_signs[{joint_idx}]: {new_sign}")
            print()
            
    except KeyboardInterrupt:
        print("\n\nAbgebrochen!")
    finally:
        # Sicherheits-Stopp
        try:
            torques = [0.0] * len(servo_types)
            driver.set_torque(torques)
        except:
            pass
        driver.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
