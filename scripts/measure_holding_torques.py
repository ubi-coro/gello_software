#!/usr/bin/env python3
"""
Gravity Compensation - Haltemoment-Messung
===========================================

Dieses Skript misst die realen Haltemomente der GELLO-Roboter an verschiedenen
Posen, um die Gravity Compensation Parameter zu validieren und zu verbessern.

Methode:
1. Motor fährt im Position-Modus zur Zielpose
2. Warten bis Bewegung abgeschlossen und Position stabilisiert
3. Haltestrom auslesen (= Strom gegen Schwerkraft)
4. Strom → Drehmoment umrechnen
5. Optional: Least-Squares-Optimierung über alle Posen

Nutzung:
    python scripts/measure_holding_torques.py --config configs/ur5e_gello_factr_sim.yaml
    python scripts/measure_holding_torques.py --config configs/ur5e_gello_factr_sim.yaml --poses custom

Autor: GELLO Team
Datum: 2026-01-02
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    print("[WARN] Pinocchio nicht installiert - RNEA-Vergleich nicht verfügbar")
    HAS_PINOCCHIO = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver

# Dynamixel Control Table Addresses (X-Series)
ADDR_PRESENT_CURRENT = 126
LEN_PRESENT_CURRENT = 2
ADDR_PROFILE_VELOCITY = 112
ADDR_MOVING = 122
ADDR_MOVING_STATUS = 123

# Umrechnungsfaktoren
# XC330: Current unit = 1 mA, Torque constant ≈ 1.136 Nm/A
# XM430: Current unit = 2.69 mA, Torque constant ≈ 1.783 Nm/A
CURRENT_TO_TORQUE = {
    "XC330_T288_T": 1.136 / 1000.0,      # Nm/mA
    "XM430_W210_T": 1.304 * 2.69 / 1000.0,  # Nm/mA (unit 2.69mA)
    "XM430_W350_T": 1.783 * 2.69 / 1000.0,  # Nm/mA (unit 2.69mA)
}


@dataclass
class CalibrationPose:
    """Eine Kalibrierpose mit Namen und Gelenkwinkeln"""
    name: str
    joint_angles_deg: List[float]
    
    @property
    def joint_angles_rad(self) -> np.ndarray:
        return np.deg2rad(self.joint_angles_deg)


class HoldingTorqueMeasurement:
    """Klasse zur Messung von Haltemomenten"""
    
    def __init__(
        self,
        driver: DynamixelDriver,
        servo_types: List[str],
        num_arm_joints: int = 6,
        urdf_path: Optional[str] = None,
    ):
        self.driver = driver
        self.servo_types = servo_types
        self.num_arm_joints = num_arm_joints
        self.measurements: List[Dict] = []
        
        # Torque conversion factors for each joint
        self.torque_factors = np.array([
            CURRENT_TO_TORQUE.get(servo_type, 0.001)
            for servo_type in servo_types[:num_arm_joints]
        ])

        # Calibration parameters (default to identity if not calibrated)
        self.offsets = np.zeros(num_arm_joints)
        self.signs = np.ones(num_arm_joints)
        
        # Pinocchio model for RNEA comparison
        self.pin_model = None
        self.pin_data = None
        if HAS_PINOCCHIO and urdf_path:
            try:
                self.pin_model = pin.buildModelFromUrdf(urdf_path)
                self.pin_data = self.pin_model.createData()
                print(f"\n[INFO] URDF geladen: {urdf_path}")
                print(f"[INFO] Pinocchio-Modell: {self.pin_model.nq} DOF")
            except Exception as e:
                print(f"[WARN] URDF konnte nicht geladen werden: {e}")
                print(f"[WARN] RNEA-Vergleich nicht verfügbar")
        
        print(f"\n[INFO] Initialisiert für {num_arm_joints} Gelenke")
        print(f"[INFO] Servo-Typen: {servo_types[:num_arm_joints]}")
        print(f"[INFO] Torque-Faktoren [Nm/mA]: {self.torque_factors}")
    
    def read_present_current(self) -> np.ndarray:
        """Liest den aktuellen Motorstrom aus (in mA)"""
        currents = np.zeros(self.num_arm_joints)
        
        if self.driver._is_fake:
            # In fake mode, return zeros
            return currents
        
        with self.driver._lock:
            for i, dxl_id in enumerate(self.driver._ids[:self.num_arm_joints]):
                try:
                    # Read 2-byte signed current value
                    raw_current, comm_result, error = (
                        self.driver._packetHandler.read2ByteTxRx(
                            self.driver._portHandler,
                            dxl_id,
                            ADDR_PRESENT_CURRENT
                        )
                    )
                    
                    if comm_result == 0:  # COMM_SUCCESS
                        # Convert to signed 16-bit
                        if raw_current > 0x7FFF:
                            raw_current -= 0x10000
                        currents[i] = float(raw_current)
                    else:
                        print(f"[WARN] Failed to read current from motor {dxl_id}")
                except Exception as e:
                    print(f"[ERROR] Reading current from motor {dxl_id}: {e}")
        
        return currents
    
    def current_to_torque(self, currents_ma: np.ndarray) -> np.ndarray:
        """Konvertiert Ströme (mA) zu Drehmomenten (Nm)"""
        return currents_ma * self.torque_factors
    
    def motor_to_urdf_angles(self, motor_angles: np.ndarray) -> np.ndarray:
        """
        Konvertiert Motor-Winkel zu URDF-Kinematik-Winkeln.
        
        Verwendet die in self.offsets und self.signs gespeicherten Parameter
        der Kalibrierung, um konsistent mit gravity_compensation.py zu sein.
        
        Formel: urdf = (motor - offset) * sign
        """
        # Stelle sicher dass wir Arrays haben
        motor_angles = np.array(motor_angles)
        offsets = np.array(self.offsets[:len(motor_angles)])
        signs = np.array(self.signs[:len(motor_angles)])
        
        return (motor_angles - offsets) * signs
    
    def compute_rnea_torques(self, position_rad: np.ndarray) -> Optional[np.ndarray]:
        """
        Berechnet Gelenkmomente mit Pinocchio RNEA basierend auf URDF.
        
        Args:
            position_rad: Gelenkwinkel in Motor-Konvention [rad] (nur Arm-Gelenke)
        
        Returns:
            Gelenkmomente [Nm] oder None wenn Pinocchio nicht verfügbar
        """
        if not self.pin_model or not self.pin_data:
            return None
        
        try:
            # Konvertiere zu URDF-Konvention (nur Arm-Gelenke)
            q_arm = self.motor_to_urdf_angles(position_rad)
            
            # URDF hat 7 DOF (6 Arm + 1 Gripper), füge Gripper bei 0° hinzu
            q = np.zeros(self.pin_model.nq)
            q[:self.num_arm_joints] = q_arm
            q[self.num_arm_joints] = 0.0  # Gripper bei Nullposition
            
            # RNEA mit Gravitation, keine Bewegung
            v = np.zeros(self.pin_model.nv)  # Geschwindigkeit = 0
            a = np.zeros(self.pin_model.nv)  # Beschleunigung = 0
            
            # Berechne Inverse Dynamik (Gravitations- + Coriolis-Kompensation)
            # Bei v=0, a=0 bleibt nur Gravitation übrig
            tau = pin.rnea(self.pin_model, self.pin_data, q, v, a)
            
            return tau[:self.num_arm_joints]
        
        except Exception as e:
            print(f"[WARN] RNEA-Berechnung fehlgeschlagen: {e}")
            return None
    
    def is_motion_complete(
        self,
        target_pos: np.ndarray,
        position_threshold: float = 0.02,  # rad
        velocity_threshold: float = 0.05,  # rad/s
    ) -> bool:
        """Prüft ob die Bewegung abgeschlossen ist"""
        current_pos, current_vel = self.driver.get_positions_and_velocities()
        
        # Only check arm joints
        current_pos = current_pos[:self.num_arm_joints]
        current_vel = current_vel[:self.num_arm_joints]
        
        pos_error = np.max(np.abs(current_pos - target_pos))
        vel_magnitude = np.max(np.abs(current_vel))
        
        return pos_error < position_threshold and vel_magnitude < velocity_threshold
    
    def move_to_pose(
        self,
        target_rad: np.ndarray,
        timeout: float = 15.0,
        check_interval: float = 0.1,
    ) -> bool:
        """
        Fährt zur Zielpose und wartet bis die Bewegung abgeschlossen ist.
        
        Returns:
            True wenn erfolgreich, False bei Timeout
        """
        print(f"    → Fahre zu Position: {np.rad2deg(target_rad).round(1)}°")
        
        # Add gripper joint (keep at current position)
        current_pos, _ = self.driver.get_positions_and_velocities() # Use standard method
        full_target = np.zeros(len(self.driver._ids))
        full_target[:self.num_arm_joints] = target_rad
        if len(current_pos) > self.num_arm_joints:
            full_target[self.num_arm_joints:] = current_pos[self.num_arm_joints:]
        
        # Send position commands individually (more reliable than SyncWrite in position mode)
        if not self.driver._is_fake:
            ADDR_GOAL_POSITION = 116
            for dxl_id, angle in zip(self.driver._ids, full_target):
                position_value = int(angle * 2048 / np.pi)
                try:
                    with self.driver._lock:
                        self.driver._packetHandler.write4ByteTxRx(
                            self.driver._portHandler,
                            dxl_id,
                            ADDR_GOAL_POSITION,
                            position_value
                        )
                except Exception as e:
                    print(f"    ✗ Fehler beim Setzen Position Motor {dxl_id}: {e}")
                    return False
        else:
            self.driver.set_joints(full_target)
        
        start_time = time.time()
        last_print = start_time
        
        while time.time() - start_time < timeout:
            if self.is_motion_complete(target_rad):
                elapsed = time.time() - start_time
                print(f"    ✓ Position erreicht nach {elapsed:.1f}s")
                return True
            
            # Progress indicator every 2 seconds
            if time.time() - last_print > 2.0:
                current_pos, _ = self.driver.get_positions_and_velocities() # Use standard
                current_pos = current_pos[:self.num_arm_joints]
                error = np.max(np.abs(current_pos - target_rad))
                print(f"    ... Fehler: {np.rad2deg(error):.1f}° (max)")
                last_print = time.time()
            
            time.sleep(check_interval)
        
        print(f"    ✗ Timeout nach {timeout}s - Position nicht erreicht")
        return False
    def calibrate_offsets(self, calibration_pose: np.ndarray, joint_signs: np.ndarray) -> None:
        """
        Kalibriert Multi-Turn Offsets basierend auf einer bekannten Pose.
        
        Sucht den Offset k*2pi, der den Fehler |sign*(raw-offset) - target| minimiert.
        
        Args:
           calibration_pose: Ziel-Winkel im URDF-Frame [rad]
           joint_signs: Vorzeichen-Mapping vom Motor zum URDF-Frame
        """
        print("\n[INFO] Kalibriere Offsets...")
        
        # Mehrfaches Lesen zur Stabilisierung
        for _ in range(5):
            self.driver.get_positions_and_velocities()
            time.sleep(0.02)
            
        raw_pos, _ = self.driver.get_positions_and_velocities()
        raw_pos = np.array(raw_pos[:self.num_arm_joints])
        
        self.signs = np.array(joint_signs[:self.num_arm_joints])
        self.offsets = []
        
        # Suche im Bereich +/- 10 Umdrehungen
        cal_range = 10
        cal_steps = 721
        offsets_grid = np.linspace(-cal_range * np.pi, cal_range * np.pi, cal_steps)
        
        for i in range(self.num_arm_joints):
            sign = float(self.signs[i])
            raw = float(raw_pos[i])
            target = float(calibration_pose[i])
            
            # signed_joint = sign * (raw - offset)
            # error = signed_joint - target
            
            signed_joint_grid = sign * (raw - offsets_grid)
            err = signed_joint_grid - target
            best_idx = int(np.argmin(np.abs(err)))
            
            best_offset = offsets_grid[best_idx]
            self.offsets.append(best_offset)
            
            # Debug output
            actual = sign * (raw - best_offset)
            print(f"  J{i+1}: Raw={raw:5.2f}, Target={target:5.2f} -> Offset={best_offset:5.2f} (Err={actual - target:.4f})")
            
        self.offsets = np.array(self.offsets)
        print("[INFO] Kalibrierung abgeschlossen.\n")

    def measure_holding_torque(
        self,
        settling_time: float = 1.0,
        n_samples: int = 50,
        sample_interval: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Misst das Haltemoment im stationären Zustand.
        
        Returns:
            (position_mean, current_mean, current_std, torque_mean, rnea_torque)
        """
        print(f"    → Warte {settling_time}s zum Einschwingen...")
        time.sleep(settling_time)
        
        print(f"    → Messe {n_samples} Samples...", end=" ", flush=True)
        
        positions = []
        currents = []
        
        for _ in range(n_samples):
            # Nutze get_positions_and_velocities für Konsistenz
            pos, _ = self.driver.get_positions_and_velocities()
            pos = pos[:self.num_arm_joints]
            curr = self.read_present_current()
            positions.append(pos)
            currents.append(curr)
            time.sleep(sample_interval)
        
        pos_mean = np.mean(positions, axis=0)
        current_mean = np.mean(currents, axis=0)
        current_std = np.std(currents, axis=0)
        torque_mean = self.current_to_torque(current_mean)
        
        # Berechne RNEA-Momente für Vergleich
        # WARNUNG: Dieser Vergleich nutzt nun die kalibrierten URDF Winkel!
        rnea_torque = self.compute_rnea_torques(pos_mean)
        
        print("✓")
        print(f"    Position (URDF): {np.rad2deg(self.motor_to_urdf_angles(pos_mean)).round(1)}°")
        print(f"    Strom:           {current_mean.round(1)} mA")
        print(f"    Std:      {current_std.round(1)} mA")
        print(f"    Moment (gemessen): {(torque_mean * 1000).round(2)} mNm")
        if rnea_torque is not None:
            print(f"    Moment (RNEA):     {(rnea_torque * 1000).round(2)} mNm")
            error = np.abs(torque_mean - rnea_torque) * 1000
            print(f"    Fehler (|Δ|):      {error.round(2)} mNm")
        
        return pos_mean, current_mean, current_std, torque_mean, rnea_torque
    
    def add_measurement(
        self,
        pose_name: str,
        position: np.ndarray,
        current: np.ndarray,
        current_std: np.ndarray,
        torque: np.ndarray,
        rnea_torque: Optional[np.ndarray] = None,
    ):
        """Speichert eine Messung"""
        measurement = {
            'name': pose_name,
            'position_rad': position.tolist(),
            'position_deg': np.rad2deg(position).tolist(),
            # Nutze die korrigierte Funktion, die Offsets und Signs respektiert
            'position_urdf_rad': self.motor_to_urdf_angles(position).tolist(),
            'current_ma': current.tolist(),
            'current_std_ma': current_std.tolist(),
            'torque_nm': torque.tolist(),
            'torque_mnm': (torque * 1000).tolist(),
        }
        
        if rnea_torque is not None:
            measurement['rnea_torque_nm'] = rnea_torque.tolist()
            measurement['rnea_torque_mnm'] = (rnea_torque * 1000).tolist()
            error = np.abs(torque - rnea_torque)
            measurement['rnea_error_nm'] = error.tolist()
            measurement['rnea_error_mnm'] = (error * 1000).tolist()
        
        self.measurements.append(measurement)
        
        print(f"    ✓ Messung #{len(self.measurements)} gespeichert: '{pose_name}'")
    
    def save_results(self, output_file: str):
        """Speichert die Messergebnisse als JSON"""
        data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_joints': self.num_arm_joints,
                'servo_types': self.servo_types[:self.num_arm_joints],
                'torque_factors_nm_per_ma': self.torque_factors.tolist(),
                'num_measurements': len(self.measurements),
            },
            'measurements': self.measurements,
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[✓] Ergebnisse gespeichert: {output_file}")
    
    def print_summary(self):
        """Gibt eine Zusammenfassung der Messungen aus"""
        if not self.measurements:
            print("\n[INFO] Keine Messungen vorhanden")
            return
        
        print("\n" + "="*70)
        print("ZUSAMMENFASSUNG DER MESSUNGEN")
        print("="*70)
        
        # Extract data
        joint_names = [f"J{i+1}" for i in range(self.num_arm_joints)]
        has_rnea = 'rnea_torque_nm' in self.measurements[0]
        
        print(f"\nAnzahl Messungen: {len(self.measurements)}")
        print(f"\nGelenk-Momente über alle Posen [mNm]:")
        print(f"{'Pose':<20} " + " ".join([f"{j:>8}" for j in joint_names]))
        print("-" * 70)
        
        for m in self.measurements:
            torques_mnm = m['torque_mnm']
            print(f"{m['name']:<20} " + " ".join([f"{t:>8.2f}" for t in torques_mnm]))
        
        # Statistics
        all_torques = np.array([m['torque_nm'] for m in self.measurements])
        mean_torques = np.mean(np.abs(all_torques), axis=0) * 1000  # mNm
        max_torques = np.max(np.abs(all_torques), axis=0) * 1000   # mNm
        
        print("-" * 70)
        print(f"{'Mean (abs)':<20} " + " ".join([f"{t:>8.2f}" for t in mean_torques]))
        print(f"{'Max (abs)':<20} " + " ".join([f"{t:>8.2f}" for t in max_torques]))
        
        # RNEA comparison
        if has_rnea:
            print("\n" + "="*70)
            print("RNEA-VERGLEICH (URDF-basiert)")
            print("="*70)
            
            all_rnea = np.array([m['rnea_torque_nm'] for m in self.measurements])
            all_errors = np.abs(all_torques - all_rnea) * 1000  # mNm
            
            mean_rnea = np.mean(np.abs(all_rnea), axis=0) * 1000
            mean_error = np.mean(all_errors, axis=0)
            rmse = np.sqrt(np.mean((all_torques - all_rnea)**2, axis=0)) * 1000
            
            print(f"\n{'Metric':<20} " + " ".join([f"{j:>8}" for j in joint_names]))
            print("-" * 70)
            print(f"{'Measured (mean)':<20} " + " ".join([f"{t:>8.2f}" for t in mean_torques]))
            print(f"{'RNEA (mean)':<20} " + " ".join([f"{t:>8.2f}" for t in mean_rnea]))
            print(f"{'Error (mean)':<20} " + " ".join([f"{t:>8.2f}" for t in mean_error]))
            print(f"{'RMSE':<20} " + " ".join([f"{t:>8.2f}" for t in rmse]))
            
            # Relative error where measured > 50 mNm (avoid division by small numbers)
            mask = mean_torques > 50
            if np.any(mask):
                rel_error = np.zeros_like(mean_error)
                rel_error[mask] = (mean_error[mask] / mean_torques[mask]) * 100
                print(f"{'Rel. Error [%]':<20} " + " ".join([
                    f"{e:>8.1f}" if m else "    -   "
                    for e, m in zip(rel_error, mask)
                ]))
        
        print("="*70)


def get_default_calibration_poses() -> List[CalibrationPose]:
    """
    Standard-Kalibrierposen für GELLO UR5e.
    Winkel in Grad, angepasst an mechanische Nullposition:
    - Mechanische Nullposition: [180, 0, 180, 0, 180, 0]
    - Base (J1): Nur 180°-260° nutzbar (Kabel-Einschränkung)
    - WICHTIG: J2, J4, J6 bei 0° verwenden wir 360° (verhindert 360°-Drehung)
    
    Notation: Winkel relativ zur Motorposition (nicht URDF!)
    """
    return [
        # Basis-Posen
        CalibrationPose("home", [180, 0, 180, 0, 180, 0]),
        CalibrationPose("base_200", [200, 0, 180, 0, 180, 0]),
        CalibrationPose("base_220", [220, 0, 180, 0, 180, 0]),
        CalibrationPose("base_240", [240, 0, 180, 0, 180, 0]),
        
        # Shoulder (J2) Variationen
        CalibrationPose("shoulder_30", [220, 30, 180, 0, 180, 0]),
        CalibrationPose("shoulder_45", [220, 45, 180, 0, 180, 0]),
        CalibrationPose("shoulder_60", [220, 60, 180, 0, 180, 0]),
        CalibrationPose("shoulder_90", [220, 90, 180, 0, 180, 0]),
        CalibrationPose("shoulder_100", [220, 100, 180, 0, 180, 0]),
        CalibrationPose("shoulder_120", [220, 120, 180, 0, 180, 0]),
        CalibrationPose("shoulder_150", [220, 150, 180, 0, 180, 0]),
        CalibrationPose("shoulder_180", [220, 180, 180, 0, 180, 0]),
        CalibrationPose("shoulder_210", [220, 210, 180, 0, 180, 0]),
        CalibrationPose("shoulder_240", [220, 240, 180, 0, 180, 0]),
        CalibrationPose("shoulder_260", [220, 260, 180, 0, 180, 0]),
        CalibrationPose("shoulder_270", [220, 270, 180, 0, 180, 0]),
        CalibrationPose("shoulder_300", [220, 300, 180, 0, 180, 0]),
        CalibrationPose("shoulder_330", [220, 330, 180, 0, 180, 0]),
        
        # Elbow (J3) Variationen - symmetrisch um 180°
        CalibrationPose("elbow_150_S0", [220, 0, 150, 0, 180, 0]),
        CalibrationPose("elbow_210_s0", [220, 0, 210, 0, 180, 0]),
        CalibrationPose("elbow_120_S30", [220, 30, 120, 0, 180, 0]),
        CalibrationPose("elbow_90_S30", [220, 30, 90, 0, 180, 0]),
        CalibrationPose("elbow_240_S30", [220, 30, 240, 0, 180, 0]),
        CalibrationPose("elbow_270_S30", [220, 30, 270, 0, 180, 0]),
        CalibrationPose("elbow_120_S90", [220, 90, 120, 0, 180, 0]),
        CalibrationPose("elbow_90_S90", [220, 90, 90, 0, 180, 0]),
        CalibrationPose("elbow_240_S90", [220, 90, 240, 0, 180, 0]),
        CalibrationPose("elbow_270_S90", [220, 90, 270, 0, 180, 0]),
        CalibrationPose("elbow_120_S120", [220, 120, 120, 0, 180, 0]),
        CalibrationPose("elbow_90_S120", [220, 120, 90, 0, 180, 0]),
        CalibrationPose("elbow_240_S120", [220, 120, 240, 0, 180, 0]),
        CalibrationPose("elbow_270_S120", [220, 120, 270, 0, 180, 0]),

        # L4 Variationen
        CalibrationPose("wrist_30", [220, 90, 180, 30, 180, 0]),
        CalibrationPose("wrist_60", [220, 90, 180, 60, 180, 0]),
        CalibrationPose("wrist_90", [220, 90, 180, 90, 180, 0]),
        CalibrationPose("wrist_120", [220, 90, 180, 120, 180, 0]),
        CalibrationPose("wrist_150", [220, 90, 180, 150, 180, 0]),
        
        # Kombinierte Posen
        CalibrationPose("combined_1", [200, 30, 150, 20, 180, 0]),
        CalibrationPose("combined_2", [220, 330, 210, 0, 180, 0]),
        CalibrationPose("combined_3", [220, 45, 135, 30, 170, 30]),
    ]


def get_quick_test_poses() -> List[CalibrationPose]:
    """
    Schnelle Test-Posen für Debugging.
    WICHTIG: 360° statt 0° für J2, J4, J6 (verhindert 360°-Drehung)
    """
    return [
        CalibrationPose("home", [180, 0, 180, 0, 180, 0]),
        CalibrationPose("base_220", [220, 0, 180, 0, 180, 0]),
        CalibrationPose("shoulder_45", [220, 45, 180, 0, 180, 0]),
        CalibrationPose("elbow_150", [220, 30, 150, 0, 180, 0]),
    ]


def load_config(config_path: str) -> dict:
    """Lädt die YAML-Konfiguration"""
    # Resolve path relative to workspace root if not absolute
    if not os.path.isabs(config_path):
        # Get workspace root (parent of scripts directory)
        workspace_root = Path(__file__).parent.parent
        config_path = workspace_root / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_driver_from_config(config: dict) -> DynamixelDriver:
    """Erstellt DynamixelDriver aus der Konfiguration"""
    dyn_config = config['dynamixel']
    arm_config = config['arm_teleop']
    
    num_arm_joints = arm_config['num_arm_joints']
    
    # Motor IDs: 1-based, sequential
    motor_ids = list(range(1, num_arm_joints + 2))  # +1 for gripper
    
    print(f"\n[INFO] Initialisiere Dynamixel Driver...")
    print(f"  Port: {dyn_config['dynamixel_port']}")
    print(f"  Baudrate: {dyn_config['baudrate']}")
    print(f"  Motor IDs: {motor_ids}")
    
    driver = DynamixelDriver(
        ids=motor_ids,
        servo_types=dyn_config['servo_types'],
        port=dyn_config['dynamixel_port'],
        baudrate=dyn_config['baudrate'],
        use_fake_fallback=False,
    )
    
    return driver


def configure_position_mode(driver: DynamixelDriver, slow_velocity: int = 50):
    """
    Konfiguriert Position-Modus mit langsamer Geschwindigkeit.
    
    Args:
        slow_velocity: Profil-Geschwindigkeit (niedriger = langsamer)
                      Unit: 0.229 rpm (XC330) oder 0.114 rpm (XM430)
    """
    print(f"\n[INFO] Konfiguriere Position-Modus...")
    
    # Disable torque to change mode
    driver.set_torque_mode(False)
    
    # Set position control mode (mode 3)
    driver.set_operating_mode(3)
    
    # Set slow profile velocity for all motors
    if not driver._is_fake:
        with driver._lock:
            for dxl_id in driver._ids:
                driver._packetHandler.write4ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_PROFILE_VELOCITY,
                    slow_velocity
                )
    
    # Enable torque
    driver.set_torque_mode(True)
    
    # Wait for mode to be fully active
    time.sleep(0.5)
    
    print(f"  ✓ Position-Modus aktiv (Profil-Geschwindigkeit: {slow_velocity})")


def run_interactive_measurement(
    measurement: HoldingTorqueMeasurement,
    poses: List[CalibrationPose],
    output_dir: str,
    calibration_config: Optional[dict] = None, # Added config parameter
):
    """Führt die interaktive Messung durch"""
    
    # -----------------------------------------------------------
    # Initial Calibration Step (New Logic)
    # -----------------------------------------------------------
    if calibration_config:
        print(f"[DEBUG] Config keys: {list(calibration_config.keys())}")
        print("\n" + "="*70)
        print("INITIALE KALIBRIERUNG")
        print("="*70)
        print("Bitte den Roboter in die KALIBRIER-POSE bringen.")
        print("Dies ist notwendig, um die Motor-Offsets korrekt zu bestimmen.")
        
        calib_pos = np.array(calibration_config["arm_teleop"]["initialization"]["calibration_joint_pos"])
        joint_signs = np.array(calibration_config["dynamixel"]["joint_signs"])
        
        print(f"\nErwartete Pose (URDF Frame): {calib_pos}")
        input("\nDrücke [Enter], wenn der Roboter in Position ist...")
        
        measurement.calibrate_offsets(calib_pos, joint_signs)
    
    print("\n" + "="*70)
    print("INTERAKTIVE HALTEMOMENT-MESSUNG")
    print("="*70)
    print(f"\n{len(poses)} Kalibrierposen definiert\n")
    
    print("BEDIENUNG:")
    print("  [Enter]  → Zur nächsten Pose fahren und messen")
    print("  [s]      → Pose überspringen")
    print("  [r]      → Aktuelle Pose wiederholen")
    print("  [m]      → Aktuelle Position messen (ohne Fahren)")
    print("  [t]      → Torque AN/AUS (Manuelles Bewegen)")
    print("  [l]      → Alle Posen auflisten")
    print("  [q]      → Beenden und Ergebnisse speichern")
    print("  [Strg+C] → Notfall-Abbruch")
    print("="*70)
    
    input("\n[Enter] zum Starten...")
    
    pose_idx = 0
    
    try:
        while pose_idx < len(poses):
            pose = poses[pose_idx]
            
            print(f"\n{'='*70}")
            print(f"POSE {pose_idx + 1}/{len(poses)}: {pose.name}")
            print(f"{'='*70}")
            
            command = input("\n[Enter] fahren und messen | [s]kip | [r]epeat | [m]easure | [t]orque | [l]ist | [q]uit: ").strip().lower()
            
            if command == 'q':
                print("\n[INFO] Beende Messung...")
                break
            
            elif command == 's':
                print(f"[INFO] Überspringe Pose '{pose.name}'")
                pose_idx += 1
                continue
            
            elif command == 't':
                print("\\n[ACHTUNG] Ändere Torque-Status...")
                print("    0: Torque AUS (Manuell bewegen - Roboter festhalten!)")
                print("    1: Torque AN (Position halten)")
                sub = input("    Wahl [0/1]: ").strip()
                
                if sub == '0':
                    print("    ACHTUNG: Gravitation! Roboter festhalten.")
                    measurement.driver.set_torque_mode(False)
                    print("    ✓ Torque ist AUS.")
                elif sub == '1':
                    measurement.driver.set_torque_mode(True)
                    print("    ✓ Torque ist AN.")
                continue

            elif command == 'r':
                print(f"[INFO] Wiederhole Pose '{pose.name}'")
                # Don't increment pose_idx
                continue
            
            elif command == 'l':
                print("\nVERFÜGBARE POSEN:")
                for i, p in enumerate(poses):
                    status = "✓" if i < pose_idx else "○"
                    print(f"  {status} {i+1:2d}. {p.name:<20} {p.joint_angles_deg}")
                continue
            
            elif command == 'm':
                # Measure current position without moving
                print("[INFO] Messe aktuelle Position...")
                pos, curr, curr_std, torque, rnea = measurement.measure_holding_torque()
                measurement.add_measurement(
                    f"{pose.name}_current",
                    pos, curr, curr_std, torque, rnea
                )
                # Don't increment pose_idx
                continue
            
            elif command == '':
                # Default: Move and measure
                success = measurement.move_to_pose(pose.joint_angles_rad)
                
                if not success:
                    retry = input("    Bewegung fehlgeschlagen. Trotzdem messen? [j/N]: ").strip().lower()
                    if retry != 'j':
                        continue
                
                # Measure holding torque
                pos, curr, curr_std, torque, rnea = measurement.measure_holding_torque()
                measurement.add_measurement(pose.name, pos, curr, curr_std, torque, rnea)
                
                pose_idx += 1
            
            else:
                print(f"[WARN] Unbekannter Befehl: '{command}'")
    
    except KeyboardInterrupt:
        print("\n\n[ABBRUCH] Strg+C gedrückt")
    
    # Save results
    if measurement.measurements:
        output_file = os.path.join(
            output_dir,
            f"holding_torques_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        measurement.save_results(output_file)
        measurement.print_summary()
    else:
        print("\n[INFO] Keine Messungen durchgeführt - nichts zu speichern")


def main():
    parser = argparse.ArgumentParser(
        description='Misst Haltemomente des GELLO-Roboters an verschiedenen Posen'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ur5e_gello_factr_sim.yaml',
        help='Pfad zur YAML-Konfigurationsdatei'
    )
    parser.add_argument(
        '--poses',
        type=str,
        choices=['default', 'quick'],
        default='default',
        help='Welche Posen-Set zu verwenden (default: alle, quick: nur Test-Posen)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='recordings',
        help='Ausgabe-Verzeichnis für Messungen (relativ zum scripts-Verzeichnis)'
    )
    parser.add_argument(
        '--profile-velocity',
        type=int,
        default=40,
        help='Profil-Geschwindigkeit für Bewegungen (niedriger = langsamer)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          GELLO - HALTEMOMENT-MESSUNG                              ║
║                                                                   ║
║  Methode: Position-Modus → Haltestrom auslesen                   ║
║  Ziel: Validierung der Gravity Compensation Parameter            ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Load configuration
    print(f"[INFO] Lade Konfiguration: {args.config}")
    config = load_config(args.config)
    
    # Select poses
    if args.poses == 'quick':
        poses = get_quick_test_poses()
        print(f"[INFO] Verwende Quick-Test Posen ({len(poses)} Posen)")
    else:
        poses = get_default_calibration_poses()
        print(f"[INFO] Verwende Standard-Posen ({len(poses)} Posen)")
    
    # Setup driver
    driver = setup_driver_from_config(config)
    
    # Setup measurement class
    # Versuche URDF aus Config zu laden
    urdf_path = config.get('arm_teleop', {}).get('leader_urdf')
    if urdf_path:
        # Auflösen
        abs_config_path = os.path.abspath(args.config)
        config_dir = os.path.dirname(abs_config_path)
        repo_root = os.path.dirname(os.path.dirname(abs_config_path)) # Assuming configs/file.yaml
        if 'scripts' in os.getcwd():
            # Fallback if running from scripts dir
            repo_root = os.path.dirname(os.getcwd())

        urdf_candidates = [
            os.path.join(config_dir, urdf_path), # Relative to config
            os.path.join(repo_root, urdf_path),  # Relative to repo root
            os.path.abspath(urdf_path),          # Relative to CWD
            urdf_path
        ]
        
        resolved_urdf = None
        for cand in urdf_candidates:
            if os.path.exists(cand):
                resolved_urdf = cand
                break
        
        if resolved_urdf:
             print(f"[INFO] Gefundene URDF: {resolved_urdf}")
        else:
             print(f"[WARN] URDF nicht gefunden: {urdf_path}")
    else:
        resolved_urdf = None

    measurement = HoldingTorqueMeasurement(
        driver=driver,
        servo_types=config['dynamixel']['servo_types'],
        num_arm_joints=config['arm_teleop']['num_arm_joints'],
        urdf_path=resolved_urdf
    )
    
    try:
        # Configure motors
        configure_position_mode(driver, slow_velocity=args.profile_velocity)
        
        print(f"[DEBUG] Übergabe Config an run_interactive: Keys={list(config.keys())}")
        if 'arm_teleop' in config:
             print(f"[DEBUG] arm_teleop found. Keys={list(config['arm_teleop'].keys())}")
        
        # Run measurement
        run_interactive_measurement(
            measurement, 
            poses, 
            args.output_dir,
            calibration_config=config 
        )
        
    finally:
        # Cleanup
        print("\n[INFO] Deaktiviere Motoren...")
        driver.set_torque_mode(False)
        driver.close()
        print("[INFO] Verbindung geschlossen")


if __name__ == "__main__":
    main()
