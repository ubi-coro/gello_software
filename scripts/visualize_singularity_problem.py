#!/usr/bin/env python3
"""
GELLO Shoulder Singularity Diagnostic Tool

This script helps understand the gravity compensation behavior near θ₂ = ±180°
and diagnoses why the arm might "fall" to one side near this configuration.

Key insight: At θ₂ = 180°, the arm is in UNSTABLE equilibrium.
The gravity compensation torque is ~0, but any small deviation causes
the arm to accelerate away from 180°.
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pinocchio as pin
import yaml

from gello.dynamixel.driver import DynamixelDriver


class SingularityDiagnostic:
    """Diagnose and visualize gravity compensation near shoulder singularity."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()
        self._setup_model()
        self._setup_hardware()
        self._calibrate()
        
    def _load_config(self):
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        self.calibration_pos = np.array(
            self.config["arm_teleop"]["initialization"]["calibration_joint_pos"]
        )
        
        # Gravity comp settings
        gc_cfg = self.config.get("controller", {}).get("gravity_comp", {})
        self.gc_gain = gc_cfg.get("gain", 1.0)
        self.gc_per_joint = np.array(gc_cfg.get("gain_per_joint", [1.0] * self.num_arm_joints))
        
    def _setup_model(self):
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]
        workspace_root = Path(__file__).parent.parent.resolve()
        
        # Possible candidates for the URDF path
        candidates = [
            Path(urdf_filename), # Absolute or relative to CWD
            Path(self.config_path).parent / urdf_filename, # Relative to config
            workspace_root / urdf_filename, # Relative to workspace root
        ]
        
        # Add a recursive search if not found in obvious places
        urdf_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                urdf_path = c.resolve()
                break
        
        if urdf_path is None:
             # Search the workspace for this filename
             print(f"URDF {urdf_filename} not found in obvious locations. Searching workspace...")
             filename = Path(urdf_filename).name
             matches = list(workspace_root.glob(f"**/{filename}"))
             if matches:
                 # Pick the best match (shortest path or first one)
                 urdf_path = matches[0].resolve()
        
        if urdf_path is None:
             raise FileNotFoundError(f"Could not find URDF file: {urdf_filename} in {workspace_root}")

        print(f"Loading URDF: {urdf_path}")
        self.model, _, _ = pin.buildModelsFromUrdf(str(urdf_path), str(urdf_path.parent))
        self.data = self.model.createData()
        
    def _setup_hardware(self):
        servo_types = self.config["dynamixel"]["servo_types"]
        port = self.config["dynamixel"]["dynamixel_port"]
        if not port.startswith("/"):
            port = "/dev/serial/by-id/" + port
        
        baudrate = self.config["dynamixel"].get("baudrate", 57600)
        
        print(f"Connecting to Dynamixel on {port}...")
        self.driver = DynamixelDriver(
            list(range(1, len(servo_types) + 1)),
            servo_types,
            port,
            baudrate=baudrate
        )
        self.driver.set_torque_mode(False)
        print("Connected (torque disabled)")
        
    def _calibrate(self):
        print("Calibrating offsets...")
        for _ in range(10):
            self.driver.get_positions_and_velocities()
        
        raw_pos, _ = self.driver.get_positions_and_velocities()
        
        self.offsets = []
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            for offset in np.linspace(-10*np.pi, 10*np.pi, 721):
                calibrated = self.joint_signs[i] * (raw_pos[i] - offset)
                error = abs(calibrated - self.calibration_pos[i])
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.offsets.append(best_offset)
        self.offsets = np.array(self.offsets)
        print("Calibration complete")
        
    def get_joint_pos(self) -> np.ndarray:
        raw_pos, _ = self.driver.get_positions_and_velocities()
        return (raw_pos[:self.num_arm_joints] - self.offsets) * self.joint_signs[:self.num_arm_joints]
    
    def compute_gravity_torque(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torque for given joint positions."""
        q = np.zeros(self.model.nq)
        q[:self.num_arm_joints] = joint_pos
        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)
        
        tau = pin.rnea(self.model, self.data, q, v, a)
        tau_arm = np.array(tau[:self.num_arm_joints])
        
        # Apply same gains as gravity_compensation.py
        tau_arm *= self.gc_gain
        tau_arm *= self.gc_per_joint
        
        return tau_arm
    
    def analyze_singularity(self):
        """Analyze and explain the singularity behavior."""
        print("\n" + "="*70)
        print("SHOULDER SINGULARITY ANALYSIS")
        print("="*70)
        
        # Get current position
        joint_pos = self.get_joint_pos()
        theta2_deg = np.rad2deg(joint_pos[1])
        
        print(f"\nCurrent shoulder angle θ₂ = {theta2_deg:+.1f}°")
        
        # Compute torques at key positions
        print("\n" + "-"*70)
        print("GRAVITY TORQUE AT KEY SHOULDER ANGLES:")
        print("-"*70)
        print(f"{'θ₂ [deg]':>10} | {'τ₂ [Nm]':>10} | {'Direction':>15} | {'Stability':>15}")
        print("-"*70)
        
        test_angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        
        for angle_deg in test_angles:
            test_pos = joint_pos.copy()
            test_pos[1] = np.deg2rad(angle_deg)
            tau = self.compute_gravity_torque(test_pos)
            tau2 = tau[1]
            
            # Determine direction
            if abs(tau2) < 0.01:
                direction = "ZERO (equilibrium)"
                stability = "UNSTABLE" if abs(angle_deg) == 180 or angle_deg == 0 else "NEUTRAL"
            elif tau2 > 0:
                direction = "→ Positive"
                stability = ""
            else:
                direction = "← Negative"
                stability = ""
            
            marker = " ← YOU ARE HERE" if abs(theta2_deg - angle_deg) < 10 else ""
            print(f"{angle_deg:>10.0f} | {tau2:>+10.4f} | {direction:>15} | {stability:>15}{marker}")
        
        print("-"*70)
        
        # Explain the physics
        print("\n" + "="*70)
        print("PHYSICS EXPLANATION")
        print("="*70)
        print("""
At θ₂ = ±180° (shoulder pointing straight up):
  
  1. GRAVITY TORQUE IS ZERO
     - The arm's center of mass is directly above/below the joint axis
     - There's no lever arm for gravity to create torque
     
  2. THIS IS AN UNSTABLE EQUILIBRIUM
     - Like balancing a pencil on its tip
     - Any tiny deviation creates a torque that pushes AWAY from 180°
     
  3. WHY THE ARM "FALLS":
     - Small model errors or calibration offsets
     - Friction/stiction in the joint
     - Sensor noise in position reading
     
     These create a small net torque that accelerates the arm away from 180°.
     
  4. THE DERIVATIVE IS MAXIMUM AT 180°:
     - dτ/dθ is largest near the singularity
     - Small position changes → large torque changes
     - This amplifies any disturbance

""")
        
        # Compute actual derivative at current position
        delta = 0.01  # 0.01 rad ≈ 0.57°
        test_pos_plus = joint_pos.copy()
        test_pos_plus[1] += delta
        test_pos_minus = joint_pos.copy()
        test_pos_minus[1] -= delta
        
        tau_plus = self.compute_gravity_torque(test_pos_plus)[1]
        tau_minus = self.compute_gravity_torque(test_pos_minus)[1]
        dtau_dtheta = (tau_plus - tau_minus) / (2 * delta)
        
        print(f"\nAt your current position θ₂ = {theta2_deg:+.1f}°:")
        print(f"  τ₂ = {self.compute_gravity_torque(joint_pos)[1]:+.4f} Nm")
        print(f"  dτ₂/dθ₂ = {dtau_dtheta:+.4f} Nm/rad = {dtau_dtheta * np.pi/180:+.6f} Nm/deg")
        
        dist_from_180 = min(abs(theta2_deg - 180), abs(theta2_deg + 180))
        if dist_from_180 < 15:
            print(f"\n  ⚠️  WARNING: You are {dist_from_180:.1f}° from singularity!")
            print(f"      Sensitivity is HIGH. Small errors cause large torque changes.")
        
    def run_live_visualization(self):
        """Run live visualization with clear annotations."""
        print("\n" + "="*70)
        print("LIVE VISUALIZATION")
        print("="*70)
        print("Move the shoulder joint slowly through ±180°")
        print("Watch how the torque changes sign and the arm becomes unstable")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Shoulder Singularity Analysis - Live", fontsize=14)
        
        # === Plot 1: Theoretical torque curve ===
        theta_range = np.linspace(-np.pi, np.pi, 361)
        tau_theoretical = []
        
        base_pos = self.get_joint_pos()
        for theta in theta_range:
            test_pos = base_pos.copy()
            test_pos[1] = theta
            tau = self.compute_gravity_torque(test_pos)
            tau_theoretical.append(tau[1])
        
        tau_theoretical = np.array(tau_theoretical)
        
        axes[0, 0].plot(np.rad2deg(theta_range), tau_theoretical, 'b-', linewidth=2, label='τ_gravity(θ₂)')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 0].axvline(x=180, color='r', linestyle='--', linewidth=2, label='Singularity')
        axes[0, 0].axvline(x=-180, color='r', linestyle='--', linewidth=2)
        axes[0, 0].axvspan(165, 195, alpha=0.2, color='red')
        axes[0, 0].axvspan(-195, -165, alpha=0.2, color='red')
        
        # Current position marker
        current_marker, = axes[0, 0].plot([], [], 'go', markersize=15, label='Current position')
        
        axes[0, 0].set_xlabel("θ₂ [deg]")
        axes[0, 0].set_ylabel("τ₂ [Nm]")
        axes[0, 0].set_title("Gravity Torque vs. Shoulder Angle\n(Red zone = unstable region)")
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(-200, 200)
        
        # === Plot 2: Torque direction explanation ===
        axes[0, 1].set_xlim(-1, 1)
        axes[0, 1].set_ylim(-1, 1)
        axes[0, 1].set_aspect('equal')
        axes[0, 1].axis('off')
        axes[0, 1].set_title("Torque Direction Explanation")
        
        # Draw a simple arm representation
        arm_line, = axes[0, 1].plot([0, 0], [0, 0.7], 'b-', linewidth=8, solid_capstyle='round')
        joint_circle = patches.Circle((0, 0), 0.08, color='gray', zorder=5)
        axes[0, 1].add_patch(joint_circle)
        
        # Gravity arrow
        axes[0, 1].annotate('', xy=(0.5, -0.5), xytext=(0.5, 0),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
        axes[0, 1].text(0.55, -0.25, 'Gravity', fontsize=10, color='green')
        
        # Torque arrow (will be updated)
        torque_arrow = axes[0, 1].annotate('', xy=(0, 0), xytext=(0, 0),
                                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
        torque_text = axes[0, 1].text(0, -0.9, '', fontsize=12, ha='center', fontweight='bold')
        
        # === Plot 3: Real-time angle ===
        history_len = 500
        time_history = deque(maxlen=history_len)
        theta2_history = deque(maxlen=history_len)
        
        line_theta, = axes[1, 0].plot([], [], 'b-', linewidth=1.5)
        axes[1, 0].axhline(y=180, color='r', linestyle='--', alpha=0.7, label='Singularity')
        axes[1, 0].axhline(y=-180, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].axhspan(165, 195, alpha=0.15, color='red')
        axes[1, 0].axhspan(-195, -165, alpha=0.15, color='red')
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("θ₂ [deg]")
        axes[1, 0].set_title("Shoulder Angle vs. Time")
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # === Plot 4: Real-time torque ===
        tau2_history = deque(maxlen=history_len)
        
        line_tau, = axes[1, 1].plot([], [], 'b-', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("τ₂ [Nm]")
        axes[1, 1].set_title("Gravity Compensation Torque vs. Time")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Status text
        status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=(0, 0.05, 1, 0.95))
        
        start_time = time.time()
        
        try:
            while True:
                # Get current state
                joint_pos = self.get_joint_pos()
                tau = self.compute_gravity_torque(joint_pos)
                
                theta2_deg = np.rad2deg(joint_pos[1])
                tau2 = tau[1]
                t = time.time() - start_time
                
                # Update histories
                time_history.append(t)
                theta2_history.append(theta2_deg)
                tau2_history.append(tau2)
                
                # Update current position marker on theoretical curve
                current_marker.set_data([theta2_deg], [tau2])
                
                # Update arm visualization
                arm_angle = joint_pos[1] + np.pi/2  # Convert to visual angle
                arm_x = 0.6 * np.cos(arm_angle)
                arm_y = 0.6 * np.sin(arm_angle)
                arm_line.set_data([0, arm_x], [0, arm_y])
                
                # Update torque arrow
                if abs(tau2) > 0.01:
                    arrow_scale = min(abs(tau2) * 2, 0.5)  # Scale arrow size
                    if tau2 > 0:
                        torque_arrow.xy = (arrow_scale, 0)
                        torque_arrow.xytext = (-arrow_scale, 0)
                        torque_text.set_text(f'τ = {tau2:+.3f} Nm\n→ Pushes arm THIS way')
                        torque_text.set_color('red')
                    else:
                        torque_arrow.xy = (-arrow_scale, 0)
                        torque_arrow.xytext = (arrow_scale, 0)
                        torque_text.set_text(f'τ = {tau2:+.3f} Nm\n← Pushes arm THIS way')
                        torque_text.set_color('blue')
                else:
                    torque_arrow.xy = (0, 0)
                    torque_arrow.xytext = (0, 0)
                    torque_text.set_text(f'τ ≈ 0 Nm\n⚠️ UNSTABLE EQUILIBRIUM')
                    torque_text.set_color('red')
                
                # Update time series plots
                if len(time_history) > 2:
                    t_arr = np.array(time_history)
                    t_rel = t_arr - t_arr[0]
                    
                    line_theta.set_data(t_rel, np.array(theta2_history))
                    axes[1, 0].set_xlim(t_rel[0], max(t_rel[-1], 1))
                    axes[1, 0].set_ylim(min(theta2_history) - 20, max(theta2_history) + 20)
                    
                    line_tau.set_data(t_rel, np.array(tau2_history))
                    axes[1, 1].set_xlim(t_rel[0], max(t_rel[-1], 1))
                    tau_range = max(abs(min(tau2_history)), abs(max(tau2_history)), 0.1)
                    axes[1, 1].set_ylim(-tau_range * 1.2, tau_range * 1.2)
                
                # Update status
                dist_from_180 = min(abs(theta2_deg - 180), abs(theta2_deg + 180))
                if dist_from_180 < 5:
                    status = f"🔴 CRITICAL: θ₂ = {theta2_deg:+.1f}° | τ₂ = {tau2:+.4f} Nm | IN SINGULARITY ZONE!"
                    status_text.set_backgroundcolor('red')
                elif dist_from_180 < 15:
                    status = f"🟠 WARNING: θ₂ = {theta2_deg:+.1f}° | τ₂ = {tau2:+.4f} Nm | Approaching singularity"
                    status_text.set_backgroundcolor('orange')
                else:
                    status = f"🟢 NORMAL: θ₂ = {theta2_deg:+.1f}° | τ₂ = {tau2:+.4f} Nm"
                    status_text.set_backgroundcolor('lightgreen')
                status_text.set_text(status)
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.driver.close()
            plt.ioff()
            plt.show()
    
    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()


def main():
    parser = argparse.ArgumentParser(description="GELLO Shoulder Singularity Diagnostic")
    parser.add_argument("--config", "-c", default="configs/ur5e_gello_factr_hw.yaml")
    parser.add_argument("--analyze-only", "-a", action="store_true",
                        help="Only print analysis, no live visualization")
    args = parser.parse_args()
    
    # Handle relative path
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        script_dir = Path(__file__).parent
        config_path = str(script_dir / config_path)
    
    diag = SingularityDiagnostic(config_path)
    
    try:
        diag.analyze_singularity()
        
        if not args.analyze_only:
            diag.run_live_visualization()
    finally:
        diag.close()


if __name__ == "__main__":
    main()
