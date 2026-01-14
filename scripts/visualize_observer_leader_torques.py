#!/usr/bin/env python3
"""
GELLO Leader Observer Torque Visualizer

Visualizes estimated external joint torques from the sensorless force observer
while moving the GELLO leader arm.

Usage:
    python visualize_gello_observer.py --config configs/ur5e_gello_factr_hw.yaml
"""

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gello.dynamixel.driver import DynamixelDriver
from gello.bilat_4ch.gello_ur5e_observer import LeaderObserver

class GelloObserverVisualizer:
    """Real-time visualization of GELLO observer estimated torques."""

    def __init__(self, config_path: str, history_seconds: float = 10.0):
        self.config_path = config_path
        self.running = False
        self.driver: Optional[DynamixelDriver] = None

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup parameters
        self.dt = 1.0 / self.config["controller"]["frequency"]
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.calibration_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["calibration_joint_pos"]
        )

        # Joint signs for coordinate transformation
        self.joint_signs = np.array(
            self.config["dynamixel"]["joint_signs"], dtype=float
        )

        # History for plotting
        self.history_len = int(history_seconds / self.dt)
        self.time_history = deque(maxlen=self.history_len)
        self.tau_ext_history = [deque(maxlen=self.history_len) for _ in range(self.num_arm_joints)]
        self.dq_hat_history = [deque(maxlen=self.history_len) for _ in range(self.num_arm_joints)]
        self.q_history = [deque(maxlen=self.history_len) for _ in range(self.num_arm_joints)]

        # Initialize hardware
        self._setup_dynamixel()
        self._calibrate_offsets()
        self._setup_observer()
        self._setup_plot()

    def _setup_dynamixel(self) -> None:
        """Initialize Dynamixel driver."""
        servo_types = self.config["dynamixel"]["servo_types"]
        port_config = self.config["dynamixel"]["dynamixel_port"]
        
        if port_config.startswith("/"):
            port = port_config
        else:
            port = "/dev/serial/by-id/" + port_config

        joint_ids = list(range(1, len(servo_types) + 1))
        baudrate = self.config["dynamixel"].get("baudrate", 57600)

        print(f"Connecting to Dynamixel on {port}...")
        self.driver = DynamixelDriver(
            joint_ids, servo_types, port, baudrate=baudrate
        )
        # Keep torque disabled - we just want to read
        self.driver.set_torque_mode(False)
        print("Dynamixel connected (torque disabled for free movement)")

    def _calibrate_offsets(self) -> None:
        """Calibrate joint offsets (simplified version)."""
        print("Calibrating joint offsets...")
        
        # Warm up readings
        for _ in range(10):
            self.driver.get_positions_and_velocities()

        curr_joints, _ = self.driver.get_positions_and_velocities()

        # Find offsets that minimize error to calibration pose
        self.joint_offsets = []
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            target = self.calibration_joint_pos[i]
            sign = self.joint_signs[i]
            raw = curr_joints[i]

            for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
                calibrated = sign * (raw - offset)
                error = abs(calibrated - target)
                if error < best_error:
                    best_error = error
                    best_offset = offset

            self.joint_offsets.append(best_offset)
            print(f"  Joint {i+1}: offset={np.rad2deg(best_offset):+.1f}°, error={np.rad2deg(best_error):.2f}°")

        self.joint_offsets = np.array(self.joint_offsets)
        print("Calibration complete!")

    def _setup_observer(self) -> None:
        """Initialize the force/velocity observer."""
        # Find URDF path
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]
        config_dir = Path(self.config_path).parent
        urdf_path = config_dir / urdf_filename

        if not urdf_path.exists():
            # Fall back to root relative
            repo_root = Path(__file__).resolve().parent.parent
            urdf_path = repo_root / urdf_filename

        if not urdf_path.exists():
            # Try searching in scripts/urdf for fallback
            urdf_path = Path(__file__).resolve().parent / "urdf" / Path(urdf_filename).name

        print(f"Loading URDF: {urdf_path}")

        # Get initial joint position
        q0 = self._get_joint_positions()

        # Observer parameters from paper
        omega_c = 50.0  # Cut-off frequency [rad/s]

        self.observer = LeaderObserver(
            urdf_path=str(urdf_path),
            omega_c=omega_c,
            dt=self.dt,
            q0=q0,
        )
        print(f"Observer initialized (ωc={omega_c} rad/s, dt={self.dt*1000:.1f}ms)")

    def _get_joint_positions(self) -> np.ndarray:
        """Get calibrated joint positions."""
        raw_pos, _ = self.driver.get_positions_and_velocities()
        pos = (raw_pos[:self.num_arm_joints] - self.joint_offsets[:self.num_arm_joints]) * self.joint_signs[:self.num_arm_joints]
        return pos

    def _get_joint_states(self):
        """Get calibrated joint positions and velocities."""
        raw_pos, raw_vel = self.driver.get_positions_and_velocities()
        pos = (raw_pos[:self.num_arm_joints] - self.joint_offsets[:self.num_arm_joints]) * self.joint_signs[:self.num_arm_joints]
        vel = raw_vel[:self.num_arm_joints] * self.joint_signs[:self.num_arm_joints]
        return pos, vel

    def _setup_plot(self) -> None:
        """Setup matplotlib figure for real-time plotting."""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        self.fig.suptitle("GELLO Observer: Estimated External Torques", fontsize=14)

        # Joint colors
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.num_arm_joints))
        self.joint_names = [f"J{i+1}" for i in range(self.num_arm_joints)]

        # Torque plot
        self.axes[0].set_ylabel("τ_ext [Nm]")
        self.axes[0].set_title("Estimated External Joint Torques (from Observer)")
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Velocity plot
        self.axes[1].set_ylabel("dq_hat [rad/s]")
        self.axes[1].set_title("Estimated Joint Velocities (from Observer)")
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Position plot
        self.axes[2].set_ylabel("q [rad]")
        self.axes[2].set_xlabel("Time [s]")
        self.axes[2].set_title("Joint Positions")
        self.axes[2].grid(True, alpha=0.3)

        # Initialize line objects
        self.tau_lines = []
        self.dq_lines = []
        self.q_lines = []

        for i in range(self.num_arm_joints):
            line_tau, = self.axes[0].plot([], [], color=self.colors[i], label=self.joint_names[i], linewidth=1.5)
            line_dq, = self.axes[1].plot([], [], color=self.colors[i], label=self.joint_names[i], linewidth=1.5)
            line_q, = self.axes[2].plot([], [], color=self.colors[i], label=self.joint_names[i], linewidth=1.5)
            self.tau_lines.append(line_tau)
            self.dq_lines.append(line_dq)
            self.q_lines.append(line_q)

        # Legends
        self.axes[0].legend(loc='upper right', ncol=self.num_arm_joints, fontsize=8)
        self.axes[1].legend(loc='upper right', ncol=self.num_arm_joints, fontsize=8)
        self.axes[2].legend(loc='upper right', ncol=self.num_arm_joints, fontsize=8)

        # Text annotations for current values
        self.tau_text = self.axes[0].text(0.02, 0.95, "", transform=self.axes[0].transAxes,
                                           fontsize=9, verticalalignment='top', fontfamily='monospace')
        self.dq_text = self.axes[1].text(0.02, 0.95, "", transform=self.axes[1].transAxes,
                                          fontsize=9, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    def _update_plot(self) -> None:
        """Update the plot with new data."""
        if len(self.time_history) < 2:
            return

        t_array = np.array(self.time_history)
        t_min = t_array[0]
        t_rel = t_array - t_min

        # Update lines
        for i in range(self.num_arm_joints):
            tau_array = np.array(self.tau_ext_history[i])
            dq_array = np.array(self.dq_hat_history[i])
            q_array = np.array(self.q_history[i])

            self.tau_lines[i].set_data(t_rel, tau_array)
            self.dq_lines[i].set_data(t_rel, dq_array)
            self.q_lines[i].set_data(t_rel, q_array)

        # Update axis limits
        for ax in self.axes:
            ax.set_xlim(t_rel[0], t_rel[-1])
            ax.relim()
            ax.autoscale_view(scalex=False)

        # Update text annotations with current values
        if len(self.tau_ext_history[0]) > 0:
            tau_curr = [self.tau_ext_history[i][-1] for i in range(self.num_arm_joints)]
            dq_curr = [self.dq_hat_history[i][-1] for i in range(self.num_arm_joints)]
            
            tau_str = "τ_ext: " + " ".join([f"{t:+.3f}" for t in tau_curr]) + " Nm"
            dq_str = "dq_hat: " + " ".join([f"{v:+.3f}" for v in dq_curr]) + " rad/s"
            
            self.tau_text.set_text(tau_str)
            self.dq_text.set_text(dq_str)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self) -> None:
        """Main visualization loop."""
        print("\n" + "="*60)
        print("GELLO Observer Torque Visualizer")
        print("="*60)
        print("Move the GELLO arm to see estimated external torques.")
        print("The observer estimates forces without a force/torque sensor!")
        print("Press Ctrl+C to stop.")
        print("="*60 + "\n")

        self.running = True
        start_time = time.time()
        loop_count = 0
        
        # Since we're not commanding torques, use zero as tau_cmd
        # In a real bilateral control setup, this would be the actual commanded torque
        tau_cmd = np.zeros(self.num_arm_joints)
        dq_ref = np.zeros(self.num_arm_joints)

        try:
            while self.running:
                loop_start = time.time()

                # Get current joint states
                q, dq = self._get_joint_states()

                # Run observer update
                dq_hat, tau_ext_hat = self.observer.update(
                    q_measured=q,
                    dq_measured=dq,
                    tau_cmd=tau_cmd,
                    dq_ref=dq_ref,
                )

                # Store history
                current_time = time.time() - start_time
                self.time_history.append(current_time)
                for i in range(self.num_arm_joints):
                    self.tau_ext_history[i].append(tau_ext_hat[i])
                    self.dq_hat_history[i].append(dq_hat[i])
                    self.q_history[i].append(q[i])

                # Update plot every N iterations (for performance)
                loop_count += 1
                if loop_count % 10 == 0:
                    self._update_plot()

                # Maintain loop timing
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown."""
        self.running = False
        if self.driver is not None:
            self.driver.set_torque_mode(False)
            self.driver.close()
        plt.ioff()
        plt.close('all')
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="GELLO Observer Torque Visualizer")
    parser.add_argument(
        "--config", "-c",
        default="configs/ur5e_gello_factr_hw.yaml",
        help="Path to GELLO config YAML"
    )
    parser.add_argument(
        "--history", "-t",
        type=float,
        default=10.0,
        help="History length in seconds (default: 10)"
    )
    args = parser.parse_args()

    # Signal handler
    visualizer = None
    def signal_handler(signum, frame):
        if visualizer:
            visualizer.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    visualizer = GelloObserverVisualizer(args.config, history_seconds=args.history)
    visualizer.run()


if __name__ == "__main__":
    main()