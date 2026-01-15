#!/usr/bin/env python3
"""
UR5e External Joint Torque Visualizer

Visualizes external joint torques from the UR5e robot using:
- rtde_control.getJointTorques() - External torques (gravity/friction compensated)
- rtde_receive.getTargetMoment() - Target torques from controller

Usage:
    python visualize_ur5e_torques.py --ip 192.168.1.10
    python visualize_ur5e_torques.py --ip 192.168.1.10 --freedrive
"""

import argparse
import signal
import time
import math
from collections import deque
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("ur_rtde not found. Please install it with: pip install ur_rtde")
    exit(1)


class SignalFilter:
    """
    Lightweight signal filtering suite.
    Supported: 'ema', 'mean', 'butter', 'kalman', 'one_euro'.
    """
    def __init__(
        self, 
        filter_type: str = "none", 
        sampling_rate: float = 125.0,
        # EMA / Mean args
        alpha: float = 0.2, 
        window: int = 5,
        # Butterworth args
        cutoff_hz: float = 10.0,
        # Kalman args (process noise covariance, measurement noise covariance)
        q_process: float = 1e-5,
        r_measure: float = 1e-2,
        # One Euro args
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0
    ):
        self.type = filter_type.lower()
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # --- EMA State ---
        self.alpha = alpha
        self.ema_state: Optional[np.ndarray] = None
        
        # --- Mean State ---
        self.window = window
        self.buffer = deque(maxlen=window)

        # --- Butterworth State (2nd order Low Pass) ---
        # Coefficients calculated manually to avoid scipy dependency
        if self.type == "butter":
            # Precompute coefficients for 2nd order Butterworth Lowpass
            # https://stackoverflow.com/a/20936214
             # f_c = cutoff / fs
            w_c = 2 * np.pi * cutoff_hz
            # Bilinear transform approximation terms
            # This is a bit complex to do generic without scipy, using a simple 1st order LPF 
            # (which is EMA) is safer. Let's use a standard implementation if we can, 
            # or finding coefficients for a fixed normalized frequency.
            # actually, let's implement a simple 2nd order section.
            
            # Derived from simple discretization
            t = self.dt
            wc = 2 * np.pi * cutoff_hz
            c = 1.0 / (np.tan(wc * t / 2.0))
            
            self.a0 = 1.0 / (1.0 + np.sqrt(2)*c + c**2)
            self.b0 = 1.0 * self.a0
            self.b1 = 2.0 * self.a0
            self.b2 = 1.0 * self.a0
            self.a1 = 2.0 * (1.0 - c**2) * self.a0
            self.a2 = (1.0 - np.sqrt(2)*c + c**2) * self.a0
            
            self.x_hist = [None, None] # x[n-1], x[n-2]
            self.y_hist = [None, None] # y[n-1], y[n-2]

        # --- Kalman State (1D constant model) ---
        self.q_process = q_process
        self.r_measure = r_measure
        self.P: Optional[np.ndarray] = None # Error covariance
        self.x_est: Optional[np.ndarray] = None # Estimated state

        # --- One Euro State ---
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.one_euro_x: Optional[np.ndarray] = None
        self.one_euro_dx: Optional[np.ndarray] = None
        self.last_time = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def _low_pass(self, alpha, x, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def update(self, sample: np.ndarray) -> np.ndarray:
        if self.type == "none":
            return sample
        
        # --- EMA ---
        if self.type == "ema":
            if self.ema_state is None:
                self.ema_state = sample
            else:
                self.ema_state = self.alpha * sample + (1.0 - self.alpha) * self.ema_state
            return self.ema_state
            
        # --- Moving Average ---
        if self.type == "mean":
            self.buffer.append(sample)
            if len(self.buffer) == 0:
                return sample
            return np.mean(self.buffer, axis=0)

        # --- Butterworth (2nd Order) ---
        if self.type == "butter":
            if self.x_hist[0] is None:
                # Initialize history with current sample to avoid startup transient
                self.x_hist = [sample, sample]
                self.y_hist = [sample, sample]
                return sample
            
            # Direct Difference Equation:
            # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            out = (self.b0 * sample + 
                   self.b1 * self.x_hist[0] + 
                   self.b2 * self.x_hist[1] - 
                   self.a1 * self.y_hist[0] - 
                   self.a2 * self.y_hist[1])
            
            # Shift history
            self.x_hist[1] = self.x_hist[0]
            self.x_hist[0] = sample
            self.y_hist[1] = self.y_hist[0]
            self.y_hist[0] = out
            return out

        # --- Kalman Filter (Simple 1D) ---
        if self.type == "kalman":
            # Initialize
            if self.x_est is None:
                self.x_est = sample
                self.P = np.ones_like(sample) * 1.0
                return sample
            
            # Predict (Constant model: x_k = x_k-1)
            # x_pred = x_est
            # P_pred = P + Q
            P_pred = self.P + self.q_process
            
            # Update
            # K = P_pred / (P_pred + R)
            K = P_pred / (P_pred + self.r_measure)
            
            # x_est = x_pred + K * (z - x_pred)
            self.x_est = self.x_est + K * (sample - self.x_est)
            
            # P = (1 - K) * P_pred
            self.P = (1.0 - K) * P_pred
            
            return self.x_est

        # --- One Euro Filter ---
        if self.type == "one_euro":
            now = time.time()
            if self.last_time is None:
                dt = self.dt # Use default dt on first step
            else:
                dt = now - self.last_time
            self.last_time = now

            if self.one_euro_x is None:
                self.one_euro_x = sample
                self.one_euro_dx = np.zeros_like(sample)
                return sample

            # Filter derivative
            dx_raw = (sample - self.one_euro_x) / dt
            alpha_d = self._smoothing_factor(dt, self.d_cutoff)
            self.one_euro_dx = self._low_pass(alpha_d, dx_raw, self.one_euro_dx)

            # Filter signal
            # Use derivative magnitude to tune cutoff
            # The faster we move, the higher the cutoff (less filtering)
            cutoff = self.min_cutoff + self.beta * np.abs(self.one_euro_dx)
            alpha = self._smoothing_factor(dt, cutoff)
            self.one_euro_x = self._low_pass(alpha, sample, self.one_euro_x)
            
            return self.one_euro_x

        return sample


class UR5eTorqueVisualizer:
    """Real-time visualization of UR5e external joint torques."""

    def __init__(
        self,
        robot_ip: str,
        mode: str = "direct",
        history_seconds: float = 10.0,
        sample_rate_hz: float = 125.0,
        enable_freedrive: bool = False,
        tare: bool = False,
        filter_type: str = "none",
        filter_alpha: float = 0.2,
        filter_window: int = 5,
        filter_cutoff: float = 10.0,
        filter_beta: float = 0.5,
        filter_min_cutoff: float = 1.0,
    ):
        self.robot_ip = robot_ip
        self.mode = mode
        self.running = False
        self.enable_freedrive = enable_freedrive
        self.tare = tare
        self.dt = 1.0 / sample_rate_hz
        self.num_joints = 6
        self.torque_offsets = np.zeros(self.num_joints)

        # Initialize Filter
        self.filter = SignalFilter(
            filter_type=filter_type,
            sampling_rate=sample_rate_hz,
            alpha=filter_alpha,
            window=filter_window,
            cutoff_hz=filter_cutoff,
            beta=filter_beta,
            min_cutoff=filter_min_cutoff
        )

        # History for plotting
        self.history_len = int(history_seconds * sample_rate_hz)
        self.time_history = deque(maxlen=self.history_len)
        
        # Torque histories (only one needed based on mode)
        self.tau_history = [deque(maxlen=self.history_len) for _ in range(self.num_joints)]

        # Initialize robot connection
        self._connect_robot()
        self._setup_plot()

    def _connect_robot(self) -> None:
        """Connect to UR5e via RTDE."""
        # rtde imports are now global

        print(f"Connecting to UR5e at {self.robot_ip}...")
        
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            print("RTDE Receive interface connected")
        except Exception as e:
            raise RuntimeError(f"Failed to connect RTDE Receive: {e}")

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            print("RTDE Control interface connected")
            
            if self.enable_freedrive:
                self.rtde_c.freedriveMode()
                print("FREEDRIVE MODE ENABLED - Move the robot by hand")
        except Exception as e:
            raise RuntimeError(f"Failed to connect RTDE Control: {e}")

        # Print robot info
        self._print_robot_info()

    def _print_robot_info(self) -> None:
        """Print useful robot information."""
        print("\n" + "="*70)
        print("UR5e Robot Information")
        print("="*70)
        
        # Payload info
        try:
            self.payload_mass = self.rtde_r.getPayloadMass()
            print(f"Payload mass (kg):     {self.payload_mass:.3f}")
        except Exception:
            print("Payload info: Not available")

        # Current joint positions
        q = self.rtde_r.getActualQ()
        print(f"Joint positions (deg): {[f'{np.rad2deg(x):+.1f}' for x in q]}")
        
        if self.mode == "direct":
            tau_ext = self.rtde_c.getJointTorques()
            print(f"External torques (Nm): {[f'{x:+.2f}' for x in tau_ext]}")
        
        elif self.mode == "manual":
            tcp_force = np.array(self.rtde_r.getActualTCPForce())
            tau_manual = self._calculate_jacobian_torques_manual(np.array(q), tcp_force)
            print(f"Manual Jacobian (Nm):  {[f'{x:+.2f}' for x in tau_manual]}")

        elif self.mode == "semi-manual":
            # Just print a placeholder or do a one-off fetch
            print(f"Semi-manual Jacobian:  (Will be fetched at start of run)")

        print("="*70 + "\n")

    def _setup_plot(self) -> None:
        """Setup matplotlib figure for real-time plotting (Publication Style)."""
        # Configure publication-quality style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans'],
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 12,
            'lines.linewidth': 2.0,
            'grid.alpha': 0.3,
        })
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        title_map = {
            "direct": "External Joint Torques (Controller Reported)",
            "manual": "External Joint Torques (Calculated: $J^T F_{tcp}$)",
            "semi-manual": "External Joint Torques (Semi-Manual: $J_{fixed}^T F_{tcp}$)"
        }
        self.fig.suptitle(f"UR5e {title_map.get(self.mode, 'Unknown')}", fontsize=20, weight='bold')

        self.colors = plt.cm.tab10(np.linspace(0, 1, self.num_joints))
        self.joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]

        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Torque [Nm]")
        self.ax.grid(True)
        self.ax.axhline(y=0, color='k', linestyle='-', linewidth=1.0)

        # Initialize line objects
        self.lines = []
        for i in range(self.num_joints):
            line, = self.ax.plot([], [], color=self.colors[i], label=self.joint_names[i])
            self.lines.append(line)

        self.ax.legend(loc='upper right', ncol=1, framealpha=0.9)
        
        # Add a text box for statistics (RMS)
        self.stats_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

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
        for i in range(self.num_joints):
            tau = np.array(self.tau_history[i])
            self.lines[i].set_data(t_rel, tau)

        # Update axis limits
        self.ax.set_xlim(t_rel[0], t_rel[-1])
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)

        # Update stats
        if len(self.tau_history[0]) > 0:
            current_vals = [self.tau_history[i][-1] for i in range(self.num_joints)]
            rms = np.sqrt(np.mean(np.array(current_vals)**2))
            
            stats_str = f"RMS Torque: {rms:.2f} Nm\n"
            stats_str += "\n".join([f"{name}: {val:+.2f} Nm" for name, val in zip(self.joint_names, current_vals)])
            self.stats_text.set_text(stats_str)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _calculate_jacobian_torques_manual(self, q: np.ndarray, tcp_force: np.ndarray) -> np.ndarray:
        """Calculate joint torques using manual DH parameters and pre-fetched TCP force."""
        # UR5e DH parameters (standard/classical DH convention as per UR documentation)
        d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]  # meters
        a = [0, -0.425, -0.3922, 0, 0, 0]  # meters
        alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]  # radians

        # Compute transformation matrices and Jacobian
        T = np.eye(4)
        transforms = [T.copy()]

        for i in range(6):
            c = np.cos(q[i])
            s = np.sin(q[i])
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])

            T_i = np.array([
                [c, -s * ca, s * sa, a[i] * c],
                [s, c * ca, -c * sa, a[i] * s],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
            transforms.append(T.copy())

        # End-effector position
        p_ee = transforms[6][:3, 3]

        # Build geometric Jacobian (in base frame)
        J = np.zeros((6, 6))
        for i in range(6):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i

        # Use pre-fetched compensated force
        F_ee_compensated = tcp_force
        
        # Calculate joint torques: τ = J^T * F_ee_compensated
        joint_torques = J.T @ F_ee_compensated
        return joint_torques


    def run(self) -> None:
        """Main visualization loop."""
        print("\n" + "="*70)
        print("UR5e Torque Visualizer")
        print(f"Mode: {self.mode.upper()}")
        print("="*70)
        
        # Determine Jacobian for semi-manual mode
        self.J_semi = None
        if self.mode == "semi-manual":
            print("Fetching initial Jacobian from controller (semi-manual)...")
            q_start = self.rtde_r.getActualQ()
            # RTDE getJacobian() returns a flattened 6x6 matrix (36 floats)
            J_flat = self.rtde_c.getJacobian(q_start)
            self.J_semi = np.array(J_flat).reshape(6, 6)
            print("Jacobian cached.")

        if self.enable_freedrive:
            print("FREEDRIVE MODE: Move the robot arm by hand")
        else:
            print("NORMAL MODE: Robot holds position")
            
        print("Press Ctrl+C to stop.")
        print("="*70 + "\n")

        self.running = True
        start_time = time.time()
        loop_count = 0

        try:
            if self.tare:
                print("Taring F/T sensor (RTDE zeroFtSensor)...")
                try:
                    if self.rtde_c.zeroFtSensor():
                        print("F/T sensor tared successfully.")
                        print("Hardware tare applied. Software offsets set to 0 to visualize raw effect.")
                        # Reset software offsets - we rely on the hardware tare now
                        self.torque_offsets = np.zeros(self.num_joints)
                        time.sleep(0.2)
                    else:
                        print("WARNING: zeroFtSensor() failed! (Robot moving?) Falling back to software tare.")
                        
                        print("Taring sensors via software (averaging 1 second)...")
                        offsets = []
                        # Collect 1 second of data
                        num_samples = int(1.0 / self.dt) + 1
                        for _ in range(num_samples):
                            if self.mode == "direct":
                                t = np.array(self.rtde_c.getJointTorques())
                            elif self.mode == "manual":
                                q = self.rtde_r.getActualQ()
                                tcp_force = np.array(self.rtde_r.getActualTCPForce())
                                t = self._calculate_jacobian_torques_manual(np.array(q), tcp_force)
                            elif self.mode == "semi-manual":
                                tcp_force = np.array(self.rtde_r.getActualTCPForce())
                                t = self.J_semi.T @ tcp_force
                            offsets.append(t)
                            time.sleep(self.dt)

                        self.torque_offsets = np.mean(offsets, axis=0)
                        print(f"Torque offsets applied: {[f'{x:+.2f}' for x in self.torque_offsets]}")
                        
                except Exception as e:
                    print(f"Error calling zeroFtSensor: {e}")

            while self.running:
                loop_start = time.time()
                
                tau_plot = np.zeros(6)

                # --- Fetch Data based on Mode ---
                try:
                    # --- Fetch Data based on Mode ---
                    if self.mode == "direct":
                        # Direct mode: Just get the torques from the controller (already compensated)
                        # We fetch from CONTROL interface
                        tau_plot = np.array(self.rtde_c.getJointTorques())
                        
                    elif self.mode == "manual":
                        # Manual mode: Fetch state and calculate J^T * F
                        # We fetch from RECEIVE interface
                        q = self.rtde_r.getActualQ()
                        tcp_force = np.array(self.rtde_r.getActualTCPForce())
                        
                        tau_plot = self._calculate_jacobian_torques_manual(np.array(q), tcp_force)

                    elif self.mode == "semi-manual":
                        # Semi-manual: Use cached Jacobian and current tcp force
                        tcp_force = np.array(self.rtde_r.getActualTCPForce())
                        tau_plot = self.J_semi.T @ tcp_force
                
                except Exception as e:
                    # Handle potential connection drop (e.g. End of File from RTDE)
                    err_str = str(e)
                    if "End of file" in err_str or "Broken pipe" in err_str:
                        print(f"\nConnection lost ({err_str}). Attempting to reconnect RTDE Receive...")
                        try:
                            # Try to reconnect receive interface
                            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                            print("RTDE Receive reconnected.")
                            # Retry this loop iteration
                            continue 
                        except Exception as rec_e:
                            print(f"Reconnection failed: {rec_e}")
                            break
                    else:
                        print(f"Error fetching data: {e}")
                        break
                # Apply tare offset
                tau_plot = tau_plot - self.torque_offsets

                # Apply filter
                tau_plot = self.filter.update(tau_plot)

                # Store history
                current_time = time.time() - start_time
                self.time_history.append(current_time)
                
                for i in range(self.num_joints):
                    self.tau_history[i].append(tau_plot[i])

                # Update plot every N iterations
                loop_count += 1
                if loop_count % 5 == 0:
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
        
        # End freedrive if enabled
        if self.enable_freedrive:
            try:
                self.rtde_c.endFreedriveMode()
                print("Freedrive mode ended")
            except Exception:
                pass
        
        plt.ioff()
        plt.close('all')
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="UR5e External Torque Visualizer")
    parser.add_argument(
        "--ip",
        default="192.168.1.10",
        help="UR5e robot IP address (default: 192.168.1.10)"
    )
    parser.add_argument(
        "--history", "-t",
        type=float,
        default=10.0,
        help="History length in seconds (default: 10)"
    )
    parser.add_argument(
        "--rate", "-r",
        type=float,
        default=125.0,
        help="Sample rate in Hz (default: 125)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["direct", "manual", "semi-manual"],
        default="direct",
        help="Torque visualization mode: 'direct' (Controller), 'manual' (Jacobian Calculation), or 'semi-manual' (Fixed Jacobian)"
    )
    parser.add_argument(
        "--tare",
        action="store_true",
        help="Tare the sensors at startup (average over 1s)"
    )
    parser.add_argument(
        "--freedrive",
        action="store_true",
        help="Enable Freedrive mode (guide by hand)"
    )
    parser.add_argument(
        "--filter",
        choices=["none", "ema", "mean", "butter", "kalman", "one_euro"],
        default="none",
        help="Filter type: [none, ema, mean, butter, kalman, one_euro]"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="EMA: Smoothing factor (0 < alpha <= 1). Default: 0.2"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Mean: Window size. Default: 5"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Butterworth: Cutoff frequency in Hz. Default: 10.0"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.007,
        help="One Euro: Speed coefficient (beta). Higher = less lag. Default: 0.007"
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.0,
        help="One Euro: Min cutoff frequency (Hz). Lower = more damping at low speed. Default: 1.0"
    )
    args = parser.parse_args()

    # Signal handler
    visualizer = None
    def signal_handler(signum, frame):
        if visualizer:
            visualizer.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    visualizer = UR5eTorqueVisualizer(
        robot_ip=args.ip,
        mode=args.mode,
        history_seconds=args.history,
        sample_rate_hz=args.rate,
        enable_freedrive=args.freedrive,
        tare=args.tare,
        filter_type=args.filter,
        filter_alpha=args.alpha,
        filter_window=args.window,
        filter_cutoff=args.cutoff,
        filter_beta=args.beta,
        filter_min_cutoff=args.min_cutoff,
    )
    visualizer.run()


if __name__ == "__main__":
    main()