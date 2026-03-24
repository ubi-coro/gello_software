import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from gello.robots.robot import Robot
from gello.utils.filters import SignalFilter


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self, 
        robot_ip: str = "192.168.1.10", 
        no_gripper: bool = False,
        gripper_feedback_enabled: bool = False,
        # Filter args
        filter_type: str = "none",
        filter_alpha: float = 0.2,
        filter_window: int = 5,
        filter_cutoff: float = 10.0,
        filter_beta: float = 0.007,
        filter_min_cutoff: float = 1.0,
    ):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        self._gripper_state_lock = threading.Lock()
        self._gripper_state_cache = {
               "position": 0.0,
               "is_gripping": False,
               "position_error": 0.0,
               "force_N": 0.0,
               "force_estimate": 0.0,
               "force_normalized": 0.0,
               "commanded_force_N": 20.0,
               "motor_current": 0.0,
        }
        self._last_gripper_cmd = None  # Track last commanded position for debouncing
        self._gripper_cmd_threshold = 2  # Only send if changed by >2 units (out of 255)
        
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
            self.c_inter = self.robot
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")
            # gripper.activate()
            
            # Start background poller for gripper feedback
            self._gripper_running = True
            self._gripper_thread = threading.Thread(target=self._poll_gripper, daemon=True)
            self._gripper_thread.start()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper
        self._gripper_feedback_enabled = gripper_feedback_enabled

        # Initialize Torque Filter and Offset
        self.torque_offsets = np.zeros(6)
        self.filter = SignalFilter(
            filter_type=filter_type,
            sampling_rate=500.0, # UR5e typical control frequency
            alpha=filter_alpha,
            window=filter_window,
            cutoff_hz=filter_cutoff,
            beta=filter_beta,
            min_cutoff=filter_min_cutoff
        )

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = int(joint_state[-1] * 255)
            # Debounce: Only send gripper command if position changed significantly
            # This prevents saturating the gripper socket (which blocks current reading)
            if self._last_gripper_cmd is None or abs(gripper_pos - self._last_gripper_cmd) > self._gripper_cmd_threshold:
                self.gripper.move(gripper_pos, 255, 255)
                self._last_gripper_cmd = gripper_pos
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_joint_torques_controller(self) -> np.ndarray:
        """Get the current joint torques of the robot directly from the controller.

        Returns the torques of all joints, corrected by the torque needed to move
        the robot itself (gravity, friction, etc.).

        Returns:
            np.ndarray: The joint torque vector in Nm [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
        """
        return np.array(self.robot.getJointTorques())

    def get_joint_torques(self) -> np.ndarray:
        """Get the current external joint torques estimate.

        Proxy for get_joint_torques_jacobian_calcualted(), which uses:
        1. J^T * F_ext (Calculated from F/T sensor)
        2. Tare Compensation
        3. Signal Filtering (One-Euro/EMA/etc.)

        This is generally less noisy and more reliable for sensitive force-feedback
        than the raw controller torques on some firmware versions.

        Returns:
            np.ndarray: The joint torque vector in Nm
        """
        return self.get_joint_torques_jacobian_calcualted()
    
    def _poll_gripper(self):
        """Background thread to poll gripper state without blocking main loop."""
        while self._gripper_running:
            try:
                # Read all values (these are blocking TCP calls)
                # Note: Reading current first for lowest latency
                current = float(self.gripper.get_current_motor_current())
                is_gripping = self.gripper.is_gripping()
                pos = self._get_gripper_pos()
                position_error = self.gripper.get_position_error()
                force_est = self.gripper.get_grip_force_estimate()
                force_norm = self.gripper.get_grip_force_normalized()
                commanded_force_N = self.gripper.get_commanded_force_N()
                
                with self._gripper_state_lock:
                    self._gripper_state_cache = {
                        "position": pos,
                        "is_gripping": is_gripping,
                        "position_error": position_error,
                        "force_N": force_est,
                        "force_estimate": force_est,
                        "force_normalized": force_norm,
                        "commanded_force_N": commanded_force_N,
                        "motor_current": current
                    }
                
                # Poll at ~30Hz (fast enough for feedback, leaves bandwidth for commands)
                time.sleep(0.033)
            except Exception as e:
                time.sleep(0.1) # Backoff on error

    def get_ft_wrench(self) -> np.ndarray:
        """Get the raw force and torque measurement from the UR's built-in F/T sensor.

        Not compensated for forces and torques caused by the payload.

        Returns:
            np.ndarray: The raw wrench [fx, fy, fz, tx, ty, tz] in N and Nm.
        """
        return np.array(self.r_inter.getFtRawWrench())
    
    def get_actual_tcp_force(self) -> np.ndarray:
        """
        getActualTCPForce(self: rtde_receive.RTDEReceiveInterface) -> List[float]
        
        Returns:
            Generalized forces in the TCP
        """
        return np.array(self.r_inter.getActualTCPForce())
    
    def tare_jacobian_torques(self, num_samples: int = 100) -> None:
        """Tare the torque sensors using the robot's built-in zeroFtSensor().

        This resets the F/T sensor's zero point in the controller. 
        It is far superior to a software offset because the controller can 
        continue to correctly compensate for payload gravity as the robot 
        orientation changes.

        Args:
            num_samples: Unused, kept for API compatibility.
        """
        print("Taring F/T sensor (zeroFtSensor)...")
        try:
            # Sends command to controller to zero the sensor
            success = self.robot.zeroFtSensor()
            if success:
                print("F/T sensor tared successfully.")
            else:
                print("Warning: zeroFtSensor() returned False (Robot might be moving or not ready).")
        except Exception as e:
            print(f"Error calling zeroFtSensor: {e}")
        
        # Reset software offsets - we rely on the hardware tare now
        self.torque_offsets = np.zeros(6)
        
        # Allow filter to settle on new zero values
        time.sleep(0.2)

    def _calculate_raw_jacobian_torque(self, q: np.ndarray, F_ee_compensated: np.ndarray) -> np.ndarray:
        """Helper to calculate raw J^T * F torque without offsets or filtering."""
        # UR5e DH parameters
        d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        a = [0, -0.425, -0.3922, 0, 0, 0]
        alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]

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

        p_ee = transforms[6][:3, 3]

        J = np.zeros((6, 6))
        for i in range(6):
            z_i = transforms[i][:3, 2]
            p_i = transforms[i][:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i

        return J.T @ F_ee_compensated

    def get_joint_torques_jacobian_calcualted(self) -> np.ndarray:
        """Calculate joint torques from end-effector wrench using Jacobian transpose.

        Uses the UR's built-in F/T sensor via getFtRawWrench() and computes:
        τ = J^T * F_ee
        
        Applies Tare Offset and Signal Filtering.

        Where:
            τ = joint torques (6x1)
            J = geometric Jacobian (6x6)
            F_ee = end-effector wrench [fx, fy, fz, tx, ty, tz] (6x1)

        Returns:
            np.ndarray: Estimated joint torques in Nm (length 6)

        Note:
            The direct method `get_joint_torques()` is preferred as the UR controller
            provides gravity/friction compensated torques directly from motor currents.
        """
        # Get current joint positions for Jacobian calculation
        q = np.array(self.r_inter.getActualQ())
        
        # Get end-effector wrench from UR's built-in F/T sensor
        # F_ee = [fx, fy, fz, tx, ty, tz] - raw, not payload compensated
        F_ee_compensated = self.get_actual_tcp_force()

        # Calculate raw joint torques: τ = J^T * F_ee
        joint_torques = self._calculate_raw_jacobian_torque(q, F_ee_compensated)
        
        # Apply Tare Offset
        joint_torques = joint_torques - self.torque_offsets
        
        # Apply Filter
        joint_torques = self.filter.update(joint_torques)

        return joint_torques
    
    def get_jacobian(
        self, q: Optional[np.ndarray] = None, tcp: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get the Jacobian matrix from the UR controller.

        Args:
            q (np.ndarray, optional): Joint positions. Defaults to current pose.
            tcp (np.ndarray, optional): TCP offset. Defaults to active TCP.

        Returns:
            np.ndarray: The 6x6 Jacobian matrix.
        """
        pos = q.tolist() if q is not None else []
        tcp_offset = tcp.tolist() if tcp is not None else []
        # getJacobian returns a flat list of 36 elements (6x6 matrix row-major)
        jacobian_flat = self.robot.getJacobian(pos, tcp_offset)
        return np.array(jacobian_flat).reshape(6, 6)

    def get_gripper_feedback(self) -> Dict[str, Any]:
        """Get gripper feedback including position, object status, and current.
        
        Returns:
            Dict containing gripper feedback.
        """
        if not self._use_gripper:
            return {
                "position": 0.0,
                "is_gripping": False,
                "position_error": 0.0,
                "force_N": 0.0,
                "force_estimate": 0.0,
                "force_normalized": 0.0,
                "commanded_force_N": 20.0,
                "motor_current": 0.0,
            }
        
        # Return cached state (non-blocking)
        with self._gripper_state_lock:
            return self._gripper_state_cache.copy()

    def get_joint_torques_jacobian(self) -> np.ndarray:
        """Calculate joint torques from end-effector wrench using Jacobian transpose.

        Uses the UR's built-in Jacobian via getJacobian() and F/T sensor via getFtRawWrench():
        τ = J^T * F_ee

        Where:
            τ = joint torques (6x1)
            J = geometric Jacobian from UR controller (6x6)
            F_ee = end-effector wrench [fx, fy, fz, tx, ty, tz] (6x1)

        Returns:
            np.ndarray: Estimated joint torques in Nm (length 6)

        Note:
            The direct method `get_joint_torques()` is preferred as the UR controller
            provides gravity/friction compensated torques directly from motor currents.
        """
        # Get Jacobian directly from UR controller (more accurate than manual DH calculation)
        J = self.get_jacobian()

        # Get end-effector wrench from UR's built-in F/T sensor
        # F_ee = [fx, fy, fz, tx, ty, tz] - raw, not payload compensated
        F_ee = self.get_ft_wrench()

        # Calculate joint torques: τ = J^T * F_ee
        joint_torques = J.T @ F_ee

        return joint_torques

    def command_joint_torques(
        self, torques: np.ndarray, friction_comp: bool = True
    ) -> None:
        """Command joint torques using Direct Joint Torque Control.

        This function must be called continuously at each robot time step (500Hz);
        otherwise, the robot will return to position control mode.
        The function always compensates for gravity internally.

        Args:
            torques (np.ndarray): Target joint torques in Nm (length 6).
            friction_comp (bool): Enable internal friction compensation. Default is True.

        Note:
            - This is an advanced low-level function that bypasses compliance features.
            - You are responsible for keeping the robot within safety limits.
            - When returning to position control mode, use speedj() or stopj().
        """
        assert len(torques) == 6, "Torques must be a vector of length 6"
        self.robot.directTorque(torques.tolist(), friction_comp)

    def set_gripper_feedback_enabled(self, enabled: bool) -> None:
        """Enable or disable gripper feedback queries.
        
        Disabling gripper feedback avoids the ~10ms socket delay per control cycle
        when gripper force-feedback is not needed.
        
        Args:
            enabled: Whether to query gripper feedback in get_observations()
        """
        self._gripper_feedback_enabled = enabled

    def get_observations(self) -> Dict[str, Any]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])
        joint_torques = self.get_joint_torques()
        joint_velocities = np.array(self.r_inter.getActualQd())

        if self._use_gripper:
            joint_velocities = np.append(joint_velocities, 0.0)

        # Only query gripper feedback if enabled (avoids ~10ms socket delay per call)
        if self._gripper_feedback_enabled:
            gripper_feedback = self.get_gripper_feedback()
        else:
            gripper_feedback = {
                "position": float(gripper_pos[0]),
                "is_gripping": False,
                "position_error": 0.0,
                "force_N": 0.0,
                "force_normalized": 0.0,
                "commanded_force_N": 20.0,
            }
        return {
            "joint_positions": joints,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
            "joint_torques": joint_torques,
            "gripper_feedback": gripper_feedback,
        }
    
    def command_joint_state_impedance(
        self,
        target_joints: np.ndarray,
        target_velocities: Optional[np.ndarray] = None,
        kp: float = 50.0,
        kd: float = 5.0,
        tau_ff: Optional[np.ndarray] = None,
    ) -> bool:
        """Impedance control using directTorque for compliant trajectory following.
        
        Computes: τ = Kp*(q_target - q) + Kd*(q̇_target - q̇) + τ_ff
        
        The robot behaves like a spring-damper system to the target position.
        On contact with obstacles, the robot yields (compliant behavior).
        
        MUST be called at 500 Hz! If interrupted, robot returns to position control.
        
        Args:
            target_joints: Desired joint positions (rad) - from leader
            target_velocities: Desired velocities (rad/s) - None means pure damping
            kp: Position stiffness (Nm/rad) - higher = stiffer/faster
            kd: Velocity damping (Nm*s/rad) - higher = more damping
            tau_ff: Feedforward torques (Nm) - e.g., for known payloads
            
        Returns:
            bool: True if command was successful
        """
        # Current state
        q = np.array(self.r_inter.getActualQ())
        qd = np.array(self.r_inter.getActualQd())
        
        # Target velocity (default: 0 = pure damping, most stable)
        if target_velocities is None:
            qd_target = np.zeros(6)
        else:
            qd_target = np.asarray(target_velocities[:6])
        
        # Impedance control law
        q_error = np.asarray(target_joints[:6]) - q
        qd_error = qd_target - qd
        
        tau_cmd = kp * q_error + kd * qd_error
        
        # Add feedforward if provided
        if tau_ff is not None:
            tau_cmd += np.asarray(tau_ff[:6])
        
        # Safety: per-joint torque limits (conservative for UR5e)
        # These are well below the motor limits but safe for testing
        tau_max = np.array([100.0, 100.0, 60.0, 25.0, 25.0, 25.0])  # Nm
        tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
        
        # Send to robot (friction_comp=True uses UR's internal friction model)
        return self.robot.directTorque(tau_cmd.tolist(), True)


def main():
    robot_ip = "192.168.1.11"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
