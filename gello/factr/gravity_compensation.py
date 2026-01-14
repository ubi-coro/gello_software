"""
Standalone FACTR Gravity Compensation Script (Non-ROS)

This script provides the similar gravity compensation functionality as the ROS-based
FACTR teleop system, but without ROS dependencies.
Usage:
    python3 gello/factr/gravity_compensation.py --config configs/yam_gello_factr_hw.yaml
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import yaml

import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread
from dataclasses import dataclass, field
from typing import Dict, List, Deque

from gello.dynamixel.driver import DynamixelDriver

import threading
from importlib import import_module
from typing import Any, Dict, cast


def find_ttyusb(port_name: str) -> str:
    """Locate the underlying ttyUSB device."""
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(
            f"Unable to resolve the symbolic link for '{port_name}'. {e}"
        ) from e


def _instantiate_from_dict(cfg: Dict[str, Any]) -> Any:
    """Lightweight instantiation from a dict with a _target_ path.

    Keeps this script self-contained without importing broader launch utilities.
    """
    assert isinstance(cfg, dict) and "_target_" in cfg, "Invalid instantiation config"
    module_path, class_name = cfg["_target_"].rsplit(".", 1)
    cls = getattr(import_module(module_path), class_name)
    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}

    # Recurse into nested dicts/lists
    def _recurse(v):
        if isinstance(v, dict) and "_target_" in v:
            return _instantiate_from_dict(v)
        if isinstance(v, dict):
            return {kk: _recurse(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_recurse(x) for x in v]
        return v

    return cls(**{k: _recurse(v) for k, v in kwargs.items()})


@dataclass
class TorqueComponentLogger:
    """Logger for torque components with thread-safe deques."""
    history_len: int = 1000

    # Component histories
    time_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    tau_gravity: List[Deque[float]] = field(default_factory=list)
    tau_friction: List[Deque[float]] = field(default_factory=list)
    tau_damping: List[Deque[float]] = field(default_factory=list)
    tau_null: List[Deque[float]] = field(default_factory=list)
    tau_limit: List[Deque[float]] = field(default_factory=list)
    tau_feedback: List[Deque[float]] = field(default_factory=list)
    tau_total: List[Deque[float]] = field(default_factory=list)
    positions: List[Deque[float]] = field(default_factory=list)
    velocities: List[Deque[float]] = field(default_factory=list)

    def __post_init__(self):
        self.time_history = deque(maxlen=self.history_len)

    def initialize(self, num_joints: int):
        """Initialize deques for each joint."""
        for _ in range(num_joints):
            self.tau_gravity.append(deque(maxlen=self.history_len))
            self.tau_friction.append(deque(maxlen=self.history_len))
            self.tau_damping.append(deque(maxlen=self.history_len))
            self.tau_null.append(deque(maxlen=self.history_len))
            self.tau_limit.append(deque(maxlen=self.history_len))
            self.tau_feedback.append(deque(maxlen=self.history_len))
            self.tau_total.append(deque(maxlen=self.history_len))
            self.positions.append(deque(maxlen=self.history_len))
            self.velocities.append(deque(maxlen=self.history_len))

    def log(
            self,
            timestamp: float,
            positions: np.ndarray,
            velocities: np.ndarray,
            tau_gravity: np.ndarray,
            tau_friction: np.ndarray,
            tau_damping: np.ndarray,
            tau_null: np.ndarray,
            tau_limit: np.ndarray,
            tau_feedback: np.ndarray,
            tau_total: np.ndarray,
    ):
        """Log one timestep of torque components."""
        self.time_history.append(timestamp)
        for i in range(len(positions)):
            self.positions[i].append(positions[i])
            self.velocities[i].append(velocities[i])
            self.tau_gravity[i].append(tau_gravity[i])
            self.tau_friction[i].append(tau_friction[i])
            self.tau_damping[i].append(tau_damping[i])
            self.tau_null[i].append(tau_null[i])
            self.tau_limit[i].append(tau_limit[i])
            self.tau_feedback[i].append(tau_feedback[i])
            self.tau_total[i].append(tau_total[i])


class FACTRGravityCompensation:
    """
    Standalone FACTR gravity compensation system without ROS dependencies.

    This class implements the core functionality of FACTR teleop gravity compensation,
    including:
    - Gravity compensation using inverse dynamics
    - Null-space regulation
    - Joint limit barriers
    - Static friction compensation
    """

    # Calibration constants - search range and resolution
    # Range: -10π to 10π (±5 full rotations should be enough)
    # Resolution: ~5 degrees (π/36) for much better accuracy than original 90°
    CALIBRATION_RANGE_MULTIPLIER = 10  # Range: -10π to 10π
    CALIBRATION_STEP_COUNT = 721  # 10 * 2 * 36 + 1 = 721 steps (every 5°)

    def __init__(self, config_path: str, enable_visualization: bool = False):
        self.running = False
        self.config_path = config_path
        self.driver: Optional[DynamixelDriver] = None  # Initialize early for cleanup

        # Teleop-related fields
        self.teleop_enabled: bool = False
        self.teleop_env = None
        self.teleop_client = None
        self.teleop_rate_hz: float = 30.0
        self.teleop_thread: Optional[threading.Thread] = None
        self.teleop_robot_server = None
        self.teleop_threads: list[threading.Thread] = []
        self.teleop_prepared: bool = False
        # Mapping from leader (arm joints) -> follower (first K joints)
        self.map_index: Optional[np.ndarray] = None
        self.map_signs: Optional[np.ndarray] = None
        self.map_offsets: Optional[np.ndarray] = None
        # Optional gripper teleop mapping using explicit open/close angles (degrees)
        self.gripper_open_rad: Optional[float] = None
        self.gripper_close_rad: Optional[float] = None
        # Last raw leader gripper reading in radians (before offsets/signs)
        self.leader_gripper_raw_rad: float = 0.0
        # Teleop smoothing (match baseline non-FACTR behavior)
        self.teleop_smoothing_alpha: float = 0.99
        self._teleop_last_action: Optional[np.ndarray] = None

        # Visualization setup
        self.enable_visualization = enable_visualization
        self._viz_logger: Optional[TorqueComponentLogger] = None
        self._viz_thread: Optional[Thread] = None
        self._viz_start_time: float = 0.0

        try:
            self._load_config()
            self._setup_parameters()
            
            # Setup visualization after parameters are loaded (needs self.dt)
            if self.enable_visualization:
                self._setup_visualization()
            
            self._prepare_dynamixel()
            self._prepare_inverse_dynamics()
            self._calibrate_system()
            # Optional teleop setup
            self._maybe_setup_teleop()
        except Exception as e:
            # Cleanup on initialization failure
            self.shutdown()
            raise RuntimeError(f"Failed to initialize FACTR system: {e}") from e

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        print(f"Loaded config: {self.config['name']}")
        try:
            print(f"Config file: {Path(self.config_path).expanduser().resolve()}")
        except Exception:
            print(f"Config file: {self.config_path}")

    def _setup_parameters(self) -> None:
        """Initialize parameters from config."""
        self.name = self.config["name"]
        self.dt = 1 / self.config["controller"]["frequency"]

        # Leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_max"])
            - self.safety_margin
        )
        self.arm_joint_limits_min = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_min"])
            + self.safety_margin
        )
        self.calibration_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["calibration_joint_pos"]
        )
        self.initial_match_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"]
        )
        init_cfg = self.config["arm_teleop"].get("initialization", {})
        self.enforce_initial_match = bool(init_cfg.get("enforce_initial_match", False))
        self.calibration_sanity_threshold = float(
            init_cfg.get("calibration_sanity_threshold", 0.35)
        )

        # Gripper parameters
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # Control parameters
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_modifier = self.config["controller"]["gravity_comp"]["gain"]
        # Optional viscous damping term in joint space (helps holding stability)
        self.gravity_comp_velocity_damping = float(
            self.config["controller"]["gravity_comp"].get("velocity_damping", 0.0)
        )
        # Optional per-joint scaling for gravity compensation (lets you boost weaker joints)
        gcfg = self.config.get("controller", {}).get("gravity_comp", {})
        default_gpj = [1.0] * self.num_arm_joints
        gpj = gcfg.get("gain_per_joint", default_gpj)
        if not isinstance(gpj, list) or len(gpj) < self.num_arm_joints:
            gpj = default_gpj
        self.gravity_comp_gain_per_joint = np.asarray(gpj[: self.num_arm_joints], dtype=float)
        self.tau_g = np.zeros(self.num_arm_joints)
        # Cache last gripper velocity so we can build URDF-sized vectors for Pinocchio
        self._last_gripper_vel: float = 0.0

        # Friction compensation
        self.stiction_comp_enable_speed = self.config["controller"][
            "static_friction_comp"
        ]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"][
            "gain"
        ]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)
        
        # Per-joint friction feedforward (estimated from physical behavior recordings)
        # These are Coulomb friction torques added in direction of motion
        friction_cfg = self.config["controller"].get("static_friction_comp", {})
        default_ff = [0.0] * self.num_arm_joints
        self.friction_feedforward = np.array(
            friction_cfg.get("friction_feedforward", default_ff)
        )
        # Viscous friction coefficient (Nm*s/rad)
        self.viscous_friction = np.array(
            friction_cfg.get("viscous_friction", default_ff)
        )
        # Velocity deadband - don't apply friction comp below this
        default_deadband = [0.05] * self.num_arm_joints
        self.friction_velocity_deadband = np.array(
            friction_cfg.get("velocity_deadband", default_deadband)
        )

        # Joint limit barrier
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        # Null space regulation
        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"][
                "null_space_joint_target"
            ]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]

        # Torque feedback (force-feedback from follower robot) - FACTR scaled style
        torque_feedback_cfg = self.config["controller"].get("torque_feedback", {})
        self.enable_torque_feedback = torque_feedback_cfg.get("enable", False)
        self.torque_feedback_gain = torque_feedback_cfg.get("gain", 1.0)
        self.torque_feedback_motor_scalar = torque_feedback_cfg.get("motor_scalar", 1.0)
        self.torque_feedback_damping = torque_feedback_cfg.get("damping", 0.0)
        self._follower_torques = np.zeros(self.num_arm_joints)  # Cache for follower torques

        # Force-Position feedback - alternative to FACTR scaled feedback
        # τ = Kp*(q_follower - q_leader) + Kd*(dq_follower - dq_leader)
        force_pos_cfg = self.config["controller"].get("force_position_feedback", {})
        self.enable_force_position_feedback = force_pos_cfg.get("enable", False)
        self.force_position_kp = force_pos_cfg.get("kp", 2.0)
        self.force_position_kd = force_pos_cfg.get("kd", 0.1)
        self.force_position_max_torque = force_pos_cfg.get("max_torque", 1.5)
        self._follower_arm_pos = np.zeros(self.num_arm_joints)  # Cache for follower position
        self._follower_arm_vel = np.zeros(self.num_arm_joints)  # Cache for follower velocity

        # Gripper feedback (position-position or position-force feedback)
        gripper_feedback_cfg = self.config["controller"].get("gripper_feedback", {})
        self.enable_gripper_feedback = gripper_feedback_cfg.get("enable", False)
        self.gripper_feedback_gain = gripper_feedback_cfg.get("gain", 1.0)
        self.gripper_feedback_damping = gripper_feedback_cfg.get("damping", 0.1)
        self._follower_gripper_feedback: Dict[str, Any] = {}  # Cache for follower gripper state

        # Validate: only one feedback mode should be active
        if self.enable_torque_feedback and self.enable_force_position_feedback:
            print("WARNING: Both torque_feedback and force_position_feedback are enabled!")
            print("         Only one should be active. Disabling torque_feedback.")
            self.enable_torque_feedback = False

        print(f"Control frequency: {1 / self.dt:.1f} Hz")
        print(
            f"Gravity compensation: {'enabled' if self.enable_gravity_comp else 'disabled'}"
        )
        if self.enable_gravity_comp:
            print(
                f"Gravity comp damping: {self.gravity_comp_velocity_damping:.4f} (Nm*s/rad)"
            )
            if np.any(np.abs(self.gravity_comp_gain_per_joint - 1.0) > 1e-9):
                print(
                    "Gravity comp per-joint gain: "
                    + str([float(f"{x:.3f}") for x in self.gravity_comp_gain_per_joint])
                )
        print(
            f"Torque feedback (FACTR): {'enabled' if self.enable_torque_feedback else 'disabled'}"
        )
        print(
            f"Force-Position feedback: {'enabled' if self.enable_force_position_feedback else 'disabled'}"
        )
        print(
            f"Gripper feedback: {'enabled' if self.enable_gripper_feedback else 'disabled'}"
        )

    def _prepare_dynamixel(self) -> None:
        """Initialize Dynamixel servo driver."""
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(
            self.config["dynamixel"]["joint_signs"], dtype=float
        )
        # Torque signs: separate from position signs for motors where torque direction
        # differs from position direction. Defaults to joint_signs if not specified.
        self.torque_signs = np.array(
            self.config["dynamixel"].get("torque_signs", self.joint_signs.tolist()),
            dtype=float
        )

        # Print sign configuration clearly (helps diagnose torque direction issues)
        print(f"Joint signs:  {[f'{x:+.0f}' for x in self.joint_signs]}")
        if "torque_signs" in self.config.get("dynamixel", {}):
            print(f"Torque signs: {[f'{x:+.0f}' for x in self.torque_signs]}")
        else:
            print("Torque signs: (not set) using joint_signs")
        
        port_config = self.config["dynamixel"]["dynamixel_port"]
        if port_config.startswith("/"):
            self.dynamixel_port = port_config
        else:
            self.dynamixel_port = "/dev/serial/by-id/" + port_config

        # Check latency timer (only meaningful for ttyUSB devices)
        try:
            resolved_path = os.path.realpath(self.dynamixel_port)
            dev_name = os.path.basename(resolved_path)

            ttyUSBx: Optional[str] = None
            if dev_name.startswith("ttyUSB"):
                ttyUSBx = dev_name
            elif self.dynamixel_port.startswith("/dev/serial/by-id/"):
                # Fall back to explicit by-id resolution
                ttyUSBx = find_ttyusb(os.path.basename(self.dynamixel_port))

            if ttyUSBx is None:
                print(
                    f"Latency timer check skipped (device '{dev_name}' is not ttyUSB*)"
                )
            else:
                latency_path = f"/sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
                result = subprocess.run(
                    ["cat", latency_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ttyUSB_latency_timer = int(result.stdout)
                if ttyUSB_latency_timer != 1:
                    print(
                        f"Warning: Latency timer of {ttyUSBx} is {ttyUSB_latency_timer}, should be 1 for optimal performance."
                    )
                    print(
                        f"Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
                    )
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
            print(f"Could not check latency timer (file access issue): {e}")
        except (ValueError, IndexError) as e:
            print(f"Could not parse latency timer value: {e}")
        except Exception as e:
            print(f"Could not check latency timer: {e}")

        # Initialize driver
        joint_ids = (np.arange(self.num_motors) + 1).tolist()
        baudrate = self.config["dynamixel"].get("baudrate", 57600)
        try:
            self.driver = DynamixelDriver(
                joint_ids, self.servo_types, self.dynamixel_port, baudrate=baudrate
            )
            print(f"Connected to Dynamixel servos on {self.dynamixel_port} with baudrate {baudrate}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Dynamixel servos: {e}") from e

        # Configure servos
        self.driver.set_torque_mode(False)
        if self.enable_gravity_comp:
            # Use current control with torque enabled when GC is active
            self.driver.set_operating_mode(0)  # Current control mode
            self.driver.set_torque_mode(True)
        else:
            # When GC is disabled, keep motors free/backdrivable similar to baseline
            # Try to switch to position mode (not strictly necessary) and keep torque disabled
            try:
                self.driver.set_operating_mode(3)  # Position control mode
            except Exception:
                pass
            self.driver.set_torque_mode(False)

    def _prepare_inverse_dynamics(self) -> None:
        """Initialize Pinocchio model for inverse dynamics."""
        # Construct URDF path - make GELLO completely self-contained
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]

        # Try relative to config file first
        config_dir = Path(self.config_path).parent
        urdf_path = config_dir / urdf_filename

        # If not found, try relative to GELLO factr directory (self-contained)
        if not urdf_path.exists():
            gello_factr_urdf_path = (
                Path(__file__).parent / "urdf" / Path(urdf_filename).name
            )
            if gello_factr_urdf_path.exists():
                urdf_path = gello_factr_urdf_path
            else:
                # Try the full GELLO path
                gello_full_urdf_path = (
                    Path(__file__).parent.parent.parent / urdf_filename
                )
                if gello_full_urdf_path.exists():
                    urdf_path = gello_full_urdf_path
                else:
                    raise FileNotFoundError(
                        f"URDF file {urdf_filename} not found in GELLO paths"
                    )

        print(f"Loading URDF: {urdf_path}")
        urdf_model_dir = str(urdf_path.parent)
        self.pin_model, _, _ = pin.buildModelsFromUrdf(
            filename=str(urdf_path), package_dirs=urdf_model_dir
        )
        self.pin_data = self.pin_model.createData()
        # Cache sizes for quick sanity checks and vector padding
        self._pin_nq = int(getattr(self.pin_model, "nq", 0))
        self._pin_nv = int(getattr(self.pin_model, "nv", 0))
        if self._pin_nq <= 0 or self._pin_nv <= 0:
            print("Warning: Pinocchio model has invalid nq/nv; inverse dynamics may fail.")
        if self._pin_nq != self._pin_nv:
            print(
                f"Warning: Pinocchio model nq ({self._pin_nq}) != nv ({self._pin_nv}). "
                "This code assumes fixed-base manipulators where nq==nv."
            )

    def _calibrate_system(self) -> None:
        """Calibrate Dynamixel offsets and match initial position."""
        print("Calibrating Dynamixel offsets...")
        self._get_dynamixel_offsets()

        # Sanity check: after applying offsets, the current pose should be close to calibration_joint_pos.
        try:
            curr_pos, _, _, _ = self.get_leader_joint_states()
            target = self.calibration_joint_pos[0 : self.num_arm_joints]
            pose_err = np.abs(curr_pos - target)
            max_err = float(np.max(pose_err)) if pose_err.size else 0.0
            mean_err = float(np.mean(pose_err)) if pose_err.size else 0.0
            print(
                f"Calibration sanity: max|q-target|={max_err:.3f} rad, mean={mean_err:.3f} rad"
            )
            if max_err > self.calibration_sanity_threshold:
                print(
                    "WARNING: Leader pose is far from calibration_joint_pos after offset calibration. "
                    "This usually means you started in a different pose than arm_teleop.initialization.calibration_joint_pos, "
                    "or joint_signs are wrong. Gravity compensation will likely be poor in some configurations."
                )
        except Exception as e:
            print(f"Warning: calibration sanity check failed: {e}")

        if self.enforce_initial_match:
            print("Waiting for leader to match initial_match_joint_pos...")
            self._match_start_pos()
            print("Initial match reached.")
        else:
            print("Skipping initial position match...")

        print("System calibrated and ready!")

    def _maybe_setup_teleop(self) -> None:
        """Optionally set up a follower robot and teleop loop if enabled by config."""
        teleop_cfg = self.config.get("teleop", {})
        enabled = bool(teleop_cfg.get("enable", False))
        if not enabled:
            return

        # Lazily import here to avoid adding dependencies when teleop is disabled
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot
        from gello.env import RobotEnv

        self.teleop_enabled = True
        self.teleop_rate_hz = float(teleop_cfg.get("hz", 30))
        server_host = teleop_cfg.get("robot", {}).get("host", "127.0.0.1")
        base_port = int(teleop_cfg.get("robot", {}).get("port", 6001))
        server_timeout_s = float(teleop_cfg.get("wait_for_server_timeout_s", 5.0))

        robot_cfg = teleop_cfg.get("robot")
        if not isinstance(robot_cfg, dict) or "_target_" not in robot_cfg:
            raise ValueError("teleop.robot must be a dict containing a _target_ field")

        # Optional gripper config: [id, open_deg, close_deg]
        gc = teleop_cfg.get("gripper_config")
        if isinstance(gc, list) and len(gc) == 3:
            try:
                _, open_deg, close_deg = gc
                self.gripper_open_rad = float(open_deg) * np.pi / 180.0
                self.gripper_close_rad = float(close_deg) * np.pi / 180.0
                print(
                    f"Teleop gripper_config set (rad): open={self.gripper_open_rad:.3f}, close={self.gripper_close_rad:.3f}"
                )
            except Exception as e:
                print(f"Warning: invalid teleop.gripper_config, ignoring: {e}")

        # Instantiate follower robot from config
        follower_robot = _instantiate_from_dict(robot_cfg)
        
        # Auto-Tare follower robot if supported (CRITICAL for safe operation)
        if hasattr(follower_robot, "tare_jacobian_torques"):
            try:
                print("Auto-Taring follower robot torques (100 samples)...")
                # Ensure we are in a safe state/mode if needed, though tare usually just reads
                follower_robot.tare_jacobian_torques()
            except Exception as e:
                print(f"Warning: Failed to tare follower robot: {e}")

        self.teleop_robot_server = follower_robot

        # Determine server port to use
        server_port = base_port
        if hasattr(follower_robot, "serve"):
            # Sim server exposes port in its own config; prefer that
            try:
                server_port = int(robot_cfg.get("port", base_port))
            except Exception:
                server_port = base_port

        # Start server in background if needed
        if hasattr(follower_robot, "serve"):
            server_thread = threading.Thread(target=follower_robot.serve, daemon=True)
            server_thread.start()
            self.teleop_threads.append(server_thread)
        else:
            # Hardware robot; wrap with ZMQServerRobot and auto-select port if needed
            from zmq.error import ZMQError  # type: ignore

            selected_port = None
            last_error: Optional[Exception] = None
            for port_delta in range(0, 16):
                try:
                    candidate_port = base_port + port_delta
                    server = ZMQServerRobot(
                        follower_robot, port=candidate_port, host=server_host
                    )
                    server_thread = threading.Thread(target=server.serve, daemon=True)
                    server_thread.start()
                    self.teleop_robot_server = server
                    self.teleop_threads.append(server_thread)
                    selected_port = candidate_port
                    break
                except (ZMQError, Exception) as e:  # bind may fail if address in use
                    last_error = e
                    msg = str(e)
                    if "Address already in use" in msg or "address in use" in msg:
                        continue
                    raise
            if selected_port is None:
                raise RuntimeError(
                    f"Failed to create ZMQ server for hardware follower: {last_error}"
                )
            server_port = selected_port

        # Wait for server to become ready
        start = time.time()
        while True:
            try:
                client = ZMQClientRobot(port=server_port, host=server_host)
                # Probe an RPC to ensure server responsiveness
                _ = client.num_dofs()
                self.teleop_client = client
                break
            except Exception:
                if time.time() - start > server_timeout_s:
                    raise RuntimeError(
                        f"Follower server failed to start on {server_host}:{server_port} within {server_timeout_s} seconds"
                    )
                time.sleep(0.1)

        # Create env for follower
        self.teleop_env = RobotEnv(
            self.teleop_client, control_rate_hz=self.teleop_rate_hz
        )

        # Determine follower DOFs and build mapping defaults
        try:
            follower_dofs = int(self.teleop_client.num_dofs())
        except Exception:
            follower_dofs = self.num_arm_joints
        # If follower appears to include a gripper as last joint
        follower_has_gripper = follower_dofs == (self.num_arm_joints + 1)
        map_dims = min(
            self.num_arm_joints, follower_dofs - (1 if follower_has_gripper else 0)
        )
        default_index = np.arange(map_dims, dtype=int)
        default_signs = np.ones(map_dims, dtype=float)
        default_offsets = np.zeros(map_dims, dtype=float)

        # Load mapping config
        mapping_cfg = (
            teleop_cfg.get("mapping", {})
            if isinstance(teleop_cfg.get("mapping", {}), dict)
            else {}
        )
        index_map = mapping_cfg.get("index_map")
        signs = mapping_cfg.get("signs")
        offsets = mapping_cfg.get("offsets")
        auto_align = bool(mapping_cfg.get("auto_align", True))

        if index_map is None:
            self.map_index = default_index
        else:
            self.map_index = np.array(index_map, dtype=int)
        if signs is None:
            self.map_signs = default_signs
        else:
            self.map_signs = np.array(signs, dtype=float)
        if offsets is None:
            self.map_offsets = default_offsets
        else:
            self.map_offsets = np.array(offsets, dtype=float)

        # Use local, non-optional views for validation
        map_index_local = cast(np.ndarray, self.map_index)
        map_signs_local = cast(np.ndarray, self.map_signs)
        map_offsets_local = cast(np.ndarray, self.map_offsets)

        # Validate lengths
        if not (
            len(map_index_local)
            == len(map_signs_local)
            == len(map_offsets_local)
            == map_dims
        ):
            print(
                "Warning: teleop.mapping lengths mismatch or not equal to map_dims; using defaults."
            )
            self.map_index = default_index
            self.map_signs = default_signs
            self.map_offsets = default_offsets
            map_index_local = default_index
            map_signs_local = default_signs
            map_offsets_local = default_offsets

        # Optionally move follower to a start position
        start_joints = teleop_cfg.get("start_joints")
        if start_joints is not None:
            try:
                self._move_follower_to_start(np.array(start_joints, dtype=float))
            except Exception as e:
                print(f"Warning: failed to move follower to start position: {e}")

        # Optional auto-alignment: compute offsets so current follower == mapped leader
        if auto_align:
            try:
                obs = self.teleop_env.get_obs()
                follower_curr = obs["joint_positions"]
                leader_arm_pos, _, _, _ = self.get_leader_joint_states()
                leader_mapped = map_signs_local * leader_arm_pos[map_index_local]
                follower_slice = follower_curr[: int(len(map_index_local))]
                self.map_offsets = follower_slice - leader_mapped
                map_offsets_local = cast(np.ndarray, self.map_offsets)
                print(
                    f"Teleop auto-aligned offsets set to: {[float(x) for x in map_offsets_local]}"
                )
            except Exception as e:
                print(f"Warning: auto_align failed: {e}")

        # Mark prepared; DO NOT start thread yet (start in run() after running=True)
        self.teleop_prepared = True
        print(
            f"Teleop prepared: follower ready at {server_host}:{server_port}, will mirror at {self.teleop_rate_hz:.1f} Hz"
        )
        # Print mapping and initial states for quick tuning
        try:
            map_index_local = cast(np.ndarray, self.map_index)
            map_signs_local = cast(np.ndarray, self.map_signs)
            map_offsets_local = cast(np.ndarray, self.map_offsets)
            leader_arm_pos, _, _, _ = self.get_leader_joint_states()
            leader_mapped_dbg = (
                map_signs_local * leader_arm_pos[map_index_local] + map_offsets_local
            )
            follower_dbg = self.teleop_env.get_obs()["joint_positions"][
                : int(len(map_index_local))
            ]

            def _fmt(arr):
                return [float(f"{x:.3f}") for x in np.array(arr).tolist()]

            print("Teleop mapping:")
            print(f"  index_map: {map_index_local.tolist()}")
            print(f"  signs: {_fmt(map_signs_local)}")
            print(f"  offsets: {_fmt(map_offsets_local)}")
            print(f"  leader(mapped): {_fmt(leader_mapped_dbg)}")
            print(f"  follower(curr): {_fmt(follower_dbg)}")
        except Exception as e:
            print(f"Warning: teleop debug print failed: {e}")

    def _move_follower_to_start(self, target_joints: np.ndarray) -> None:
        assert self.teleop_env is not None
        obs = self.teleop_env.get_obs()
        curr = obs["joint_positions"]
        if curr.shape != target_joints.shape:
            print("Warning: follower start joints shape mismatch; skipping")
            return
        steps = int(min(max(np.abs(curr - target_joints).max() / 0.01, 1), 100))
        for jnt in np.linspace(curr, target_joints, steps):
            self.teleop_env.step(jnt)
            time.sleep(0.001)

    def _build_follower_action(
        self,
        self_arm_pos: npt.NDArray[np.float64],
        self_gripper_pos: float,
    ) -> np.ndarray:
        """Build follower joint command from leader arm and gripper positions.

        Applies configured mapping: index, signs, offsets. Handles extra gripper DOF.
        """
        assert self.teleop_client is not None
        try:
            follower_dofs = self.teleop_client.num_dofs()
        except Exception:
            follower_dofs = len(self_arm_pos)

        # Prepare mapped arm command
        if self.map_index is None or self.map_signs is None or self.map_offsets is None:
            map_index = np.arange(min(len(self_arm_pos), follower_dofs), dtype=int)
            map_signs = np.ones_like(map_index, dtype=float)
            map_offsets = np.zeros_like(map_index, dtype=float)
        else:
            map_index = self.map_index
            map_signs = self.map_signs
            map_offsets = self.map_offsets

        arm_cmd = map_signs * self_arm_pos[map_index] + map_offsets

        # Compose final command with optional gripper channel
        if follower_dofs == len(arm_cmd) + 1:
            # Determine normalized gripper using explicit config if provided
            if self.gripper_open_rad is not None and self.gripper_close_rad is not None:
                denom = self.gripper_close_rad - self.gripper_open_rad
                if abs(denom) < 1e-6:
                    gripper_norm = 0.0
                else:
                    gripper_norm = (
                        self.leader_gripper_raw_rad - self.gripper_open_rad
                    ) / denom
            else:
                # Fallback to actuation_range
                gripper_norm = self_gripper_pos / max(self.gripper_limit_max, 1e-6)
            gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
            return np.concatenate([arm_cmd, np.array([gripper_norm], dtype=float)])

        if follower_dofs == len(arm_cmd):
            # Special case: If we have 7 joints (incl gripper) and follower has 7,
            # but we have explicit gripper config, override the last joint with normalized value.
            if self.gripper_open_rad is not None and self.gripper_close_rad is not None:
                denom = self.gripper_close_rad - self.gripper_open_rad
                if abs(denom) < 1e-6:
                    gripper_norm = 0.0
                else:
                    gripper_norm = (
                        self.leader_gripper_raw_rad - self.gripper_open_rad
                    ) / denom
                gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
                # Override the last element (gripper joint)
                arm_cmd[-1] = gripper_norm
            
            return arm_cmd
        if follower_dofs < len(arm_cmd):
            return arm_cmd[:follower_dofs]
        # Pad with zeros for any extra joints (unlikely)
        padded = np.zeros((follower_dofs,), dtype=float)
        padded[: len(arm_cmd)] = arm_cmd
        return padded

    def _teleop_loop(self) -> None:
        assert self.teleop_env is not None
        rate_dt = 1.0 / max(self.teleop_rate_hz, 1e-3)
        print("Starting teleop loop (follower control)")
        while self.running:
            t0 = time.time()
            try:
                # Use the same leader state access used by GC, which already applies offsets/signs
                (
                    leader_arm_pos,
                    leader_arm_vel,
                    leader_gripper_pos,
                    leader_gripper_vel,
                ) = self.get_leader_joint_states()
                action = self._build_follower_action(leader_arm_pos, leader_gripper_pos)
                # Apply exponential smoothing to follower command to mirror baseline
                if self._teleop_last_action is None or len(
                    self._teleop_last_action
                ) != len(action):
                    self._teleop_last_action = action
                else:
                    action = (
                        self._teleop_last_action * (1.0 - self.teleop_smoothing_alpha)
                        + action * self.teleop_smoothing_alpha
                    )
                    self._teleop_last_action = action
                self.teleop_env.step(action)
            except Exception as e:
                print(f"Teleop loop warning: {e}")
            # Timing
            elapsed = time.time() - t0
            sleep_t = rate_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _get_dynamixel_offsets(self, verbose: bool = True) -> None:
        """Calibrate Dynamixel servos to match expected joint positions.
        
        This finds the offset for each joint such that:
            joint_sign[i] * (raw_position[i] - offset[i]) ≈ calibration_joint_pos[i]
        
        The offset accounts for the arbitrary zero position of multi-turn Dynamixel servos.
        """
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        
        # Warm up - ensure stable readings
        print("\n" + "="*60)
        print("DYNAMIXEL OFFSET CALIBRATION")
        print("="*60)
        print(f"Expected calibration pose: {[f'{x:.3f}' for x in self.calibration_joint_pos]}")
        print(f"Joint signs: {[f'{x:+.0f}' for x in self.joint_signs]}")
        print(f"Search resolution: ~{np.rad2deg(2 * self.CALIBRATION_RANGE_MULTIPLIER * np.pi / self.CALIBRATION_STEP_COUNT):.1f}°")
        
        for _ in range(10):
            self.driver.get_positions_and_velocities()

        def get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # Get current raw positions
        curr_joints, _ = self.driver.get_positions_and_velocities()
        if len(curr_joints) < self.num_arm_joints:
            raise RuntimeError(
                f"Dynamixel returned {len(curr_joints)} joints, but num_arm_joints={self.num_arm_joints}. "
                "Check your config (arm_teleop.num_arm_joints vs dynamixel.servo_types length)."
            )
        
        print(f"\nRaw Dynamixel positions (rad): {[f'{x:.4f}' for x in curr_joints]}")
        print(f"Raw Dynamixel positions (deg): {[f'{np.rad2deg(x):.1f}' for x in curr_joints]}")

        # Calibrate arm joints with fine-grained search
        self.joint_offsets = []
        print("\nCalibrating each joint:")
        print("-" * 60)
        
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            target = self.calibration_joint_pos[i]
            sign = self.joint_signs[i]
            raw = curr_joints[i]
            
            # Fine-grained search (now ~5° resolution instead of 90°)
            for offset in np.linspace(
                -self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
                self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
                self.CALIBRATION_STEP_COUNT,
            ):
                error = get_error(self.calibration_joint_pos, offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            
            self.joint_offsets.append(best_offset)
            
            # Calculate the resulting calibrated position
            calibrated_pos = sign * (raw - best_offset)
            
            if verbose:
                print(f"  Joint {i+1}: raw={np.rad2deg(raw):+7.1f}°, "
                      f"offset={np.rad2deg(best_offset):+7.1f}° ({best_offset/np.pi:.2f}π), "
                      f"sign={sign:+.0f}, "
                      f"result={np.rad2deg(calibrated_pos):+7.1f}° "
                      f"(target={np.rad2deg(target):+.1f}°, error={np.rad2deg(best_error):.2f}°)")

        # Any remaining joints (e.g. gripper) use current position as offset (defines current pos as zero)
        for j in range(self.num_arm_joints, len(curr_joints)):
            self.joint_offsets.append(float(curr_joints[j]))
            if verbose:
                print(f"  Joint {j+1} (gripper): offset={np.rad2deg(curr_joints[j]):.1f}° (raw as zero-ref)")

        self.joint_offsets = np.asarray(self.joint_offsets, dtype=float)
        print("-" * 60)
        print(f"Final offsets (rad): {[f'{x:.4f}' for x in self.joint_offsets]}")
        print("="*60 + "\n")

    def _match_start_pos(self) -> None:
        """Wait for leader arm to be moved to initial position."""
        while True:
            curr_pos, _, _, _ = self.get_leader_joint_states()
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0 : self.num_arm_joints]
            )
            if current_joint_error <= 0.6:
                break
            print(
                f"Please match starting joint position. Current error: {current_joint_error:.3f}"
            )
            time.sleep(0.5)

    def get_leader_joint_states(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Get current joint positions and velocities."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()

        # Apply offsets and signs for arm joints
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints]
            - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        joint_vel_arm = (
            joint_vel[0 : self.num_arm_joints]
            * self.joint_signs[0 : self.num_arm_joints]
        )

        # Process gripper
        self.leader_gripper_raw_rad = float(joint_pos[-1])
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[
            -1
        ]
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        self._last_gripper_vel = float(gripper_vel)

        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def set_leader_joint_torque(
        self, arm_torque: npt.NDArray[np.float64], gripper_torque: float
    ) -> None:
        """Apply torque to leader arm and gripper."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")

        # Handle case where gripper is part of arm joints (e.g. 7-DOF arm where last is gripper)
        if len(self.joint_signs) == len(arm_torque):
            # Merge gripper torque into the last arm joint
            arm_gripper_torque = arm_torque.copy()
            arm_gripper_torque[-1] += gripper_torque
        else:
            # Append gripper torque as a separate joint
            arm_gripper_torque = np.append(arm_torque, gripper_torque)

        self.driver.set_torque((arm_gripper_torque * self.torque_signs).tolist())

    def joint_limit_barrier(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
        gripper_joint_pos: float,
        gripper_joint_vel: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute joint limit repulsive torques."""
        # Arm joint limits
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask

        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_min_mask

        # Gripper limits
        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max)
                - self.joint_limit_kd * gripper_joint_vel
            )
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min)
                - self.joint_limit_kd * gripper_joint_vel
            )
        else:
            tau_l_gripper = 0.0

        return tau_l, tau_l_gripper

    def gravity_compensation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute gravity compensation torques using inverse dynamics."""
        # Pinocchio expects vectors sized to model.nq/model.nv.
        # Our leader splits arm and gripper; configs often set num_arm_joints=6 for UR-style leaders,
        # while the URDF may include the gripper as an extra joint (nq=7). We pad accordingly.
        pin_nq = int(getattr(self, "_pin_nq", len(arm_joint_pos)))
        pin_nv = int(getattr(self, "_pin_nv", len(arm_joint_vel)))

        # Build q/v vectors sized to the URDF
        q = np.zeros((pin_nq,), dtype=float)
        v = np.zeros((pin_nv,), dtype=float)

        n_arm = int(min(len(arm_joint_pos), pin_nq))
        q[:n_arm] = np.asarray(arm_joint_pos[:n_arm], dtype=float)
        n_arm_v = int(min(len(arm_joint_vel), pin_nv))
        v[:n_arm_v] = np.asarray(arm_joint_vel[:n_arm_v], dtype=float)

        # If the URDF has exactly one extra DOF beyond the arm, assume it's the leader gripper.
        if pin_nq == len(arm_joint_pos) + 1:
            q[len(arm_joint_pos)] = float(self.gripper_pos)
        if pin_nv == len(arm_joint_vel) + 1:
            v[len(arm_joint_vel)] = float(self._last_gripper_vel)

        a = np.zeros_like(v)

        tau_full = pin.rnea(  # type: ignore[attr-defined]
            self.pin_model,
            self.pin_data,
            q,
            v,
            a,
        )

        # Use only the arm torques for the leader arm joints
        tau_arm = np.asarray(tau_full[: self.num_arm_joints], dtype=float)
        tau_arm *= self.gravity_comp_modifier
        # Optional per-joint scaling (defaults to 1.0)
        if hasattr(self, "gravity_comp_gain_per_joint"):
            tau_arm *= self.gravity_comp_gain_per_joint
        self.tau_g = tau_arm
        return tau_arm

    def friction_compensation(
        self, arm_joint_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute friction compensation torques.
        
        This combines two approaches:
        1. Dither-based stiction compensation (original) - oscillates at low velocity
           to help overcome static friction
        2. Feedforward friction compensation (new) - adds Coulomb + viscous friction
           torque in direction of motion
           
        The feedforward approach is more physically accurate:
            τ_friction = τ_coulomb * sign(dq) + τ_viscous * dq
        """
        tau_ss = np.zeros(self.num_arm_joints)
        
        for i in range(self.num_arm_joints):
            vel = arm_joint_vel[i]
            
            # New: Per-joint feedforward friction compensation
            # Only apply when velocity exceeds deadband
            if abs(vel) > self.friction_velocity_deadband[i]:
                # Coulomb friction (constant, opposes motion)
                tau_ss[i] += self.friction_feedforward[i] * np.sign(vel)
                # Viscous friction (proportional to velocity)
                tau_ss[i] += self.viscous_friction[i] * vel
            
            # Original dither-based stiction compensation
            # This helps when arm is nearly stationary
            elif abs(vel) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        
        return tau_ss

    def null_space_regulation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute null-space regulation torques."""
        # Pad q to match URDF model size
        pin_nq = int(getattr(self, "_pin_nq", len(arm_joint_pos)))
        q = np.zeros((pin_nq,), dtype=float)
        n_arm = int(min(len(arm_joint_pos), pin_nq))
        q[:n_arm] = np.asarray(arm_joint_pos[:n_arm], dtype=float)

        if pin_nq == len(arm_joint_pos) + 1:
            q[len(arm_joint_pos)] = float(self.gripper_pos)

        J = pin.computeJointJacobian(self.pin_model, self.pin_data, q, self.num_arm_joints)  # type: ignore[attr-defined]
        
        # Slice Jacobian to only include arm joints
        J_arm = J[:, :self.num_arm_joints]

        J_dagger = np.linalg.pinv(J_arm)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J_arm
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (
            -self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel
        )
        return tau_n

    def get_follower_joint_torques(self) -> np.ndarray:
        """Get external joint torques from the follower robot.

        This method retrieves the current joint torques from the follower robot
        for force-feedback. The torques are gravity/friction compensated by the
        follower's controller (e.g., UR5e's getActualJointTorques()).

        Returns:
            np.ndarray: External joint torques from follower (length num_arm_joints)
        """
        # 1. Try via ZMQ client (works for both local threads and remote/network)
        if self.teleop_enabled and self.teleop_client is not None:
            try:
                # Use client method if available (requires updated ZMQClientRobot)
                if hasattr(self.teleop_client, "get_joint_torques"):
                    torques = self.teleop_client.get_joint_torques()
                    if len(torques) >= self.num_arm_joints:
                        return np.array(torques[: self.num_arm_joints])
                    if len(torques) > 0:
                        return np.array(torques)
            except Exception:
                pass

        if not self.teleop_enabled or self.teleop_robot_server is None:
            return np.zeros(self.num_arm_joints)

        try:
            # 2. Try direct access (Direct Access Fallback)
            follower = self.teleop_robot_server

            # Handle ZMQServerRobot wrapper (supports both robot and _robot)
            if hasattr(follower, "_robot"):
                follower = follower._robot
            elif hasattr(follower, "robot"):
                follower = follower.robot

            # Check if the follower robot has a get_joint_torques method
            if hasattr(follower, "get_joint_torques"):
                torques = follower.get_joint_torques()
                # Ensure we only get arm joint torques (not gripper)
                if len(torques) >= self.num_arm_joints:
                    return np.array(torques[: self.num_arm_joints])
                return np.array(torques)

            # Alternative: check for getActualJointTorques (ur-rtde style)
            if hasattr(follower, "r_inter") and hasattr(
                follower.r_inter, "getActualJointTorques"
            ):
                torques = follower.r_inter.getActualJointTorques()
                return np.array(torques[: self.num_arm_joints])

            # Fallback: return zeros if no torque sensing available
            return np.zeros(self.num_arm_joints)

        except Exception as e:
            # print(f"Warning: Failed to get follower torques: {e}")
            return np.zeros(self.num_arm_joints)

    def torque_feedback(
        self,
        external_torque: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute joint torque for force-feedback based on follower's external torques.

        This implements the force-feedback loop from FACTR (Equation 1 in Section III.A):
        τ_ff = -gain/motor_scalar * τ_ext - damping * q̇

        The external torques from the follower robot are scaled and applied to the
        leader arm to provide haptic feedback of contact forces.

        Args:
            external_torque: External joint torques from follower robot (Nm)
            arm_joint_vel: Current leader arm joint velocities (rad/s)

        Returns:
            np.ndarray: Force-feedback torques to apply to leader arm (Nm)
        """
        # Apply feedback gain and motor scaling (as per FACTR paper Eq. 1)
        tau_ff = (
            -1.0
            * self.torque_feedback_gain
            / self.torque_feedback_motor_scalar
            * external_torque
        )
        # Add velocity damping for stability
        tau_ff -= self.torque_feedback_damping * arm_joint_vel
        return tau_ff

    def get_follower_arm_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current arm position and velocity from the follower robot.

        Returns:
            Tuple of (positions, velocities) arrays, each of length num_arm_joints
        """
        if not self.teleop_enabled or self.teleop_robot_server is None:
            return np.zeros(self.num_arm_joints), np.zeros(self.num_arm_joints)

        try:
            follower = self.teleop_robot_server

            # Handle ZMQServerRobot wrapper
            if hasattr(follower, "robot"):
                follower = follower.robot

            # Try get_observations first (most complete)
            if hasattr(follower, "get_observations"):
                obs = follower.get_observations()
                pos = np.array(obs.get("joint_positions", np.zeros(self.num_arm_joints)))
                vel = np.array(obs.get("joint_velocities", np.zeros(self.num_arm_joints)))
                # Ensure we only get arm joints (not gripper)
                return pos[:self.num_arm_joints], vel[:self.num_arm_joints]

            # Fallback: try get_joint_state
            if hasattr(follower, "get_joint_state"):
                pos = np.array(follower.get_joint_state())
                return pos[:self.num_arm_joints], np.zeros(self.num_arm_joints)

            # UR-RTDE style
            if hasattr(follower, "r_inter"):
                pos = np.array(follower.r_inter.getActualQ())
                vel = np.array(follower.r_inter.getActualQd())
                return pos[:self.num_arm_joints], vel[:self.num_arm_joints]

            return np.zeros(self.num_arm_joints), np.zeros(self.num_arm_joints)

        except Exception as e:
            print(f"Warning: Failed to get follower arm state: {e}")
            return np.zeros(self.num_arm_joints), np.zeros(self.num_arm_joints)

    def force_position_feedback(
        self,
        leader_arm_pos: npt.NDArray[np.float64],
        leader_arm_vel: npt.NDArray[np.float64],
        follower_arm_pos: npt.NDArray[np.float64],
        follower_arm_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute joint torque using force-position feedback.

        This implements direct position-based force feedback:
        τ_ff = Kp * (q_follower - q_leader) + Kd * (dq_follower - dq_leader)

        When the follower robot hits an obstacle and stops, the position error
        between leader and follower creates a restoring torque on the leader,
        providing direct haptic feedback of the obstacle.

        This is an alternative to FACTR's scaled torque feedback and can feel
        more intuitive for some applications.

        Args:
            leader_arm_pos: Current leader arm joint positions (rad)
            leader_arm_vel: Current leader arm joint velocities (rad/s)
            follower_arm_pos: Current follower arm joint positions (rad)
            follower_arm_vel: Current follower arm joint velocities (rad/s)

        Returns:
            np.ndarray: Force-feedback torques to apply to leader arm (Nm)
        """
        # Position error: when follower is behind leader (blocked), creates resistance
        pos_error = follower_arm_pos - leader_arm_pos
        vel_error = follower_arm_vel - leader_arm_vel

        # PD control law
        tau_ff = self.force_position_kp * pos_error + self.force_position_kd * vel_error

        # Safety clamp per joint
        tau_ff = np.clip(tau_ff, -self.force_position_max_torque, self.force_position_max_torque)

        return tau_ff

    def get_follower_gripper_feedback(self) -> Dict[str, Any]:
        """Get gripper feedback from the follower robot.

        For Robotiq 2F-85, this includes:
        - position: Current normalized position
        - is_gripping: Whether an object is detected
        - position_error: Error between commanded and actual position
        - force_estimate: Estimated grip force

        Returns:
            Dict with gripper feedback data
        """
        if not self.teleop_enabled or self.teleop_robot_server is None:
            return {"position": 0.0, "is_gripping": False, "force_estimate": 0.0}

        try:
            follower = self.teleop_robot_server

            # Handle ZMQServerRobot wrapper
            if hasattr(follower, "robot"):
                follower = follower.robot

            # Check for get_gripper_feedback method (UR5e with Robotiq)
            if hasattr(follower, "get_gripper_feedback"):
                return follower.get_gripper_feedback()

            # Alternative: check observations for gripper_feedback
            if hasattr(follower, "get_observations"):
                obs = follower.get_observations()
                if "gripper_feedback" in obs:
                    return obs["gripper_feedback"]

            # Fallback: return empty feedback
            return {"position": 0.0, "is_gripping": False, "force_estimate": 0.0}

        except Exception as e:
            print(f"Warning: Failed to get follower gripper feedback: {e}")
            return {"position": 0.0, "is_gripping": False, "force_N": 0.0, "force_normalized": 0.0}

    def gripper_feedback(
        self,
        leader_gripper_pos: float,
        leader_gripper_vel: float,
        follower_feedback: Dict[str, Any],
    ) -> float:
        """Compute gripper torque for force-feedback based on follower gripper state.

        Implements position-position force feedback for the gripper:
        - When follower gripper is gripping an object, the motor current limit is reached
        - The Robotiq FOR setting (0-255) maps to 20-235 N grip force
        - This force is fed back as resistance torque to the leader gripper

        Args:
            leader_gripper_pos: Current leader gripper position (rad)
            leader_gripper_vel: Current leader gripper velocity (rad/s)
            follower_feedback: Feedback dict from follower gripper containing:
                - 'is_gripping': Whether object detected
                - 'force_N': Grip force in Newtons (0 or 20-235 N)
                - 'force_normalized': Normalized force [0, 1]
                - 'position': Current gripper position [0, 1]

        Returns:
            float: Force-feedback torque to apply to leader gripper (Nm)
        """
        # Get follower gripper state
        follower_pos = follower_feedback.get("position", 0.0)
        is_gripping = follower_feedback.get("is_gripping", False)
        
        # Use normalized force for feedback (0-1 range based on 20-235 N)
        force_normalized = follower_feedback.get("force_normalized", 0.0)
        # Fallback to old format if new format not available
        if force_normalized == 0.0 and "force_estimate" in follower_feedback:
            force_normalized = follower_feedback.get("force_estimate", 0.0)

        if not is_gripping:
            # No object detected - no feedback torque needed
            return 0.0

        # Position-position feedback: resist leader movement when follower is blocked
        # Normalize leader gripper position to [0, 1] range for comparison
        leader_pos_normalized = leader_gripper_pos / max(self.gripper_limit_max, 0.01)
        leader_pos_normalized = np.clip(leader_pos_normalized, 0.0, 1.0)

        # Position error: how much the leader has moved beyond follower
        position_error = leader_pos_normalized - follower_pos

        # Apply feedback torque proportional to position error and force
        # force_normalized is 0-1 based on motor current limit (20-235 N)
        # Negative sign: resist the leader from closing further when object is gripped
        tau_gripper = -self.gripper_feedback_gain * position_error * (1.0 + force_normalized)

        # Add velocity damping
        tau_gripper -= self.gripper_feedback_damping * leader_gripper_vel

        return float(tau_gripper)

    def _setup_visualization(self):
        """Setup real-time visualization of torque components."""
        self._viz_logger = TorqueComponentLogger(history_len=int(10.0 / self.dt))
        self._viz_logger.initialize(self.num_arm_joints)
        self._viz_start_time = time.time()

        # Setup plot in main thread (matplotlib requirement)
        plt.ion()
        self._viz_fig, self._viz_axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
        self._viz_axes = self._viz_axes.flatten()
        self._viz_fig.suptitle("GELLO Gravity Compensation - Live Torque Components", fontsize=14)

        self._joint_names = ["J1 (Base)", "J2 (Shoulder)", "J3 (Elbow)",
                             "J4 (Wrist1)", "J5 (Wrist2)", "J6 (Wrist3)"]

        self._component_colors = {
            "gravity": "#2ecc71",
            "friction": "#e74c3c",
            "damping": "#3498db",
            "null": "#9b59b6",
            "limit": "#e67e22",
            "feedback": "#1abc9c",
            "total": "#2c3e50",
        }

        # Initialize lines
        self._viz_lines: Dict[str, List] = {
            "gravity": [], "friction": [], "damping": [],
            "null": [], "limit": [], "feedback": [], "total": []
        }

        for joint_idx, ax in enumerate(self._viz_axes):
            ax.set_title(self._joint_names[joint_idx])
            ax.set_ylabel("Torque [Nm]")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

            for comp_name, color in self._component_colors.items():
                linewidth = 2.5 if comp_name == "total" else 1.2
                line, = ax.plot([], [], color=color, linewidth=linewidth, alpha=0.8)
                self._viz_lines[comp_name].append(line)

            if joint_idx == 0:
                ax.legend(
                    [self._viz_lines[k][0] for k in self._component_colors.keys()],
                    ["Gravity", "Friction", "Damping", "Null-Space", "Limits", "Feedback", "TOTAL"],
                    loc='upper left', fontsize=7
                )

        for ax in self._viz_axes[3:]:
            ax.set_xlabel("Time [s]")

        plt.tight_layout()
        self._viz_fig.canvas.draw()
        plt.pause(0.01)

        print("Visualization enabled - torque components will be plotted in real-time")

    def _update_visualization(self):
        """Update the visualization plot."""
        if self._viz_logger is None or len(self._viz_logger.time_history) < 2:
            return

        t_array = np.array(self._viz_logger.time_history)
        t_rel = t_array - t_array[0]

        for joint_idx in range(self.num_arm_joints):
            # Update each component line
            self._viz_lines["gravity"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_gravity[joint_idx])
            )
            self._viz_lines["friction"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_friction[joint_idx])
            )
            self._viz_lines["damping"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_damping[joint_idx])
            )
            self._viz_lines["null"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_null[joint_idx])
            )
            self._viz_lines["limit"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_limit[joint_idx])
            )
            self._viz_lines["feedback"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_feedback[joint_idx])
            )
            self._viz_lines["total"][joint_idx].set_data(
                t_rel, np.array(self._viz_logger.tau_total[joint_idx])
            )

            # Update axis limits
            ax = self._viz_axes[joint_idx]
            ax.set_xlim(t_rel[0], t_rel[-1])
            ax.relim()
            ax.autoscale_view(scalex=False)

        self._viz_fig.canvas.draw_idle()
        self._viz_fig.canvas.flush_events()

    def control_loop_step(self) -> None:
        """Execute one step of the control loop."""
        # Get current joint states
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = (
            self.get_leader_joint_states()
        )

        # Initialize torque commands
        torque_arm = np.zeros(self.num_arm_joints)

        # Track individual components for visualization
        tau_gravity_comp = np.zeros(self.num_arm_joints)
        tau_friction_comp = np.zeros(self.num_arm_joints)
        tau_damping_comp = np.zeros(self.num_arm_joints)
        tau_null_comp = np.zeros(self.num_arm_joints)
        tau_limit_comp = np.zeros(self.num_arm_joints)
        tau_feedback_comp = np.zeros(self.num_arm_joints)

        # Joint limit barriers
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l
        tau_limit_comp = torque_l.copy()

        # Null space regulation
        tau_null = self.null_space_regulation(leader_arm_pos, leader_arm_vel)
        torque_arm += tau_null
        tau_null_comp = tau_null.copy()

        # Gravity compensation and friction compensation
        tau_gravity = np.zeros(self.num_arm_joints)
        if self.enable_gravity_comp:
            tau_gravity = self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += tau_gravity
            tau_gravity_comp = tau_gravity.copy()

            tau_friction = self.friction_compensation(leader_arm_vel)
            torque_arm += tau_friction
            tau_friction_comp = tau_friction.copy()

            if self.gravity_comp_velocity_damping != 0.0:
                tau_damping = -self.gravity_comp_velocity_damping * leader_arm_vel
                torque_arm += tau_damping
                tau_damping_comp = tau_damping.copy()

        # Torque feedback (force-feedback from follower robot arm) - FACTR scaled style
        if self.enable_torque_feedback:
            external_joint_torque = self.get_follower_joint_torques()
            self._follower_torques = external_joint_torque
            tau_fb = self.torque_feedback(external_joint_torque, leader_arm_vel)
            torque_arm += tau_fb
            tau_feedback_comp = tau_fb.copy()

        # Force-Position feedback (alternative to FACTR) - position error based
        if self.enable_force_position_feedback:
            follower_pos, follower_vel = self.get_follower_arm_state()
            if self.map_index is not None and self.map_signs is not None:
                mapped_follower_pos = np.zeros(self.num_arm_joints)
                mapped_follower_vel = np.zeros(self.num_arm_joints)
                map_len = min(len(self.map_index), len(follower_pos))
                for i, idx in enumerate(self.map_index[:map_len]):
                    if idx < self.num_arm_joints:
                        sign = self.map_signs[i] if i < len(self.map_signs) else 1.0
                        offset = self.map_offsets[i] if self.map_offsets is not None and i < len(
                            self.map_offsets) else 0.0
                        mapped_follower_pos[idx] = (follower_pos[i] - offset) / sign if sign != 0 else follower_pos[i]
                        mapped_follower_vel[idx] = follower_vel[i] / sign if sign != 0 else follower_vel[i]
                follower_pos = mapped_follower_pos
                follower_vel = mapped_follower_vel
            self._follower_arm_pos = follower_pos
            self._follower_arm_vel = follower_vel
            tau_fp = self.force_position_feedback(
                leader_arm_pos, leader_arm_vel, follower_pos, follower_vel
            )
            torque_arm += tau_fp
            tau_feedback_comp = tau_fp.copy()

        # Gripper feedback (force-feedback from follower gripper)
        if self.enable_gripper_feedback:
            follower_gripper_fb = self.get_follower_gripper_feedback()
            self._follower_gripper_feedback = follower_gripper_fb
            torque_gripper += self.gripper_feedback(
                leader_gripper_pos, leader_gripper_vel, follower_gripper_fb
            )

        # === LOG FOR VISUALIZATION ===
        if self.enable_visualization and self._viz_logger is not None:
            current_time = time.time() - self._viz_start_time
            self._viz_logger.log(
                timestamp=current_time,
                positions=leader_arm_pos,
                velocities=leader_arm_vel,
                tau_gravity=tau_gravity_comp,
                tau_friction=tau_friction_comp,
                tau_damping=tau_damping_comp,
                tau_null=tau_null_comp,
                tau_limit=tau_limit_comp,
                tau_feedback=tau_feedback_comp,
                tau_total=torque_arm,
            )
        
        # Debug output (every 100 iterations = ~0.2 second at 500Hz)
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 0:
            print(f"\n[DEBUG @ {self._debug_counter / (1/self.dt):.1f}s]")
            print(f"  Arm pos (deg): {[f'{np.rad2deg(x):+7.1f}' for x in leader_arm_pos]}")
            print(f"  Gravity τ (Nm): {[f'{x:+.4f}' for x in tau_gravity]}")
            print(f"  Total τ (Nm):   {[f'{x:+.4f}' for x in torque_arm]}")
            print(
                f"  Applied τ*motor_sign: {[f'{x:+.4f}' for x in torque_arm * self.torque_signs[:self.num_arm_joints]]}"
            )
            print(f"UR Joint Torques (Nm): {[f'{x:+.4f}' for x in self.get_follower_joint_torques()]}")
        
        # Apply torques only if GC is enabled (torque mode is off otherwise)
        if self.enable_gravity_comp:
            self.set_leader_joint_torque(torque_arm, torque_gripper)


    def run(self) -> None:
        """Run the main control loop."""
        print(f"Starting gravity compensation control loop at {1 / self.dt:.1f} Hz")
        if self.enable_visualization:
            print("Visualization enabled - close plot window to stop")
        print("Press Ctrl+C to stop")

        self.running = True
        # Start teleop thread now that running is True
        if self.teleop_enabled and self.teleop_prepared and self.teleop_thread is None:
            self.teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
            self.teleop_thread.start()
            print("Teleop started.")

        viz_update_counter = 0

        try:
            while self.running:
                start_time = time.time()

                self.control_loop_step()

                # Update visualization periodically (every 10 iterations)
                if self.enable_visualization:
                    viz_update_counter += 1
                    if viz_update_counter % 10 == 0:
                        self._update_visualization()
                        # Check if plot window was closed
                        if not plt.fignum_exists(self._viz_fig.number):
                            print("\nVisualization window closed, stopping...")
                            self.running = False
                            break

                # Maintain loop timing
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"Warning: Control loop overrun by {elapsed - self.dt:.4f}s ")
                    print(f"Control loop step took {elapsed:.4f}s ({1/elapsed:.1f} Hz)")

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()
            if self.enable_visualization:
                plt.ioff()
                plt.close('all')

    def shutdown(self) -> None:
        """Safely shutdown the system."""
        self.running = False
        # Stop teleop thread and close ZMQ resources first
        try:
            if self.teleop_thread is not None and self.teleop_thread.is_alive():
                # Give teleop loop a cycle to exit since self.running is False
                time.sleep(0.05)
                # Join briefly
                self.teleop_thread.join(timeout=1.0)
            if self.teleop_client is not None and hasattr(self.teleop_client, "close"):
                self.teleop_client.close()
            # Attempt to stop underlying server if it has a stop
            if self.teleop_robot_server is not None and hasattr(
                self.teleop_robot_server, "stop"
            ):
                try:
                    self.teleop_robot_server.stop()
                except Exception:
                    pass
        except Exception as e:
            print(f"Teleop shutdown warning: {e}")

        if hasattr(self, "driver") and self.driver is not None:
            print("Disabling motor torques...")
            try:
                self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
            except Exception:
                pass
            self.driver.set_torque_mode(False)
            self.driver.close()
        print("Shutdown complete")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone FACTR Gravity Compensation"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="xdof/sandboxs/jliu/factr_grav_comp_demo.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Enable real-time visualization of torque components",
    )

    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        # Create and run gravity compensation system
        system = FACTRGravityCompensation(args.config, enable_visualization=args.visualize)

        # Set up signal handler for clean shutdown
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal")
            system.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the system
        system.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
