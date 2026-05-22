from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from threading import Thread
import threading
from typing import Any, Optional

import numpy as np

from gello.bilat_4ch.gello_ur5e_observer_shi import (
    MinimalistEstimatorConfig,
    MinimalistTorqueEstimator,
    MotorParams,
    MotorType,
)
from gello.cr_dagger.core.epsilon_recorder import EpsilonRecorder
from gello.cr_dagger.ipc.shared_observation_snapshot import SharedObservationSnapshot
from gello.cr_dagger.ipc.shared_trajectory_buffer import SharedTrajectoryBuffer
from gello.cr_dagger.policy.filtered_ddq import FilteredDDQComputer
from gello.cr_dagger.policy.policy_worker import policy_worker
from gello.cr_dagger.policy.trajectory_interpolator import TrajectoryInterpolator
from gello.dynamixel.driver import (
    ADDR_CURRENT_LIMIT,
    ADDR_GOAL_CURRENT,
    ADDR_POSITION_D_GAIN,
    ADDR_POSITION_I_GAIN,
    ADDR_POSITION_P_GAIN,
    ADDR_VELOCITY_I_GAIN,
    ADDR_VELOCITY_P_GAIN,
)
from gello.factr.gravity_compensation import FACTRGravityCompensation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_leader_urdf(config_path: Path, leader_urdf: str) -> Path:
    for candidate in [
        (config_path.parent / leader_urdf).resolve(),
        (REPO_ROOT / "gello" / "factr" / "urdf" / Path(leader_urdf).name).resolve(),
        (REPO_ROOT / leader_urdf).resolve(),
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"URDF not found for leader_urdf='{leader_urdf}'")


def _infer_motor_params(servo_types: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    params_by_servo = {
        "XC330_T288_T": (288.35, 1.136 / 288.35),
        "XM430_W210_T": (212.6, 1.304 / 212.6),
        "XM430_W350_T": (353.5, 1.783 / 353.5),
    }
    gear_ratios, kts = [], []
    for s in servo_types[:n]:
        gr, kt = params_by_servo.get(s, (1.0, 0.00504))
        gear_ratios.append(float(gr))
        kts.append(float(kt))
    while len(gear_ratios) < n:
        gear_ratios.append(1.0)
        kts.append(0.00504)
    return np.asarray(gear_ratios, dtype=float), np.asarray(kts, dtype=float)


def _build_shi(
    system: FACTRGravityCompensation, config_path: Path, n: int
) -> MinimalistTorqueEstimator:
    leader_urdf = str(system.config["arm_teleop"]["leader_urdf"])
    urdf_path = _resolve_leader_urdf(config_path, leader_urdf)
    servo_types = list(system.config["dynamixel"]["servo_types"])
    gear_ratio, kt = _infer_motor_params(servo_types, n)

    return MinimalistTorqueEstimator(
        MinimalistEstimatorConfig(
            urdf_path=str(urdf_path),
            motor_params=MotorParams(
                kt=kt,
                gear_ratio=gear_ratio,
                eta=np.full(n, 0.65, dtype=float),
                motor_type=MotorType.CURRENT,
            ),
            alpha_ema=0.5,
            vel_threshold=0.05,
        )
    )


def _parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _expand_list(values: list[float], n: int, fill: float = 0.0) -> np.ndarray:
    if not values:
        return np.full(n, fill, dtype=float)
    if len(values) == 1:
        return np.full(n, values[0], dtype=float)
    if len(values) < n:
        return np.asarray(values + [fill] * (n - len(values)), dtype=float)
    return np.asarray(values[:n], dtype=float)


def _parse_optional_int_array(value: str | None, n: int) -> np.ndarray | None:
    if value is None or not str(value).strip():
        return None
    raw = [int(round(float(v.strip()))) for v in str(value).split(",") if v.strip()]
    if not raw:
        return None
    if len(raw) == 1:
        return np.full(n, raw[0], dtype=np.int32)
    if len(raw) < n:
        raw = raw + [raw[-1]] * (n - len(raw))
    return np.asarray(raw[:n], dtype=np.int32)


def _int_array_metadata(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    return [int(round(float(v.strip()))) for v in str(value).split(",") if v.strip()]


def _joint_array_from_value(value: Any, n: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(n, float(default), dtype=float)
    if isinstance(value, str):
        parsed = _parse_float_list(value)
        if not parsed:
            return np.full(n, float(default), dtype=float)
        arr = np.asarray(parsed, dtype=float).reshape(-1)
    elif np.isscalar(value):
        return np.full(n, float(value), dtype=float)
    else:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.full(n, float(default), dtype=float)

    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if arr.size < n:
        arr = np.pad(
            arr,
            (0, n - arr.size),
            mode="constant",
            constant_values=float(arr[-1]),
        )
    return arr[:n].astype(float, copy=True)


def _format_joint_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3g}" for v in values) + "]"


class ContactGate:
    """Lightweight online gate for contact hypotheses from residual torque."""

    NO_CONTACT = 0.0
    POSSIBLE_CONTACT = 1.0
    CONFIRMED_CONTACT = 2.0
    CORRECTION_ACTIVE = 3.0

    def __init__(
        self,
        n_joints: int,
        residual_low: float = 0.15,
        residual_high: float = 0.45,
        accel_gate: float = 8.0,
        window: int = 8,
    ):
        self.residual_low = float(residual_low)
        self.residual_high = max(float(residual_high), float(residual_low) + 1e-6)
        self.accel_gate = max(float(accel_gate), 1e-6)
        self.sign_hist = np.zeros((max(int(window), 1), int(n_joints)), dtype=float)
        self.idx = 0
        self.count = 0

    def update(
        self,
        tau_residual: np.ndarray,
        ddq_ref: np.ndarray,
        correction_active: bool = False,
    ) -> tuple[float, float]:
        residual = np.asarray(tau_residual, dtype=float)
        self.sign_hist[self.idx] = np.sign(residual)
        self.idx = (self.idx + 1) % self.sign_hist.shape[0]
        self.count = min(self.count + 1, self.sign_hist.shape[0])

        mag = float(np.linalg.norm(residual))
        mag_prob = np.clip(
            (mag - self.residual_low) / (self.residual_high - self.residual_low),
            0.0,
            1.0,
        )
        ddq_norm = float(np.linalg.norm(ddq_ref))
        motion_gate = 1.0 / np.sqrt(1.0 + (ddq_norm / self.accel_gate) ** 2)

        hist = self.sign_hist[: self.count]
        if self.count < 2 or mag <= self.residual_low:
            consistency = 0.0
        else:
            consistency = float(np.max(np.abs(np.sum(hist, axis=0))) / self.count)

        probability = float(np.clip(mag_prob * motion_gate * consistency, 0.0, 1.0))
        if correction_active and probability >= 0.6:
            state = self.CORRECTION_ACTIVE
        elif probability >= 0.6:
            state = self.CONFIRMED_CONTACT
        elif probability >= 0.25:
            state = self.POSSIBLE_CONTACT
        else:
            state = self.NO_CONTACT
        return probability, state


class BotaMiniOneReader:
    """Thin lifecycle wrapper around bota_driver for optional Phase-B wrench input."""

    def __init__(self, config_path: Path, driver_tare: bool = False):
        try:
            import bota_driver
        except ImportError as exc:
            raise RuntimeError(
                "bota_driver is not importable. Activate the environment with "
                "the Bota Systems Python driver or run without --bota-enable."
            ) from exc

        if not config_path.exists():
            raise FileNotFoundError(f"Bota config not found: {config_path}")

        self.driver = bota_driver.BotaDriver(str(config_path))
        self.driver_tare = bool(driver_tare)
        self.active = False
        self.expected_hz: float | None = None
        self.last_timestamp_us: int | None = None
        self.new_frames = 0
        self.duplicate_frames = 0
        self.status_flags = np.zeros(4, dtype=float)

    def start(self) -> None:
        print(f"[BOTA] driver version: {self.driver.get_driver_version_string()}")
        if not self.driver.configure():
            raise RuntimeError("Bota configure() failed")
        if self.driver_tare:
            print("[BOTA] driver tare in INACTIVE state")
            if not self.driver.tare():
                raise RuntimeError("Bota tare() failed")
        if not self.driver.activate():
            raise RuntimeError("Bota activate() failed")
        self.active = True
        try:
            dt = self.driver.get_expected_timestep().total_seconds()
            if dt > 0.0:
                self.expected_hz = 1.0 / float(dt)
                print(f"[BOTA] expected rate: {self.expected_hz:.2f} Hz")
        except Exception:
            self.expected_hz = None

    def read_latest(self) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, bool]:
        if not self.active:
            raise RuntimeError("Bota reader is not active")
        frame = self.driver.read_frame()
        status = frame.status
        wrench = np.asarray(list(frame.force[:3]) + list(frame.torque[:3]), dtype=float)
        flags = np.asarray(
            [
                float(bool(status.throttled)),
                float(bool(status.overrange)),
                float(bool(status.invalid)),
                float(bool(status.raw)),
            ],
            dtype=float,
        )
        timestamp_us = int(frame.timestamp)
        is_new = self.last_timestamp_us != timestamp_us
        if is_new:
            self.new_frames += 1
        else:
            self.duplicate_frames += 1
        self.last_timestamp_us = timestamp_us
        self.status_flags = flags
        return wrench, flags, timestamp_us, float(frame.temperature), np.zeros(6, dtype=float), is_new

    def close(self) -> None:
        try:
            if self.active:
                self.driver.deactivate()
                self.active = False
        except Exception as exc:
            print(f"[WARN] Bota deactivate failed: {exc}")
        try:
            self.driver.cleanup()
        except Exception:
            pass
        try:
            self.driver.shutdown()
        except Exception as exc:
            print(f"[WARN] Bota shutdown failed: {exc}")


class BotaWrenchConditioner:
    """raw -> bias -> frame transform -> gravity -> low-pass -> deadband -> saturation."""

    def __init__(
        self,
        cutoff_hz: float,
        alpha: float,
        deadband: np.ndarray,
        saturation: np.ndarray,
        gravity_comp: bool,
        payload_mass_kg: float,
        payload_com_sensor: np.ndarray,
        gravity_sign: float,
    ):
        self.cutoff_hz = max(float(cutoff_hz), 0.0)
        self.alpha = float(np.clip(float(alpha), 0.0, 1.0))
        self.deadband = np.asarray(deadband, dtype=float).reshape(6)
        self.saturation = np.asarray(saturation, dtype=float).reshape(6)
        self.gravity_comp = bool(gravity_comp)
        self.payload_mass_kg = max(float(payload_mass_kg), 0.0)
        self.payload_com_sensor = np.asarray(payload_com_sensor, dtype=float).reshape(3)
        self.gravity_sign = float(gravity_sign)
        self.bias_sensor = np.zeros(6, dtype=float)
        self.filtered_base = np.zeros(6, dtype=float)
        self.initialized = False
        self.last_filter_timestamp_us: int | None = None

    def _filter_alpha(self, dt_s: float) -> float:
        if self.cutoff_hz > 0.0:
            if dt_s <= 0.0:
                return 1.0
            return 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt_s)
        if self.alpha > 0.0:
            return self.alpha
        return 1.0

    def set_bias(self, samples_sensor: list[np.ndarray]) -> None:
        if samples_sensor:
            self.bias_sensor = np.mean(np.asarray(samples_sensor, dtype=float), axis=0)
            print(f"[BOTA] software bias: {np.array2string(self.bias_sensor, precision=4)}")

    def update(
        self,
        wrench_sensor_raw: np.ndarray,
        r_base_sensor: np.ndarray,
        dt: float,
        timestamp_us: int | None = None,
        is_new_frame: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        wrench_sensor = np.asarray(wrench_sensor_raw, dtype=float).reshape(6) - self.bias_sensor
        force_base = r_base_sensor @ wrench_sensor[:3]
        torque_base = r_base_sensor @ wrench_sensor[3:]
        wrench_base = np.concatenate([force_base, torque_base])

        if self.gravity_comp and self.payload_mass_kg > 0.0:
            g_base = np.array([0.0, 0.0, -9.80665], dtype=float)
            force_g_base = self.payload_mass_kg * g_base
            r_com_base = r_base_sensor @ self.payload_com_sensor
            torque_g_base = np.cross(r_com_base, force_g_base)
            wrench_base = wrench_base - self.gravity_sign * np.concatenate([force_g_base, torque_g_base])

        if not self.initialized:
            self.filtered_base = wrench_base.copy()
            self.initialized = True
            self.last_filter_timestamp_us = int(timestamp_us) if timestamp_us is not None else None
        else:
            should_update = bool(is_new_frame)
            filter_dt_s = max(float(dt), 0.0)
            if timestamp_us is not None:
                ts = int(timestamp_us)
                should_update = self.last_filter_timestamp_us != ts
                if should_update and self.last_filter_timestamp_us is not None:
                    filter_dt_s = max(0.0, (ts - self.last_filter_timestamp_us) * 1e-6)
                if should_update:
                    self.last_filter_timestamp_us = ts
            if should_update:
                alpha = self._filter_alpha(filter_dt_s)
                self.filtered_base = self.filtered_base + alpha * (wrench_base - self.filtered_base)

        conditioned = self.filtered_base.copy()
        conditioned = np.where(
            np.abs(conditioned) > self.deadband,
            conditioned - np.sign(conditioned) * self.deadband,
            0.0,
        )
        conditioned = np.clip(conditioned, -self.saturation, self.saturation)
        return wrench_base, conditioned


class GelloTaskspaceKinematics:
    """Pinocchio FK/Jacobian helper using base-aligned frame coordinates."""

    def __init__(self, system: FACTRGravityCompensation, n_joints: int, frame_name: str | None = None):
        import pinocchio as pin

        self.pin = pin
        self.model = system.pin_model
        self.data = system.pin_data
        self.n_joints = int(n_joints)
        self.nq = int(getattr(self.model, "nq", self.n_joints))
        if frame_name:
            frame_id = int(self.model.getFrameId(frame_name))
            if frame_id >= int(self.model.nframes):
                raise ValueError(f"Pinocchio frame not found: {frame_name}")
            self.frame_id = frame_id
            self.frame_name = str(frame_name)
        else:
            self.frame_id = int(self.model.nframes - 1)
            self.frame_name = str(self.model.frames[self.frame_id].name)

    def _q_full(self, q: np.ndarray) -> np.ndarray:
        q_full = np.zeros(self.nq, dtype=float)
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        q_full[: min(q_arr.size, self.nq)] = q_arr[: min(q_arr.size, self.nq)]
        return q_full

    def pose_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_full = self._q_full(q)
        self.pin.forwardKinematics(self.model, self.data, q_full)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        jac = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q_full,
            self.frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:, : self.n_joints]
        return (
            np.asarray(pose.translation, dtype=float).copy(),
            np.asarray(pose.rotation, dtype=float).copy(),
            np.asarray(jac, dtype=float).copy(),
        )


class CartesianAdmittance6D:
    """Base-frame 6D admittance that produces a small task-space offset."""

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
        leak: float,
        max_offset: np.ndarray,
        axis_mask: np.ndarray,
    ):
        self.mass = np.maximum(np.asarray(mass, dtype=float).reshape(6), 1e-6)
        self.damping = np.maximum(np.asarray(damping, dtype=float).reshape(6), 0.0)
        self.stiffness = np.maximum(np.asarray(stiffness, dtype=float).reshape(6), 0.0)
        self.leak = max(float(leak), 0.0)
        self.max_offset = np.maximum(np.asarray(max_offset, dtype=float).reshape(6), 1e-9)
        self.axis_mask = np.asarray(axis_mask, dtype=float).reshape(6)
        self.x = np.zeros(6, dtype=float)
        self.dx = np.zeros(6, dtype=float)

    def step(self, wrench_base: np.ndarray, dt: float) -> np.ndarray:
        wrench = np.asarray(wrench_base, dtype=float).reshape(6) * self.axis_mask
        ddx = (wrench - self.damping * self.dx - self.stiffness * self.x) / self.mass
        self.dx = (1.0 - self.leak * float(dt)) * (self.dx + ddx * float(dt))
        self.x = self.x + self.dx * float(dt)
        self.x = np.clip(self.x, -self.max_offset, self.max_offset)
        self.x *= self.axis_mask
        self.dx *= self.axis_mask
        return self.x.copy()


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=float)
    if theta < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * math.sin(theta)) * vee


def _pose_error_base(
    pos_current: np.ndarray,
    rot_current: np.ndarray,
    pos_ref: np.ndarray,
    rot_ref: np.ndarray,
) -> np.ndarray:
    pos_err = np.asarray(pos_current, dtype=float).reshape(3) - np.asarray(pos_ref, dtype=float).reshape(3)
    rot_err = _so3_log(np.asarray(rot_current, dtype=float).reshape(3, 3) @ np.asarray(rot_ref, dtype=float).reshape(3, 3).T)
    return np.concatenate([pos_err, rot_err])


class SE3PoseAdmittance6D:
    """6D wrench admittance with SE(3) pose-error logging."""

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
        leak: float,
        max_offset: np.ndarray,
        axis_mask: np.ndarray,
        stiction: np.ndarray,
        wrench_sign: float,
        output_mode: str = "offset",
    ):
        self.mass = np.maximum(np.asarray(mass, dtype=float).reshape(6), 1e-6)
        self.damping = np.maximum(np.asarray(damping, dtype=float).reshape(6), 0.0)
        self.stiffness = np.maximum(np.asarray(stiffness, dtype=float).reshape(6), 0.0)
        self.leak = max(float(leak), 0.0)
        self.max_offset = np.maximum(np.asarray(max_offset, dtype=float).reshape(6), 1e-9)
        self.axis_mask = np.asarray(axis_mask, dtype=float).reshape(6)
        self.stiction = np.maximum(np.asarray(stiction, dtype=float).reshape(6), 0.0)
        self.wrench_sign = 1.0 if float(wrench_sign) >= 0.0 else -1.0
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in ("offset", "pose_error"):
            raise ValueError("SE3PoseAdmittance6D output_mode must be 'offset' or 'pose_error'")
        self.offset = np.zeros(6, dtype=float)
        self.velocity = np.zeros(6, dtype=float)

    def reset(self) -> None:
        self.offset.fill(0.0)
        self.velocity.fill(0.0)

    def step(
        self,
        pos_ref: np.ndarray,
        rot_ref: np.ndarray,
        pos_current: np.ndarray,
        rot_current: np.ndarray,
        wrench_base: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt_s = max(float(dt), 0.0)
        pose_error = _pose_error_base(pos_current, rot_current, pos_ref, rot_ref)
        pose_error *= self.axis_mask

        wrench = self.wrench_sign * np.asarray(wrench_base, dtype=float).reshape(6)
        wrench *= self.axis_mask

        if self.output_mode == "pose_error":
            spring_state = pose_error
            wrench_all = wrench - self.stiffness * spring_state - self.damping * self.velocity
            wrench_all *= self.axis_mask
            wrench_all = np.where(
                np.abs(wrench_all) > self.stiction,
                wrench_all - np.sign(wrench_all) * self.stiction,
                0.0,
            )
            ddx = wrench_all / self.mass
            self.velocity = (1.0 - self.leak * dt_s) * (self.velocity + ddx * dt_s)
            self.velocity *= self.axis_mask
            command_error = pose_error + self.velocity * dt_s
            command_error = np.clip(command_error, -self.max_offset, self.max_offset)
            command_error *= self.axis_mask
            if dt_s > 0.0:
                self.velocity = (command_error - pose_error) / dt_s
                self.velocity *= self.axis_mask
            return command_error.copy(), pose_error.copy()

        wrench_all = wrench - self.stiffness * self.offset - self.damping * self.velocity
        wrench_all *= self.axis_mask
        wrench_all = np.where(
            np.abs(wrench_all) > self.stiction,
            wrench_all - np.sign(wrench_all) * self.stiction,
            0.0,
        )
        ddx = wrench_all / self.mass
        self.velocity = (1.0 - self.leak * dt_s) * (self.velocity + ddx * dt_s)
        self.velocity *= self.axis_mask
        self.offset = self.offset + self.velocity * dt_s
        self.offset = np.clip(self.offset, -self.max_offset, self.max_offset)
        self.offset *= self.axis_mask
        return self.offset.copy(), pose_error.copy()


class ForceControlStyleAdmittance6D:
    """Force-control-style HFVC admittance that outputs a 6D task offset.

    This mirrors the control role of the original CR-DAgger hardware wrapper:
    a virtual Cartesian mass-damper-spring centered on the policy reference.
    It intentionally stops at a task-space offset because this script still
    logs and commands leader corrections as joint-space delta_q via DLS.
    """

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
        leak: float,
        max_offset: np.ndarray,
        axis_mask: np.ndarray,
        force_dims: int,
        stiction: np.ndarray,
        wrench_sign: float,
    ):
        self.mass = np.maximum(np.asarray(mass, dtype=float).reshape(6), 1e-6)
        self.damping = np.maximum(np.asarray(damping, dtype=float).reshape(6), 0.0)
        self.stiffness = np.maximum(np.asarray(stiffness, dtype=float).reshape(6), 0.0)
        self.leak = max(float(leak), 0.0)
        self.max_offset = np.maximum(np.asarray(max_offset, dtype=float).reshape(6), 1e-9)
        self.axis_mask = np.asarray(axis_mask, dtype=float).reshape(6)
        self.stiction = np.maximum(np.asarray(stiction, dtype=float).reshape(6), 0.0)
        self.wrench_sign = 1.0 if float(wrench_sign) >= 0.0 else -1.0

        n_force = int(np.clip(int(force_dims), 0, 6))
        selection = np.zeros(6, dtype=float)
        selection[:n_force] = 1.0
        selection *= self.axis_mask
        self.force_selection = selection

        self.offset = np.zeros(6, dtype=float)
        self.velocity = np.zeros(6, dtype=float)

    def step(self, wrench_base: np.ndarray, dt: float) -> np.ndarray:
        dt_s = max(float(dt), 0.0)
        wrench = self.wrench_sign * np.asarray(wrench_base, dtype=float).reshape(6)
        wrench *= self.force_selection

        spring = self.stiffness * self.offset
        damping = self.damping * self.velocity
        wrench_all = self.force_selection * (wrench - spring - damping)
        wrench_all = np.where(
            np.abs(wrench_all) > self.stiction,
            wrench_all - np.sign(wrench_all) * self.stiction,
            0.0,
        )

        ddx = wrench_all / self.mass
        self.velocity = (1.0 - self.leak * dt_s) * (self.velocity + ddx * dt_s)
        self.velocity *= self.force_selection
        self.offset = self.offset + self.velocity * dt_s
        self.offset *= self.force_selection
        self.offset = np.clip(self.offset, -self.max_offset, self.max_offset)
        return self.offset.copy()


def _damped_least_squares_delta_q(jacobian: np.ndarray, task_delta: np.ndarray, damping: float) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    delta = np.asarray(task_delta, dtype=float).reshape(6)
    lam2 = max(float(damping), 0.0) ** 2
    lhs = jac @ jac.T + lam2 * np.eye(6)
    return jac.T @ np.linalg.solve(lhs, delta)


def _bota_contact_probability(wrench_base: np.ndarray, force_scale: float, torque_scale: float) -> tuple[float, float]:
    force_norm = float(np.linalg.norm(wrench_base[:3])) / max(float(force_scale), 1e-6)
    torque_norm = float(np.linalg.norm(wrench_base[3:])) / max(float(torque_scale), 1e-6)
    probability = float(np.clip(max(force_norm, torque_norm), 0.0, 1.0))
    if probability >= 0.6:
        return probability, ContactGate.CORRECTION_ACTIVE
    if probability >= 0.25:
        return probability, ContactGate.POSSIBLE_CONTACT
    return probability, ContactGate.NO_CONTACT


def _shi_torque_components(
    system: FACTRGravityCompensation,
    shi: Optional[MinimalistTorqueEstimator],
    q: np.ndarray,
    dq: np.ndarray,
    currents_arm: Optional[np.ndarray],
    tau_model: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(system.num_arm_joints)
    tau_ext = np.zeros(n, dtype=float)
    tau_meas = None

    if shi is not None and currents_arm is not None:
        try:
            tau_meas = shi.get_raw_motor_torque(dq, currents_arm)
            tau_ext = shi.update(q, dq, currents_arm)
        except Exception:
            tau_ext = np.zeros(n, dtype=float)
            tau_meas = None

    components = system.separate_contact_torque_components(
        tau_model=tau_model,
        tau_ext_shi=tau_ext,
        tau_meas=tau_meas,
    )
    return tau_ext, components


def _start_prepared_teleop(system: FACTRGravityCompensation) -> bool:
    if not system.teleop_enabled or not system.teleop_prepared:
        return False

    if system.teleop_thread is not None and system.teleop_thread.is_alive():
        return True

    system.running = True
    if getattr(system, "use_impedance_control", False):
        target = system._teleop_loop_impedance
        mode = "impedance"
    else:
        target = system._teleop_loop
        mode = "position"

    system.teleop_thread = Thread(target=target, daemon=True, name="factr-teleop")
    system.teleop_thread.start()
    print(f"[TELEOP] started follower mirroring thread ({mode} mode)")
    return True


def _map_leader_delta_to_follower_delta(
    system: FACTRGravityCompensation,
    delta_leader: np.ndarray,
    n: int,
) -> np.ndarray:
    delta = np.asarray(delta_leader[:n], dtype=float)
    if system.map_index is None or system.map_signs is None:
        return delta.copy()

    lim = min(len(system.map_index), len(system.map_signs), n)
    delta_follower = np.zeros(n, dtype=float)
    for follower_idx in range(lim):
        leader_idx = int(system.map_index[follower_idx])
        if 0 <= leader_idx < len(delta):
            delta_follower[follower_idx] = float(system.map_signs[follower_idx]) * delta[leader_idx]
    return delta_follower


def _start_direct_follower_command_thread(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    initial_target: np.ndarray,
) -> tuple[threading.Event, threading.Lock, dict[str, np.ndarray], Thread]:
    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("Follower command thread requires teleop.enable=true with direct RTDE")

    follower = system._direct_follower_robot
    stop_event = threading.Event()
    state_lock = threading.Lock()
    state: dict[str, np.ndarray] = {"q": np.asarray(initial_target, dtype=float).copy()}

    try:
        q_follower_now, _ = system.get_follower_arm_state()
        q_follower_now = np.asarray(q_follower_now[: len(state["q"])], dtype=float)
        q_error = state["q"] - q_follower_now
        print(
            "[FOLLOWER] initial command error "
            f"max={float(np.max(np.abs(q_error))):.4f}rad "
            f"mean={float(np.mean(np.abs(q_error))):.4f}rad"
        )
    except Exception as exc:
        print(f"[FOLLOWER] initial command error unavailable: {exc}")

    def _worker() -> None:
        dt_thread = 1.0 / 500.0
        failures = 0
        last_warn = 0.0
        while not stop_event.is_set():
            loop_t0 = time.perf_counter()
            with state_lock:
                q_target = state["q"].copy()
            try:
                ok = follower.command_joint_state_impedance(
                    target_joints=q_target,
                    target_velocities=None,
                    kp=float(args.follower_kp),
                    kd=float(args.follower_kd),
                )
                if ok is False:
                    failures += 1
                    now = time.monotonic()
                    if now - last_warn > 1.0:
                        last_warn = now
                        print(f"[FOLLOWER] directTorque returned False ({failures} total)")
            except Exception as exc:
                failures += 1
                now = time.monotonic()
                if now - last_warn > 1.0:
                    last_warn = now
                    print(f"[FOLLOWER] command failed ({failures} total): {exc}")

            sleep_time = dt_thread - (time.perf_counter() - loop_t0)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    thread = Thread(
        target=_worker,
        daemon=True,
        name="leader-admittance-position-follower-command",
    )
    thread.start()
    print("[FOLLOWER] direct command thread started (500Hz, source=q_ref+delta)")
    return stop_event, state_lock, state, thread


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Epsilon tracking test (Phase B style)")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V3_PhaseB.yaml")
    p.add_argument("--duration", type=float, default=30.0, help="0 means run until Ctrl+C")
    p.add_argument("--settle-time", type=float, default=2.0)
    p.add_argument(
        "--impedance-ramp-time",
        type=float,
        default=2.0,
        help="Seconds to ramp leader policy impedance from 0 to configured gains after settle.",
    )
    p.add_argument("--max-duration", type=float, default=120.0)

    p.add_argument("--target", choices=["leader", "follower"], default="leader")
    p.add_argument(
        "--mode",
        choices=[
            "leader_tracking",
            "leader_observer",
            "leader_admittance",
            "leader_admittance_impedance",
            "leader_admittance_position",
            "four_channel",
            "follower_tracking",
            "observer_only",
        ],
        default=None,
        help="Architecture mode. If omitted, legacy --target/--four-channel flags are used.",
    )
    p.add_argument(
        "--test-mode",
        choices=["static_hold", "sine", "chirp", "ramp", "multi_joint", "taskspace"],
        default="sine",
    )

    p.add_argument(
        "--four-channel",
        action="store_true",
        help="Use 4-channel dual-impedance loop (epsilon = q_ur5e - q_cmd).",
    )
    p.add_argument(
        "--mirror-source",
        choices=["cmd", "actual", "policy", "blend"],
        default=None,
        help="Override 4-channel mirror source (default: config).",
    )
    p.add_argument("--mirror-blend-cmd", type=float, default=0.5)
    p.add_argument("--mirror-blend-actual", type=float, default=0.3)
    p.add_argument("--mirror-blend-policy", type=float, default=0.2)

    p.add_argument("--compensation", choices=["none", "velocity_ff", "feedforward", "computed_torque"], default="none")
    p.add_argument(
        "--kp",
        type=str,
        default=None,
        help="Leader policy Kp as scalar or comma-separated per-joint list. Default: config.",
    )
    p.add_argument(
        "--kd",
        type=str,
        default=None,
        help="Leader policy Kd as scalar or comma-separated per-joint list. Default: config.",
    )

    p.add_argument("--adm-mass", type=float, default=1.0)
    p.add_argument("--adm-damp", type=float, default=8.0)
    p.add_argument("--adm-stiff", type=float, default=25.0)
    p.add_argument("--adm-leak", type=float, default=0.02)
    p.add_argument("--adm-delta-max", type=float, default=0.25)
    p.add_argument(
        "--admittance-source",
        choices=["observer", "bota_cartesian", "bota_hfvc", "bota_se3", "bota_joint"],
        default="observer",
        help="Signal used to drive leader_admittance_position. Bota sources also log the other projection.",
    )
    p.add_argument("--bota-enable", action="store_true", help="Open MiniOne and record conditioned wrench signals.")
    p.add_argument("--bota-config", type=str, default="configs/bota_binary.json")
    p.add_argument("--bota-driver-tare", action="store_true", help="Call bota_driver.tare() before activation.")
    p.add_argument("--bota-warmup-seconds", type=float, default=0.0, help="Discard Bota frames for this long after activation before bias/logging/admittance.")
    p.add_argument("--bota-bias-seconds", type=float, default=1.0, help="Software bias collection duration after activation/warmup. Set 0 to disable.")
    p.add_argument("--bota-filter-cutoff", type=float, default=20.0, help="EMA low-pass cutoff for base-frame wrench. Set <=0 to use --bota-filter-alpha or disable filtering.")
    p.add_argument("--bota-filter-alpha", type=float, default=0.0, help="Fixed EMA alpha in (0, 1]; used only when --bota-filter-cutoff <= 0.")
    p.add_argument("--bota-deadband", type=str, default="0.25,0.25,0.25,0.01,0.01,0.01")
    p.add_argument("--bota-saturation", type=str, default="25,25,25,1.5,1.5,1.5")
    p.add_argument("--bota-gravity-comp", action="store_true", help="Subtract modeled payload gravity from the MiniOne wrench.")
    p.add_argument("--bota-payload-mass", type=float, default=0.070, help="Distal payload mass seen by the sensor [kg].")
    p.add_argument("--bota-payload-com", type=str, default="0,0,0.0146", help="Payload COM in sensor frame [m].")
    p.add_argument("--bota-gravity-sign", type=float, default=1.0, help="Flip to -1 if static gravity compensation has wrong sign.")
    p.add_argument("--bota-frame", type=str, default="", help="Pinocchio end-effector frame. Empty uses the last URDF frame.")
    p.add_argument("--bota-axis-mask", type=str, default="1,1,1,1,1,1", help="6D mask for Cartesian admittance axes.")
    p.add_argument("--bota-cart-mass", type=str, default="4,4,4,0.25,0.25,0.25")
    p.add_argument("--bota-cart-damp", type=str, default="35,35,35,1.5,1.5,1.5")
    p.add_argument("--bota-cart-stiff", type=str, default="70,70,70,4,4,4")
    p.add_argument("--bota-cart-max", type=str, default="0.08,0.08,0.08,0.35,0.35,0.35")
    p.add_argument("--bota-hfvc-force-dims", type=int, default=6, help="Number of force-controlled task axes for bota_hfvc, starting from x,y,z,rx,ry,rz.")
    p.add_argument("--bota-hfvc-stiction", type=str, default="0,0,0,0,0,0", help="6D wrench deadzone inside the HFVC admittance after BOTA conditioning.")
    p.add_argument("--bota-wrench-sign", type=float, default=1.0, help="Set to -1 if bota_hfvc/bota_se3 moves opposite to the intended push direction.")
    p.add_argument(
        "--bota-se3-output-mode",
        choices=["offset", "pose_error"],
        default="offset",
        help="offset: no-force SE3 admittance decays to zero; pose_error: legacy mode that also follows leader sag.",
    )
    p.add_argument("--bota-joint-gain", type=float, default=1.0, help="Gain for J.T @ wrench before joint-space admittance.")
    p.add_argument("--bota-dls-damping", type=float, default=0.03, help="Damped least-squares factor for Cartesian delta_q.")
    p.add_argument(
        "--leader-position-operating-mode",
        type=int,
        choices=[3, 5],
        default=3,
        help="DYNAMIXEL mode used by leader_admittance_position: 3=position, 5=current-based position.",
    )
    p.add_argument(
        "--leader-position-handoff-settle-seconds",
        type=float,
        default=0.35,
        help="Seconds to let DYNAMIXEL position mode settle before rebasing the held pose.",
    )
    p.add_argument(
        "--leader-position-arm-settle-seconds",
        type=float,
        default=0.5,
        help="Unrecorded settle time after handoff/follower arming before the test clock starts.",
    )
    p.add_argument(
        "--leader-position-recover-pre-handoff",
        action="store_true",
        help="After position-mode handoff, recover the leader to the pre-handoff pose before follower/recording are armed.",
    )
    p.add_argument(
        "--leader-position-recover-seconds",
        type=float,
        default=0.8,
        help="Duration for optional unrecorded recovery to the pre-handoff pose.",
    )
    p.add_argument(
        "--leader-position-recover-max-delta",
        type=float,
        default=0.35,
        help="Skip optional pre-handoff recovery if any wrapped joint delta exceeds this value; set <=0 to disable limit.",
    )
    p.add_argument("--leader-current-limit", type=str, default="", help="Raw Current Limit(38) as scalar or comma list. Empty leaves device value unchanged.")
    p.add_argument("--leader-goal-current", type=str, default="", help="Raw Goal Current(102) as scalar or comma list. Used for current-based position tests.")
    p.add_argument("--leader-position-p-gains", type=str, default="", help="Position P Gain(84) as scalar or comma list after operating-mode switch.")
    p.add_argument("--leader-position-i-gains", type=str, default="", help="Position I Gain(82) as scalar or comma list after operating-mode switch.")
    p.add_argument("--leader-position-d-gains", type=str, default="", help="Position D Gain(80) as scalar or comma list after operating-mode switch.")
    p.add_argument("--leader-velocity-p-gains", type=str, default="", help="Velocity P Gain(78) as scalar or comma list after operating-mode switch.")
    p.add_argument("--leader-velocity-i-gains", type=str, default="", help="Velocity I Gain(76) as scalar or comma list after operating-mode switch.")
    p.add_argument("--bota-contact-force-scale", type=float, default=8.0)
    p.add_argument("--bota-contact-torque-scale", type=float, default=0.35)
    p.add_argument(
        "--admittance-observer",
        choices=["shi", "yamane", "none"],
        default="shi",
        help=(
            "External torque source for leader admittance/observer modes. "
            "Yamane is only valid when the leader is torque-commanded."
        ),
    )

    p.add_argument("--follower-kp", type=float, default=150.0)
    p.add_argument("--follower-kd", type=float, default=12.0)

    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--frequency", type=float, default=0.2)
    p.add_argument("--joint-index", type=int, default=1)

    p.add_argument("--chirp-f-start", type=float, default=0.1)
    p.add_argument("--chirp-f-end", type=float, default=2.0)
    p.add_argument("--chirp-duration", type=float, default=30.0)

    p.add_argument("--ramp-delta", type=float, default=0.2)
    p.add_argument("--ramp-duration", type=float, default=5.0)

    p.add_argument("--multi-amplitudes", type=str, default=None)
    p.add_argument("--multi-frequencies", type=str, default=None)

    p.add_argument("--taskspace-motion", choices=["line", "circle", "figure8"], default="circle")
    p.add_argument("--taskspace-amplitude", type=float, default=0.08, help="meters")
    p.add_argument("--taskspace-axis", choices=["xy", "xz", "yz"], default="xy")

    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-dt", type=float, default=0.1)

    p.add_argument("--log-dir", type=str, default="epsilon_measurements")
    p.add_argument("--no-follower", action="store_true", help="Leader mode: do not start follower teleop")
    p.add_argument("--no-shi", action="store_true", help="Skip Shi torque estimator (records zeros)")

    return p.parse_args()


def _resolve_arch_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        if args.mode == "observer_only":
            print(
                "[WARN] observer_only is deprecated; using leader_observer instead"
            )
            return "leader_observer"
        if args.mode == "leader_admittance":
            print(
                "[WARN] leader_admittance is an alias for "
                "leader_admittance_impedance. Use leader_admittance_position "
                "for true position-mode admittance."
            )
            return "leader_admittance_impedance"
        return str(args.mode)
    if args.four_channel:
        return "four_channel"
    if args.target == "follower":
        return "follower_tracking"
    return "leader_tracking"


def _leader_position_local_reference(
    args: argparse.Namespace,
    n: int,
    center: np.ndarray,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    mode = str(args.test_mode)
    q_ref = np.asarray(center[:n], dtype=float).copy()
    dq_ref = np.zeros(n, dtype=float)
    t = max(float(elapsed_s), 0.0)

    if mode == "static_hold":
        return q_ref, dq_ref

    if mode == "sine":
        amplitudes = np.zeros(n, dtype=float)
        frequencies = np.zeros(n, dtype=float)
        if 0 <= int(args.joint_index) < n:
            amplitudes[int(args.joint_index)] = float(args.amplitude)
            frequencies[int(args.joint_index)] = float(args.frequency)
        else:
            amplitudes[:] = float(args.amplitude)
            frequencies[:] = float(args.frequency)
        phase = 2.0 * np.pi * frequencies * t
        q_ref = q_ref + amplitudes * np.sin(phase)
        dq_ref = amplitudes * 2.0 * np.pi * frequencies * np.cos(phase)
        return q_ref, dq_ref

    if mode == "multi_joint":
        amplitudes = _expand_list(_parse_float_list(args.multi_amplitudes), n, 0.0)
        frequencies = _expand_list(_parse_float_list(args.multi_frequencies), n, 0.2)
        phase = 2.0 * np.pi * frequencies * t
        q_ref = q_ref + amplitudes * np.sin(phase)
        dq_ref = amplitudes * 2.0 * np.pi * frequencies * np.cos(phase)
        return q_ref, dq_ref

    if mode == "ramp":
        duration = max(float(args.ramp_duration), 1e-6)
        alpha = float(np.clip(t / duration, 0.0, 1.0))
        q_end = q_ref.copy()
        if 0 <= int(args.joint_index) < n:
            q_end[int(args.joint_index)] += float(args.ramp_delta)
        q_ref = (1.0 - alpha) * q_ref + alpha * q_end
        if alpha < 1.0 and 0 <= int(args.joint_index) < n:
            dq_ref[int(args.joint_index)] = float(args.ramp_delta) / duration
        return q_ref, dq_ref

    return None


def _build_policy(
    args: argparse.Namespace,
    n: int,
    q0: np.ndarray,
    config_path: Path,
    system_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if args.test_mode == "static_hold":
        return "dummy_hold", {"q_hold": q0.tolist()}

    if args.test_mode == "sine":
        amplitudes = np.zeros(n, dtype=float)
        frequencies = np.zeros(n, dtype=float)
        if 0 <= int(args.joint_index) < n:
            amplitudes[int(args.joint_index)] = float(args.amplitude)
            frequencies[int(args.joint_index)] = float(args.frequency)
        else:
            amplitudes[:] = float(args.amplitude)
            frequencies[:] = float(args.frequency)
        return "dummy_sine", {
            "center": q0.tolist(),
            "amplitude": amplitudes.tolist(),
            "frequency": frequencies.tolist(),
        }

    if args.test_mode == "chirp":
        return "dummy_chirp", {
            "center": q0.tolist(),
            "amplitude": float(args.amplitude),
            "f_start": float(args.chirp_f_start),
            "f_end": float(args.chirp_f_end),
            "duration": float(args.chirp_duration),
            "joint_index": int(args.joint_index),
        }

    if args.test_mode == "ramp":
        q_end = q0.copy()
        if 0 <= int(args.joint_index) < n:
            q_end[int(args.joint_index)] += float(args.ramp_delta)
        return "dummy_ramp", {
            "q_start": q0.tolist(),
            "q_end": q_end.tolist(),
            "duration": float(args.ramp_duration),
        }

    if args.test_mode == "multi_joint":
        amplitudes = _expand_list(_parse_float_list(args.multi_amplitudes), n, 0.0)
        frequencies = _expand_list(_parse_float_list(args.multi_frequencies), n, 0.2)
        return "dummy_multi_sine", {
            "center": q0.tolist(),
            "amplitudes": amplitudes.tolist(),
            "frequencies": frequencies.tolist(),
        }

    if args.test_mode == "taskspace":
        leader_urdf = str(system_config["arm_teleop"]["leader_urdf"])
        urdf_path = _resolve_leader_urdf(config_path, leader_urdf)
        return "dummy_taskspace", {
            "center": q0.tolist(),
            "urdf_path": str(urdf_path),
            "motion_type": str(args.taskspace_motion),
            "amplitude": float(args.taskspace_amplitude),
            "frequency": float(args.frequency),
            "axis": str(args.taskspace_axis),
        }

    raise ValueError(f"Unknown test_mode: {args.test_mode}")


def _sleep_remaining(dt: float, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    sleep_s = dt - elapsed
    if sleep_s > 0:
        time.sleep(sleep_s)


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _recover_leader_position_after_handoff(
    system: FACTRGravityCompensation,
    target_raw_pre_handoff: np.ndarray,
    current_raw: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    n = int(system.num_arm_joints)
    duration = max(float(getattr(args, "leader_position_recover_seconds", 0.0)), 0.0)
    max_delta = max(float(getattr(args, "leader_position_recover_max_delta", 0.0)), 0.0)
    if duration <= 0.0:
        return np.asarray(current_raw, dtype=float).copy()

    target_raw_pre_handoff = np.asarray(target_raw_pre_handoff, dtype=float)
    current_raw = np.asarray(current_raw, dtype=float)
    if target_raw_pre_handoff.shape[0] != current_raw.shape[0]:
        print("[DXL] pre-handoff recovery skipped: target/current length mismatch")
        return current_raw.copy()

    delta_raw = np.zeros_like(current_raw)
    delta_raw[:n] = _wrap_angle_delta(target_raw_pre_handoff[:n] - current_raw[:n])
    max_abs = float(np.max(np.abs(delta_raw[:n]))) if n > 0 else 0.0
    mean_abs = float(np.mean(np.abs(delta_raw[:n]))) if n > 0 else 0.0
    if max_delta > 0.0 and max_abs > max_delta:
        print(
            "[DXL] pre-handoff recovery skipped: "
            f"target delta max={max_abs:.4f}rad exceeds limit {max_delta:.4f}rad"
        )
        return current_raw.copy()

    target_raw = current_raw + delta_raw
    if system.num_motors > n:
        target_raw[n:] = current_raw[n:]

    print(
        "[DXL] recovering leader to pre-handoff pose "
        f"over {duration:.2f}s max={max_abs:.4f}rad mean={mean_abs:.4f}rad"
    )
    steps = max(1, int(np.ceil(duration / max(float(system.dt), 1e-3))))
    start_raw = current_raw.copy()
    for step in range(1, steps + 1):
        alpha = step / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        cmd = start_raw + alpha * (target_raw - start_raw)
        system.driver.set_joints(cmd.tolist())
        time.sleep(max(float(system.dt), 0.001))

    raw_after = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_after.shape[0] != current_raw.shape[0]:
        raw_after = target_raw
    system.driver.set_joints(raw_after.tolist())
    return raw_after


def _leader_gravity_hold(system: FACTRGravityCompensation, dt: float) -> np.ndarray:
    q, dq, _, _ = system.get_leader_joint_states()
    tau_grav = system.gravity_compensation(q, dq)
    tau_fric = system.friction_compensation(dq)
    tau_damp = -float(system.gravity_comp_velocity_damping) * dq
    tau_hold = tau_grav + tau_fric + tau_damp
    system.set_leader_joint_torque(tau_hold, 0.0)
    time.sleep(max(0.0, dt - 0.0005))
    return np.asarray(tau_hold, dtype=float)


def _readback_2byte(driver: Any, address: int, signed: bool = False) -> list[int]:
    if not hasattr(driver, "read_control_table_2byte"):
        return []
    try:
        return [int(v) for v in driver.read_control_table_2byte(address, signed=signed).tolist()]
    except Exception as exc:
        print(f"[DXL] readback failed for register {address}: {exc}")
        return []


def _switch_leader_to_position_hold(
    system: FACTRGravityCompensation, args: argparse.Namespace
) -> np.ndarray:
    """Enter Dynamixel position mode and return the pose actually held after handoff."""
    if system.driver is None:
        return np.zeros(int(system.num_motors), dtype=float)

    raw_pos_now = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_now.shape[0] != int(system.num_motors):
        raise RuntimeError(
            "Unexpected Dynamixel state length during position-mode handoff "
            f"(got {raw_pos_now.shape[0]}, expected {system.num_motors})"
        )

    num_motors = int(system.num_motors)
    operating_mode = int(args.leader_position_operating_mode)
    if operating_mode not in (3, 5):
        raise ValueError("--leader-position-operating-mode must be 3 or 5")

    current_limit = _parse_optional_int_array(args.leader_current_limit, num_motors)
    goal_current = _parse_optional_int_array(args.leader_goal_current, num_motors)
    pos_p = _parse_optional_int_array(args.leader_position_p_gains, num_motors)
    pos_i = _parse_optional_int_array(args.leader_position_i_gains, num_motors)
    pos_d = _parse_optional_int_array(args.leader_position_d_gains, num_motors)
    vel_p = _parse_optional_int_array(args.leader_velocity_p_gains, num_motors)
    vel_i = _parse_optional_int_array(args.leader_velocity_i_gains, num_motors)

    setattr(system, "_leader_position_pre_handoff_raw", raw_pos_now.copy())

    system.driver.set_torque_mode(False)
    time.sleep(0.02)

    restore: dict[str, list[int]] = {}
    if current_limit is not None:
        previous = _readback_2byte(system.driver, ADDR_CURRENT_LIMIT)
        if previous:
            restore["current_limit"] = previous

    system.driver.set_operating_mode(operating_mode)
    time.sleep(0.02)
    if hasattr(system.driver, "verify_operating_mode"):
        system.driver.verify_operating_mode(operating_mode)

    if current_limit is not None:
        system.driver.set_current_limits(current_limit.tolist())
        time.sleep(0.01)

    if operating_mode == 5 and goal_current is None and current_limit is not None:
        goal_current = current_limit.copy()
    if goal_current is not None:
        system.driver.set_goal_currents_raw(goal_current.tolist())
        time.sleep(0.01)

    if any(v is not None for v in (pos_p, pos_i, pos_d)):
        system.driver.set_position_pid_gains(
            p_gains=None if pos_p is None else pos_p.tolist(),
            i_gains=None if pos_i is None else pos_i.tolist(),
            d_gains=None if pos_d is None else pos_d.tolist(),
        )
        time.sleep(0.01)

    if any(v is not None for v in (vel_p, vel_i)):
        system.driver.set_velocity_pi_gains(
            p_gains=None if vel_p is None else vel_p.tolist(),
            i_gains=None if vel_i is None else vel_i.tolist(),
        )
        time.sleep(0.01)

    setattr(system, "_leader_position_handoff_restore", restore)

    print(f"[DXL] leader position handoff operating_mode={operating_mode}")
    readbacks = {
        "current_limit": _readback_2byte(system.driver, ADDR_CURRENT_LIMIT),
        "goal_current": _readback_2byte(system.driver, ADDR_GOAL_CURRENT, signed=True),
        "pos_p": _readback_2byte(system.driver, ADDR_POSITION_P_GAIN),
        "pos_i": _readback_2byte(system.driver, ADDR_POSITION_I_GAIN),
        "pos_d": _readback_2byte(system.driver, ADDR_POSITION_D_GAIN),
        "vel_p": _readback_2byte(system.driver, ADDR_VELOCITY_P_GAIN),
        "vel_i": _readback_2byte(system.driver, ADDR_VELOCITY_I_GAIN),
    }
    for name, values in readbacks.items():
        if values:
            print(f"[DXL] {name}: {values}")

    raw_pos_post_mode = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_post_mode.shape[0] != int(system.num_motors):
        raw_pos_post_mode = raw_pos_now
    handoff_motion = _wrap_angle_delta(raw_pos_post_mode[: system.num_arm_joints] - raw_pos_now[: system.num_arm_joints])
    print(
        "[DXL] handoff pre-torque drift "
        f"max={float(np.max(np.abs(handoff_motion))):.4f}rad "
        f"mean={float(np.mean(np.abs(handoff_motion))):.4f}rad"
    )

    # Preload the settled/current position as Goal Position before torque is
    # re-enabled. Otherwise the position controller first pulls back to the
    # pre-mode-switch pose, which looks like a synthetic intervention.
    if hasattr(system.driver, "write_goal_positions_unchecked"):
        system.driver.write_goal_positions_unchecked(raw_pos_post_mode.tolist())
    else:
        system.driver.set_torque_mode(True)
        time.sleep(0.005)
        system.driver.set_joints(raw_pos_post_mode.tolist())
        time.sleep(0.005)
        raw_pos_post_mode = np.asarray(system.driver.get_joints(), dtype=float)
        if raw_pos_post_mode.shape[0] != int(system.num_motors):
            raw_pos_post_mode = raw_pos_now
        system.driver.set_torque_mode(False)
        time.sleep(0.005)
    time.sleep(0.005)
    system.driver.set_torque_mode(True)
    time.sleep(0.02)

    settle_s = max(float(getattr(args, "leader_position_handoff_settle_seconds", 0.0)), 0.0)
    if settle_s > 0.0:
        print(
            f"[DXL] settling position hold for {settle_s:.2f}s before rebase "
            "(hands off unless safety requires support)"
        )
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < settle_s:
            time.sleep(min(0.02, max(0.0, settle_s - (time.perf_counter() - t_settle))))

    # Rebase again after the position controller has settled so static tests do
    # not carry residual startup error from the mode switch itself.
    raw_pos_after = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_after.shape[0] != int(system.num_motors):
        return raw_pos_post_mode
    post_enable_motion = _wrap_angle_delta(raw_pos_after[: system.num_arm_joints] - raw_pos_post_mode[: system.num_arm_joints])
    print(
        "[DXL] handoff post-enable drift "
        f"max={float(np.max(np.abs(post_enable_motion))):.4f}rad "
        f"mean={float(np.mean(np.abs(post_enable_motion))):.4f}rad"
    )
    system.driver.set_joints(raw_pos_after.tolist())
    time.sleep(0.02)
    return raw_pos_after


def _prime_policy_impedance_from_hold(system: FACTRGravityCompensation) -> np.ndarray:
    q, dq, _, _ = system.get_leader_joint_states()
    tau_hold = _leader_gravity_hold(system, float(system.dt))
    n = int(system.num_arm_joints)
    system._dq_filt = np.asarray(dq[:n], dtype=float).copy()
    system._tau_cmd_prev = np.asarray(tau_hold[:n], dtype=float).copy()
    system._tau_bias = np.zeros(n, dtype=float)
    system._impedance_initialized = True
    return np.asarray(q[:n], dtype=float).copy()


def _impedance_ramp_scale(args: argparse.Namespace, elapsed_s: float) -> float:
    ramp_time = max(float(args.impedance_ramp_time), 0.0)
    if ramp_time <= 1e-9:
        return 1.0
    x = float(np.clip(float(elapsed_s) / ramp_time, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _run_leader_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
    shi: Optional[MinimalistTorqueEstimator],
) -> None:
    from gello.cr_dagger.core.yamane_observer import GeneralizedMomentumObserver

    n = system.num_arm_joints
    dt = float(system.dt)
    observer_mode = (
        str(args.admittance_observer).lower()
        if str(getattr(args, "mode", "")).lower() in ("leader_observer", "observer_only")
        else "shi"
    )
    yamane: GeneralizedMomentumObserver | None = None
    if observer_mode == "yamane":
        four_cfg = system.config.get("teleop", {}).get("four_channel", {})
        yamane = GeneralizedMomentumObserver(
            pin_model=system.pin_model,
            num_arm_joints=n,
            K_obs=float(four_cfg.get("yamane_K_obs", 50.0)),
            dt=dt,
            filter_fc=float(four_cfg.get("yamane_filter_fc", 5.0)),
        )
        q_init, dq_init, _, _ = system.get_leader_joint_states()
        yamane.reset(
            np.asarray(q_init[:n], dtype=float),
            np.asarray(dq_init[:n], dtype=float),
        )
        print("[OBSERVER] leader tracking observer=yamane")
    elif observer_mode == "none":
        print("[OBSERVER] leader tracking observer=none")
    else:
        print("[OBSERVER] leader tracking observer=shi")

    if not args.no_follower:
        if not _start_prepared_teleop(system):
            print("[WARN] teleop could not be started; continuing without follower")

    if float(args.settle_time) > 0.0:
        print("[SETTLE] gravity comp warmup")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < float(args.settle_time):
            _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0
    contact_gate = ContactGate(n)

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        currents_arm = None
        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, cur_raw = system.driver.get_positions_velocities_and_currents()
            q = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
            dq = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

            if len(pos_raw) > n:
                grip_raw = float(pos_raw[-1])
                grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
                system.leader_gripper_raw_rad = grip_raw
            else:
                grip = 0.0
                grip_raw = 0.0

            if len(vel_raw) > n:
                grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
            else:
                grip_vel = 0.0

            system.gripper_pos = float(grip)
            system._last_gripper_vel = float(grip_vel)

            if observer_mode == "shi" and shi is not None:
                currents_arm = np.asarray(cur_raw[:n], dtype=float) * system.joint_signs[:n]
        else:
            q, dq, grip, grip_vel = system.get_leader_joint_states()
            if observer_mode == "shi" and shi is not None and system.driver is not None:
                currents = system.driver.get_currents()
                currents_arm = np.asarray(currents[:n], dtype=float) * system.joint_signs[:n]

        q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref, t_mono)
        system.policy_impedance_scale = _impedance_ramp_scale(args, t_mono - t_start)

        tau_cmd = system.control_loop_step_with_policy(
            q_ref_policy=q_ref,
            dq_ref_policy=dq_ref,
            ddq_ref_policy=ddq_ref,
            compensation_mode=str(args.compensation),
            leader_state=(q, dq, float(grip), float(grip_vel)),
        )

        if observer_mode == "yamane" and yamane is not None:
            tau_ext = yamane.update(q=q, dq=dq, tau_cmd=tau_cmd, dt=dt)
            tau_components = system.separate_contact_torque_components(
                tau_model=tau_cmd,
                tau_ext_shi=tau_ext,
            )
        elif observer_mode == "none":
            tau_ext = np.zeros(n, dtype=float)
            tau_components = system.separate_contact_torque_components(
                tau_model=tau_cmd,
                tau_ext_shi=tau_ext,
            )
        else:
            tau_ext, tau_components = _shi_torque_components(
                system=system,
                shi=shi,
                q=q,
                dq=dq,
                currents_arm=currents_arm,
                tau_model=tau_cmd,
            )
        contact_probability, contact_state = contact_gate.update(
            tau_components["tau_residual"],
            ddq_ref,
        )

        q_follower = np.zeros(n)
        dq_follower = np.zeros(n)
        if system.teleop_enabled:
            try:
                q_follower, dq_follower = system.get_follower_arm_state()
            except Exception:
                pass

        epsilon = q - q_ref
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_ref=ddq_ref,
            q_leader=q,
            dq_leader=dq,
            q_cmd_leader=q_ref,
            q_c_leader=q_ref,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=np.full(n, np.nan),
            delta_corr=np.zeros(n),
            tau_cmd=tau_cmd,
            tau_ext_shi=tau_ext,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=contact_probability,
            contact_state=contact_state,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q,
            dq=dq,
            grip=float(grip),
            tau_ext=tau_ext,
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)


def _run_follower_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
) -> None:
    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("Follower mode requires teleop.enable=true with direct RTDE")

    follower = system._direct_follower_robot
    n = system.num_arm_joints
    dt = float(system.dt)

    if float(args.settle_time) > 0.0:
        print("[SETTLE] gravity comp warmup")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < float(args.settle_time):
            _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0

    tau_max = np.array([100.0, 100.0, 60.0, 25.0, 25.0, 25.0], dtype=float)

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        q_leader, dq_leader, grip, grip_vel = system.get_leader_joint_states()

        tau_grav = system.gravity_compensation(q_leader, dq_leader)
        tau_fric = system.friction_compensation(dq_leader)
        tau_limit, torque_gripper = system.joint_limit_barrier(
            q_leader, dq_leader, float(grip), float(grip_vel)
        )
        tau_damp = -float(system.gravity_comp_velocity_damping) * dq_leader
        system.set_leader_joint_torque(tau_grav + tau_fric + tau_limit + tau_damp, float(torque_gripper))

        q_ref_leader, dq_ref_leader, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref_leader, t_mono)

        q_ref_follower = system._build_follower_action(q_ref_leader, float(grip))[:n]
        if system.map_signs is not None and system.map_index is not None:
            dq_ref_follower = system.map_signs * dq_ref_leader[system.map_index]
        else:
            dq_ref_follower = dq_ref_leader[:n]

        q_follower, dq_follower = system.get_follower_arm_state()
        q_follower = np.asarray(q_follower[:n], dtype=float)
        dq_follower = np.asarray(dq_follower[:n], dtype=float)

        follower.command_joint_state_impedance(
            target_joints=q_ref_follower,
            target_velocities=dq_ref_follower,
            kp=float(args.follower_kp),
            kd=float(args.follower_kd),
        )

        tau_cmd = float(args.follower_kp) * (q_ref_follower - q_follower) + float(
            args.follower_kd
        ) * (dq_ref_follower - dq_follower)
        tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
        tau_components = system.separate_contact_torque_components(
            tau_model=tau_cmd,
            tau_ext_shi=np.zeros(n),
        )

        epsilon = q_follower - q_ref_follower
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref_follower,
            dq_ref=dq_ref_follower,
            ddq_ref=ddq_ref,
            q_leader=q_leader,
            dq_leader=dq_leader,
            q_cmd_leader=q_ref_leader,
            q_c_leader=q_ref_leader,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=q_ref_follower,
            delta_corr=np.zeros(n),
            tau_cmd=tau_cmd,
            tau_ext_shi=np.zeros(n),
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=0.0,
            contact_state=ContactGate.NO_CONTACT,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q_leader,
            dq=dq_leader,
            grip=float(grip),
            tau_ext=np.zeros(n),
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)

    try:
        follower.robot.stopJ(2.0)
    except Exception:
        pass


def _run_leader_admittance_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
    shi: Optional[MinimalistTorqueEstimator],
) -> None:
    from gello.cr_dagger.core.yamane_observer import GeneralizedMomentumObserver

    n = system.num_arm_joints
    dt = float(system.dt)
    observer_mode = str(args.admittance_observer).lower()

    if not args.no_follower:
        if not _start_prepared_teleop(system):
            print("[WARN] teleop could not be started; continuing without follower")

    if float(args.settle_time) > 0.0:
        print("[SETTLE] gravity comp warmup")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < float(args.settle_time):
            _leader_gravity_hold(system, dt)

    q0, _, _, _ = system.get_leader_joint_states()
    q_c = np.asarray(q0[:n], dtype=float).copy()
    dq_c = np.zeros(n, dtype=float)
    contact_gate = ContactGate(n)
    last_tau_residual = np.zeros(n, dtype=float)
    yamane: GeneralizedMomentumObserver | None = None
    if observer_mode == "yamane":
        four_cfg = system.config.get("teleop", {}).get("four_channel", {})
        yamane = GeneralizedMomentumObserver(
            pin_model=system.pin_model,
            num_arm_joints=n,
            K_obs=float(four_cfg.get("yamane_K_obs", 50.0)),
            dt=dt,
            filter_fc=float(four_cfg.get("yamane_filter_fc", 5.0)),
        )
        _, dq0, _, _ = system.get_leader_joint_states()
        yamane.reset(q0[:n], np.asarray(dq0[:n], dtype=float))
        print("[ADMITTANCE] observer=yamane")
    elif observer_mode == "none":
        print("[ADMITTANCE] observer=none")
    else:
        print("[ADMITTANCE] observer=shi")

    mass = np.full(n, max(float(args.adm_mass), 1e-6), dtype=float)
    damp = np.full(n, max(float(args.adm_damp), 0.0), dtype=float)
    stiff = np.full(n, max(float(args.adm_stiff), 0.0), dtype=float)
    leak = max(float(args.adm_leak), 0.0)
    delta_max = max(float(args.adm_delta_max), 1e-6)

    t_start = time.monotonic()
    last_warn = 0.0

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        currents_arm = None
        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, cur_raw = system.driver.get_positions_velocities_and_currents()
            q = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
            dq = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

            if len(pos_raw) > n:
                grip_raw = float(pos_raw[-1])
                grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
                system.leader_gripper_raw_rad = grip_raw
            else:
                grip = 0.0
                grip_raw = 0.0

            if len(vel_raw) > n:
                grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
            else:
                grip_vel = 0.0

            system.gripper_pos = float(grip)
            system._last_gripper_vel = float(grip_vel)

            if observer_mode == "shi" and shi is not None:
                currents_arm = np.asarray(cur_raw[:n], dtype=float) * system.joint_signs[:n]
        else:
            q, dq, grip, grip_vel = system.get_leader_joint_states()
            if observer_mode == "shi" and shi is not None and system.driver is not None:
                currents = system.driver.get_currents()
                currents_arm = np.asarray(currents[:n], dtype=float) * system.joint_signs[:n]

        q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref, t_mono)

        contact_probability, contact_state = contact_gate.update(
            last_tau_residual,
            ddq_ref,
            correction_active=True,
        )

        tau_contact = last_tau_residual if contact_probability >= 0.6 else np.zeros(n)
        ddq_c = (tau_contact - damp * dq_c - stiff * (q_c - q_ref)) / mass
        dq_c = (1.0 - leak * dt) * (dq_c + ddq_c * dt)
        q_c = q_c + dq_c * dt

        delta = np.clip(q_c - q_ref, -delta_max, delta_max)
        q_c = q_ref + delta
        system.policy_impedance_scale = _impedance_ramp_scale(args, t_mono - t_start)

        tau_cmd = system.control_loop_step_with_policy(
            q_ref_policy=q_c,
            dq_ref_policy=dq_c,
            ddq_ref_policy=ddq_c,
            compensation_mode=str(args.compensation),
            leader_state=(q, dq, float(grip), float(grip_vel)),
        )
        if observer_mode == "yamane" and yamane is not None:
            tau_ext = yamane.update(q=q, dq=dq, tau_cmd=tau_cmd, dt=dt)
            tau_components = system.separate_contact_torque_components(
                tau_model=tau_cmd,
                tau_ext_shi=tau_ext,
            )
        elif observer_mode == "none":
            tau_ext = np.zeros(n, dtype=float)
            tau_components = system.separate_contact_torque_components(
                tau_model=tau_cmd,
                tau_ext_shi=tau_ext,
            )
        else:
            tau_ext, tau_components = _shi_torque_components(
                system=system,
                shi=shi,
                q=q,
                dq=dq,
                currents_arm=currents_arm,
                tau_model=tau_cmd,
            )
        last_tau_residual = tau_components["tau_residual"].copy()

        q_follower = np.zeros(n)
        dq_follower = np.zeros(n)
        if system.teleop_enabled:
            try:
                q_follower, dq_follower = system.get_follower_arm_state()
            except Exception:
                pass

        epsilon = q - q_c
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_ref=ddq_ref,
            q_leader=q,
            dq_leader=dq,
            q_cmd_leader=q_c,
            q_c_leader=q_c,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=np.full(n, np.nan),
            delta_corr=delta,
            tau_cmd=tau_cmd,
            tau_ext_shi=tau_ext,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=contact_probability,
            contact_state=contact_state,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q,
            dq=dq,
            grip=float(grip),
            tau_ext=tau_components["tau_residual"],
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)


def _run_leader_admittance_position_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
    shi: Optional[MinimalistTorqueEstimator],
) -> None:
    n = system.num_arm_joints
    dt = float(system.dt)
    observer_mode = str(args.admittance_observer).lower()
    if observer_mode == "yamane":
        raise ValueError(
            "leader_admittance_position cannot use Yamane because Dynamixel "
            "position mode hides the commanded motor torque. Use "
            "leader_admittance_impedance --admittance-observer yamane instead."
        )

    follower_cmd_stop: threading.Event | None = None
    follower_cmd_lock: threading.Lock | None = None
    follower_cmd_state: dict[str, np.ndarray] | None = None
    follower_cmd_thread: Thread | None = None
    if not args.no_follower:
        if (
            not system.teleop_enabled
            or not system.teleop_prepared
            or system._direct_follower_robot is None
        ):
            raise RuntimeError(
                "leader_admittance_position with follower requires a prepared "
                "direct-RTDE follower. Check teleop.enable and follower setup."
            )

    if float(args.settle_time) > 0.0:
        print("[SETTLE] gravity comp warmup")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < float(args.settle_time):
            _leader_gravity_hold(system, dt)

    q0, _, _, _ = system.get_leader_joint_states()
    q_c = np.asarray(q0[:n], dtype=float).copy()
    dq_c = np.zeros(n, dtype=float)
    q_bota_joint = q_c.copy()
    dq_bota_joint = np.zeros(n, dtype=float)
    contact_gate = ContactGate(n)
    last_tau_residual = np.zeros(n, dtype=float)

    mass = np.full(n, max(float(args.adm_mass), 1e-6), dtype=float)
    damp = np.full(n, max(float(args.adm_damp), 0.0), dtype=float)
    stiff = np.full(n, max(float(args.adm_stiff), 0.0), dtype=float)
    leak = max(float(args.adm_leak), 0.0)
    delta_max = max(float(args.adm_delta_max), 1e-6)

    admittance_source = str(args.admittance_source).lower()
    bota_enabled = bool(args.bota_enable or admittance_source.startswith("bota_"))
    bota_reader: BotaMiniOneReader | None = None
    bota_conditioner: BotaWrenchConditioner | None = None
    bota_kin: GelloTaskspaceKinematics | None = None
    bota_cart: CartesianAdmittance6D | None = None
    bota_hfvc: ForceControlStyleAdmittance6D | None = None
    bota_se3: SE3PoseAdmittance6D | None = None
    bota_config = Path(args.bota_config)
    if not bota_config.is_absolute():
        bota_config = (REPO_ROOT / bota_config).resolve()

    if bota_enabled:
        bota_kin = GelloTaskspaceKinematics(
            system,
            n,
            frame_name=str(args.bota_frame).strip() or None,
        )
        bota_conditioner = BotaWrenchConditioner(
            cutoff_hz=float(args.bota_filter_cutoff),
            alpha=float(args.bota_filter_alpha),
            deadband=_expand_list(_parse_float_list(args.bota_deadband), 6, 0.0),
            saturation=_expand_list(_parse_float_list(args.bota_saturation), 6, 1e9),
            gravity_comp=bool(args.bota_gravity_comp),
            payload_mass_kg=float(args.bota_payload_mass),
            payload_com_sensor=_expand_list(_parse_float_list(args.bota_payload_com), 3, 0.0),
            gravity_sign=float(args.bota_gravity_sign),
        )
        bota_cart = CartesianAdmittance6D(
            mass=_expand_list(_parse_float_list(args.bota_cart_mass), 6, 1.0),
            damping=_expand_list(_parse_float_list(args.bota_cart_damp), 6, 1.0),
            stiffness=_expand_list(_parse_float_list(args.bota_cart_stiff), 6, 0.0),
            leak=leak,
            max_offset=_expand_list(_parse_float_list(args.bota_cart_max), 6, 0.05),
            axis_mask=_expand_list(_parse_float_list(args.bota_axis_mask), 6, 1.0),
        )
        bota_hfvc = ForceControlStyleAdmittance6D(
            mass=_expand_list(_parse_float_list(args.bota_cart_mass), 6, 1.0),
            damping=_expand_list(_parse_float_list(args.bota_cart_damp), 6, 1.0),
            stiffness=_expand_list(_parse_float_list(args.bota_cart_stiff), 6, 0.0),
            leak=leak,
            max_offset=_expand_list(_parse_float_list(args.bota_cart_max), 6, 0.05),
            axis_mask=_expand_list(_parse_float_list(args.bota_axis_mask), 6, 1.0),
            force_dims=int(args.bota_hfvc_force_dims),
            stiction=_expand_list(_parse_float_list(args.bota_hfvc_stiction), 6, 0.0),
            wrench_sign=float(args.bota_wrench_sign),
        )
        bota_se3 = SE3PoseAdmittance6D(
            mass=_expand_list(_parse_float_list(args.bota_cart_mass), 6, 1.0),
            damping=_expand_list(_parse_float_list(args.bota_cart_damp), 6, 1.0),
            stiffness=_expand_list(_parse_float_list(args.bota_cart_stiff), 6, 0.0),
            leak=leak,
            max_offset=_expand_list(_parse_float_list(args.bota_cart_max), 6, 0.05),
            axis_mask=_expand_list(_parse_float_list(args.bota_axis_mask), 6, 1.0),
            stiction=_expand_list(_parse_float_list(args.bota_hfvc_stiction), 6, 0.0),
            wrench_sign=float(args.bota_wrench_sign),
            output_mode=str(args.bota_se3_output_mode),
        )
        bota_reader = BotaMiniOneReader(bota_config, driver_tare=bool(args.bota_driver_tare))
        bota_reader.start()

        warmup_seconds = max(float(args.bota_warmup_seconds), 0.0)
        if warmup_seconds > 0.0:
            print(f"[BOTA] warmup/discarding frames for {warmup_seconds:.2f}s")
            t_warmup = time.perf_counter()
            warmup_new = 0
            warmup_dup = 0
            while time.perf_counter() - t_warmup < warmup_seconds:
                try:
                    *_, is_new = bota_reader.read_latest()
                    if is_new:
                        warmup_new += 1
                    else:
                        warmup_dup += 1
                except Exception as exc:
                    print(f"[WARN] Bota warmup read failed: {exc}")
                    break
                _leader_gravity_hold(system, dt)
            print(f"[BOTA] warmup frames new={warmup_new} duplicate={warmup_dup}")

        bias_samples: list[np.ndarray] = []
        bias_seconds = max(float(args.bota_bias_seconds), 0.0)
        if bias_seconds > 0.0:
            print(f"[BOTA] collecting software bias for {bias_seconds:.2f}s")
            t_bias = time.perf_counter()
            bias_new = 0
            bias_dup = 0
            while time.perf_counter() - t_bias < bias_seconds:
                try:
                    raw_wrench, *_, is_new = bota_reader.read_latest()
                    if is_new:
                        bias_samples.append(raw_wrench.copy())
                        bias_new += 1
                    else:
                        bias_dup += 1
                except Exception as exc:
                    print(f"[WARN] Bota bias read failed: {exc}")
                    break
                _leader_gravity_hold(system, dt)
            print(f"[BOTA] bias frames new={bias_new} duplicate={bias_dup}")
            bota_conditioner.set_bias(bias_samples)
        print(
            "[BOTA] enabled "
            f"source={admittance_source} frame={bota_kin.frame_name} "
            f"gravity_comp={bool(args.bota_gravity_comp)} "
            f"filter_cutoff={float(args.bota_filter_cutoff):.2f}Hz "
            f"filter_alpha={float(args.bota_filter_alpha):.3f}"
        )

    print(f"[ADMITTANCE] position mode observer={observer_mode} source={admittance_source}")
    if system.driver is not None:
        raw_pos_hold = _switch_leader_to_position_hold(system, args)
        if bool(getattr(args, "leader_position_recover_pre_handoff", False)):
            pre_handoff_raw = getattr(system, "_leader_position_pre_handoff_raw", None)
            if pre_handoff_raw is not None:
                raw_pos_hold = _recover_leader_position_after_handoff(
                    system,
                    np.asarray(pre_handoff_raw, dtype=float),
                    raw_pos_hold,
                    args,
                )
            else:
                print("[DXL] pre-handoff recovery requested but no pre-handoff pose is available")
        last_sent_raw = raw_pos_hold.copy()
        q_c = (
            raw_pos_hold[:n] - system.joint_offsets[:n]
        ) * system.joint_signs[:n]
        q_bota_joint = q_c.copy()
        dq_c = np.zeros(n, dtype=float)
        dq_bota_joint = np.zeros(n, dtype=float)
    else:
        last_sent_raw = np.zeros(system.num_motors, dtype=float)

    position_mode_hold_q = q_c.copy()
    if bota_se3 is not None:
        bota_se3.reset()
    if str(args.test_mode) == "static_hold" and hasattr(traj_interp, "traj_buf"):
        traj_interp.traj_buf.write(
            np.tile(position_mode_hold_q, (int(args.horizon), 1)),
            time.monotonic(),
        )

    if not args.no_follower:
        initial_follower_action = system._build_follower_action(
            q_c,
            float(getattr(system, "gripper_pos", 0.0)),
        )
        initial_follower_target = np.asarray(initial_follower_action[:n], dtype=float)
        (
            follower_cmd_stop,
            follower_cmd_lock,
            follower_cmd_state,
            follower_cmd_thread,
        ) = _start_direct_follower_command_thread(system, args, initial_follower_target)

    arm_settle_s = max(float(getattr(args, "leader_position_arm_settle_seconds", 0.0)), 0.0)
    if arm_settle_s > 0.0:
        print(
            f"[ARMING] settling {arm_settle_s:.2f}s after handoff/follower start "
            "before test clock and recording"
        )
        t_arm = time.perf_counter()
        while time.perf_counter() - t_arm < arm_settle_s:
            time.sleep(min(0.02, max(0.0, arm_settle_s - (time.perf_counter() - t_arm))))

    if bota_se3 is not None:
        bota_se3.reset()
    if hasattr(traj_interp, "traj_buf"):
        traj_interp.traj_buf.write(
            np.tile(position_mode_hold_q, (int(args.horizon), 1)),
            time.monotonic(),
        )

    t_start = time.monotonic()
    local_reference_center = position_mode_hold_q.copy()
    last_warn = 0.0
    last_bota_warn = 0.0
    last_follower_cmd_warn = 0.0

    try:
        while True:
            loop_t0 = time.perf_counter()
            t_mono = time.monotonic()

            if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
                break

            currents_arm = None
            if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
                pos_raw, vel_raw, cur_raw = system.driver.get_positions_velocities_and_currents()
                q = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
                dq = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

                if len(pos_raw) > n:
                    grip_raw = float(pos_raw[-1])
                    grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
                    system.leader_gripper_raw_rad = grip_raw
                else:
                    grip = 0.0
                    grip_raw = 0.0

                if len(vel_raw) > n:
                    grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
                else:
                    grip_vel = 0.0

                system.gripper_pos = float(grip)
                system._last_gripper_vel = float(grip_vel)

                if observer_mode == "shi" and shi is not None:
                    currents_arm = np.asarray(cur_raw[:n], dtype=float) * system.joint_signs[:n]
            else:
                q, dq, grip, grip_vel = system.get_leader_joint_states()
                if observer_mode == "shi" and shi is not None and system.driver is not None:
                    currents = system.driver.get_currents()
                    currents_arm = np.asarray(currents[:n], dtype=float) * system.joint_signs[:n]

            local_ref = _leader_position_local_reference(
                args,
                n,
                local_reference_center,
                t_mono - t_start,
            )
            if local_ref is not None:
                q_ref, dq_ref = local_ref
                is_stale = False
            else:
                q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
            ddq_ref = ddq_computer.update(dq_ref, t_mono)

            bota_wrench_raw = np.zeros(6, dtype=float)
            bota_wrench_base = np.zeros(6, dtype=float)
            bota_wrench_conditioned = np.zeros(6, dtype=float)
            bota_tau_joint = np.zeros(n, dtype=float)
            bota_delta_q_cartesian = np.zeros(n, dtype=float)
            bota_delta_q_hfvc = np.zeros(n, dtype=float)
            bota_delta_q_se3 = np.zeros(n, dtype=float)
            bota_delta_q_joint = np.zeros(n, dtype=float)
            bota_task_offset = np.zeros(6, dtype=float)
            bota_task_offset_hfvc = np.zeros(6, dtype=float)
            bota_task_offset_se3 = np.zeros(6, dtype=float)
            bota_pose_error_se3 = np.zeros(6, dtype=float)
            bota_status = np.zeros(4, dtype=float)

            if bota_enabled and bota_reader is not None and bota_conditioner is not None and bota_kin is not None:
                try:
                    bota_wrench_raw, bota_status, bota_timestamp_us, _, _, bota_is_new = bota_reader.read_latest()
                    pos_ref, rot_ref, jacobian_6d = bota_kin.pose_and_jacobian(q_ref)
                    pos_current, rot_current, _ = bota_kin.pose_and_jacobian(q)
                    r_base_sensor = rot_current if admittance_source == "bota_se3" else rot_ref
                    bota_wrench_base, bota_wrench_conditioned = bota_conditioner.update(
                        bota_wrench_raw,
                        r_base_sensor,
                        dt,
                        timestamp_us=bota_timestamp_us,
                        is_new_frame=bota_is_new,
                    )
                    bota_tau_joint = (
                        float(args.bota_joint_gain)
                        * (jacobian_6d.T @ bota_wrench_conditioned)
                    )

                    ddq_bota_joint = (
                        bota_tau_joint - damp * dq_bota_joint - stiff * (q_bota_joint - q_ref)
                    ) / mass
                    dq_bota_joint = (1.0 - leak * dt) * (dq_bota_joint + ddq_bota_joint * dt)
                    q_bota_joint = q_bota_joint + dq_bota_joint * dt
                    bota_delta_q_joint = np.clip(q_bota_joint - q_ref, -delta_max, delta_max)
                    q_bota_joint = q_ref + bota_delta_q_joint

                    if bota_cart is not None:
                        bota_task_offset = bota_cart.step(bota_wrench_conditioned, dt)
                        bota_delta_q_cartesian = _damped_least_squares_delta_q(
                            jacobian_6d,
                            bota_task_offset,
                            float(args.bota_dls_damping),
                        )
                        bota_delta_q_cartesian = np.clip(
                            bota_delta_q_cartesian,
                            -delta_max,
                            delta_max,
                        )
                    if bota_hfvc is not None:
                        bota_task_offset_hfvc = bota_hfvc.step(bota_wrench_conditioned, dt)
                        bota_delta_q_hfvc = _damped_least_squares_delta_q(
                            jacobian_6d,
                            bota_task_offset_hfvc,
                            float(args.bota_dls_damping),
                        )
                        bota_delta_q_hfvc = np.clip(
                            bota_delta_q_hfvc,
                            -delta_max,
                            delta_max,
                        )
                    if bota_se3 is not None:
                        bota_task_offset_se3, bota_pose_error_se3 = bota_se3.step(
                            pos_ref,
                            rot_ref,
                            pos_current,
                            rot_current,
                            bota_wrench_conditioned,
                            dt,
                        )
                        bota_delta_q_se3 = _damped_least_squares_delta_q(
                            jacobian_6d,
                            bota_task_offset_se3,
                            float(args.bota_dls_damping),
                        )
                        bota_delta_q_se3 = np.clip(
                            bota_delta_q_se3,
                            -delta_max,
                            delta_max,
                        )
                except Exception as exc:
                    if (t_mono - last_bota_warn) > 1.0:
                        last_bota_warn = t_mono
                        print(f"[WARN] Bota read/admittance update failed: {exc}")

            if admittance_source == "observer":
                contact_probability, contact_state = contact_gate.update(
                    last_tau_residual,
                    ddq_ref,
                    correction_active=True,
                )
                tau_contact = last_tau_residual if contact_probability >= 0.6 else np.zeros(n)
                ddq_c = (tau_contact - damp * dq_c - stiff * (q_c - q_ref)) / mass
                dq_c = (1.0 - leak * dt) * (dq_c + ddq_c * dt)
                q_c = q_c + dq_c * dt
                delta = np.clip(q_c - q_ref, -delta_max, delta_max)
                q_c = q_ref + delta
            elif admittance_source == "bota_joint":
                contact_probability, contact_state = _bota_contact_probability(
                    bota_wrench_conditioned,
                    float(args.bota_contact_force_scale),
                    float(args.bota_contact_torque_scale),
                )
                delta = bota_delta_q_joint.copy()
                q_c = q_ref + delta
                dq_c = dq_bota_joint.copy()
            elif admittance_source == "bota_cartesian":
                contact_probability, contact_state = _bota_contact_probability(
                    bota_wrench_conditioned,
                    float(args.bota_contact_force_scale),
                    float(args.bota_contact_torque_scale),
                )
                delta = bota_delta_q_cartesian.copy()
                q_c = q_ref + delta
                dq_c = np.zeros(n, dtype=float)
            elif admittance_source == "bota_hfvc":
                contact_probability, contact_state = _bota_contact_probability(
                    bota_wrench_conditioned,
                    float(args.bota_contact_force_scale),
                    float(args.bota_contact_torque_scale),
                )
                delta = bota_delta_q_hfvc.copy()
                q_c = q_ref + delta
                dq_c = np.zeros(n, dtype=float)
                bota_task_offset = bota_task_offset_hfvc.copy()
            elif admittance_source == "bota_se3":
                contact_probability, contact_state = _bota_contact_probability(
                    bota_wrench_conditioned,
                    float(args.bota_contact_force_scale),
                    float(args.bota_contact_torque_scale),
                )
                delta = bota_delta_q_se3.copy()
                q_c = q_ref + delta
                dq_c = np.zeros(n, dtype=float)
                bota_task_offset = bota_task_offset_se3.copy()
            else:
                raise ValueError(f"Unsupported admittance source: {admittance_source}")

            delta = np.clip(q_c - q_ref, -delta_max, delta_max)
            q_c = q_ref + delta

            target_hw = np.zeros(system.num_motors)
            target_hw[:n] = q_c * system.joint_signs[:n] + system.joint_offsets[:n]
            if system.num_motors > n:
                target_hw[-1] = getattr(system, "leader_gripper_raw_rad", 0.0)

            for i in range(n):
                while target_hw[i] - last_sent_raw[i] > np.pi:
                    target_hw[i] -= 2.0 * np.pi
                while target_hw[i] - last_sent_raw[i] < -np.pi:
                    target_hw[i] += 2.0 * np.pi

            system.driver.set_joints(target_hw.tolist())
            last_sent_raw = target_hw.copy()

            q_cmd_follower = np.full(n, np.nan)
            if follower_cmd_state is not None and follower_cmd_lock is not None:
                try:
                    follower_action = system._build_follower_action(q_c, float(grip))
                    q_cmd_follower = np.asarray(follower_action[:n], dtype=float)
                    with follower_cmd_lock:
                        follower_cmd_state["q"] = q_cmd_follower.copy()
                except Exception as exc:
                    if (t_mono - last_follower_cmd_warn) > 1.0:
                        last_follower_cmd_warn = t_mono
                        print(f"[WARN] follower command mapping failed: {exc}")

            tau_cmd = np.zeros(n, dtype=float)
            if observer_mode == "none":
                tau_ext = np.zeros(n, dtype=float)
                tau_components = system.separate_contact_torque_components(
                    tau_model=tau_cmd,
                    tau_ext_shi=tau_ext,
                )
            else:
                tau_ext, tau_components = _shi_torque_components(
                    system=system,
                    shi=shi,
                    q=q,
                    dq=dq,
                    currents_arm=currents_arm,
                    tau_model=tau_cmd,
                )
            last_tau_residual = tau_components["tau_residual"].copy()

            q_follower = np.zeros(n)
            dq_follower = np.zeros(n)
            if system.teleop_enabled:
                try:
                    q_follower, dq_follower = system.get_follower_arm_state()
                except Exception:
                    pass

            epsilon = q - q_c
            recorder.record(
                t_mono=t_mono,
                q_ref=q_ref,
                dq_ref=dq_ref,
                ddq_ref=ddq_ref,
                q_leader=q,
                dq_leader=dq,
                q_cmd_leader=q_c,
                q_c_leader=q_c,
                q_follower=q_follower,
                dq_follower=dq_follower,
                q_cmd_follower=q_cmd_follower,
                delta_corr=delta,
                tau_cmd=tau_cmd,
                tau_ext_shi=tau_ext,
                tau_model=tau_components["tau_model"],
                tau_meas=tau_components["tau_meas"],
                tau_residual=tau_components["tau_residual"],
                wrench=bota_wrench_conditioned,
                bota_wrench_raw=bota_wrench_raw,
                bota_wrench_base=bota_wrench_base,
                bota_wrench_conditioned=bota_wrench_conditioned,
                bota_tau_joint=bota_tau_joint,
                bota_delta_q_cartesian=bota_delta_q_cartesian,
                bota_delta_q_hfvc=bota_delta_q_hfvc,
                bota_delta_q_se3=bota_delta_q_se3,
                bota_delta_q_joint=bota_delta_q_joint,
                bota_task_offset=bota_task_offset,
                bota_task_offset_hfvc=bota_task_offset_hfvc,
                bota_task_offset_se3=bota_task_offset_se3,
                bota_pose_error_se3=bota_pose_error_se3,
                bota_status=bota_status,
                contact_probability=contact_probability,
                contact_state=contact_state,
                epsilon=epsilon,
            )

            tau_obs = bota_tau_joint if admittance_source.startswith("bota_") else tau_components["tau_residual"]
            obs_snap.write(
                timestamp=t_mono,
                q=q,
                dq=dq,
                grip=float(grip),
                tau_ext=tau_obs,
                wrench=bota_wrench_conditioned,
            )

            if is_stale and (t_mono - last_warn) > 1.0:
                last_warn = t_mono
                print("[WARN] trajectory stale")

            _sleep_remaining(dt, loop_t0)
    finally:
        if follower_cmd_stop is not None:
            follower_cmd_stop.set()
        if follower_cmd_thread is not None:
            follower_cmd_thread.join(timeout=1.0)
        if not args.no_follower and system._direct_follower_robot is not None:
            try:
                follower_robot = getattr(system._direct_follower_robot, "robot", None)
                if follower_robot is not None and hasattr(follower_robot, "stopJ"):
                    follower_robot.stopJ(2.0)
            except Exception:
                pass
        if bota_reader is not None:
            bota_reader.close()
        if system.driver is not None:
            try:
                system.driver.set_torque_mode(False)
                time.sleep(0.05)
                restore = getattr(system, "_leader_position_handoff_restore", {})
                if isinstance(restore, dict) and restore.get("current_limit"):
                    system.driver.set_current_limits(restore["current_limit"])
                    time.sleep(0.02)
                system.driver.set_operating_mode(0)
                time.sleep(0.05)
                system.driver.set_torque_mode(True)
                time.sleep(0.05)
            except Exception:
                pass


def _run_four_channel_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
) -> None:
    from gello.cr_dagger.core.yamane_observer import GeneralizedMomentumObserver

    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("4-channel mode requires teleop.enable=true with direct RTDE")

    follower = system._direct_follower_robot
    n = system.num_arm_joints
    dt = 1.0 / 500.0

    four_cfg = system.config.get("teleop", {}).get("four_channel", {})
    K_virtual = float(four_cfg.get("K_virtual", 40.0))
    correction_gain = float(four_cfg.get("correction_gain", 1.0))
    force_feedback_gain = float(four_cfg.get("force_feedback_gain", 0.3))
    velocity_gate_threshold = float(four_cfg.get("velocity_gate_threshold", 1.5))
    deadband_tau = float(four_cfg.get("deadband_tau", 0.15))
    mirror_source = str(four_cfg.get("mirror_source", "cmd")).lower()
    yamane_K_obs = float(four_cfg.get("yamane_K_obs", 50.0))
    yamane_filter_fc = float(four_cfg.get("yamane_filter_fc", 5.0))
    if args.mirror_source:
        mirror_source = str(args.mirror_source)

    kp_mirror = _joint_array_from_value(system.policy_impedance_kp, n, 5.0)
    kd_mirror = _joint_array_from_value(system.policy_impedance_kd, n, 0.5)

    yamane = GeneralizedMomentumObserver(
        pin_model=system.pin_model,
        num_arm_joints=n,
        K_obs=yamane_K_obs,
        dt=dt,
        filter_fc=yamane_filter_fc,
    )

    has_gripper = bool(
        hasattr(follower, "gripper")
        and hasattr(follower, "_use_gripper")
        and follower._use_gripper
    )
    gripper_lock = threading.Lock()
    gripper_state = {"latest": None}
    if has_gripper:
        def gripper_worker() -> None:
            while True:
                pos = None
                with gripper_lock:
                    if gripper_state["latest"] is not None:
                        pos = gripper_state["latest"]
                        gripper_state["latest"] = None
                if pos is not None:
                    try:
                        follower.gripper.move(pos, 255, 10)
                    except Exception:
                        pass
                time.sleep(1.0 / 30.0)

        Thread(target=gripper_worker, daemon=True, name="epsilon-4ch-gripper").start()

    if float(args.settle_time) > 0.0:
        print("[SETTLE] gravity comp warmup")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < float(args.settle_time):
            _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0
    q_mirror_target = np.zeros(n, dtype=float)
    dq_mirror_target = np.zeros(n, dtype=float)
    delta_follower = np.zeros(n, dtype=float)
    initialized = False

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, _ = system.driver.get_positions_velocities_and_currents()
        else:
            pos_raw, vel_raw = system.driver.get_positions_and_velocities()

        q_leader = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
        dq_leader = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

        if len(pos_raw) > n:
            grip_raw = float(pos_raw[-1])
            grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
            system.leader_gripper_raw_rad = grip_raw
        else:
            grip_raw = 0.0
            grip = 0.0

        if len(vel_raw) > n:
            grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
        else:
            grip_vel = 0.0

        system.gripper_pos = float(grip)
        system._last_gripper_vel = float(grip_vel)

        if not initialized:
            yamane.reset(q_leader, dq_leader)
            q_mirror_target = q_leader.copy()
            dq_mirror_target = np.zeros(n, dtype=float)
            initialized = True

        q_ref_leader, dq_ref_leader, is_stale = traj_interp.get_reference(t_mono)
        q_ref_leader = np.asarray(q_ref_leader[:n], dtype=float)
        dq_ref_leader = (
            np.asarray(dq_ref_leader[:n], dtype=float)
            if dq_ref_leader is not None
            else np.zeros(n, dtype=float)
        )
        ddq_ref = ddq_computer.update(dq_ref_leader, t_mono)

        q_ref_follower = system._build_follower_action(q_ref_leader, float(grip))[:n]
        if system.map_signs is not None and system.map_index is not None:
            dq_ref_follower = system.map_signs * dq_ref_leader[system.map_index]
        else:
            dq_ref_follower = dq_ref_leader[:n]

        if mirror_source == "cmd":
            delta_leader = np.zeros(n, dtype=float)
            if system.map_signs is not None and system.map_index is not None:
                lim = min(len(system.map_index), len(system.map_signs), n)
                for i in range(lim):
                    leader_idx = int(system.map_index[i])
                    sign = float(system.map_signs[i])
                    if leader_idx < n and abs(sign) > 1e-6:
                        delta_leader[leader_idx] = delta_follower[i] / sign
            q_mirror_target = q_ref_leader + delta_leader
            dq_mirror_target = dq_ref_leader.copy()
        elif mirror_source in ("actual", "blend"):
            q_actual_target = q_mirror_target.copy()
            dq_actual_target = dq_mirror_target.copy()
            try:
                q_ur5e, dq_ur5e = system.get_follower_arm_state()
                q_actual_target = np.zeros(n, dtype=float)
                dq_actual_target = np.zeros(n, dtype=float)
                if (
                    system.map_index is not None
                    and system.map_signs is not None
                    and system.map_offsets is not None
                ):
                    lim = min(len(system.map_index), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            q_actual_target[leader_idx] = (
                                q_ur5e[i] - system.map_offsets[i]
                            ) / sign
                            dq_actual_target[leader_idx] = dq_ur5e[i] / sign
            except Exception:
                pass
            if mirror_source == "actual":
                q_mirror_target = q_actual_target
                dq_mirror_target = dq_actual_target
            else:
                delta_leader = np.zeros(n, dtype=float)
                if system.map_signs is not None and system.map_index is not None:
                    lim = min(len(system.map_index), len(system.map_signs), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            delta_leader[leader_idx] = delta_follower[i] / sign
                q_cmd_target = q_ref_leader + delta_leader
                dq_cmd_target = dq_ref_leader.copy()
                weights = np.array(
                    [
                        float(args.mirror_blend_cmd),
                        float(args.mirror_blend_actual),
                        float(args.mirror_blend_policy),
                    ],
                    dtype=float,
                )
                weights = np.maximum(weights, 0.0)
                denom = float(np.sum(weights))
                weights = weights / denom if denom > 1e-9 else np.array([1.0, 0.0, 0.0])
                q_mirror_target = (
                    weights[0] * q_cmd_target
                    + weights[1] * q_actual_target
                    + weights[2] * q_ref_leader
                )
                dq_mirror_target = (
                    weights[0] * dq_cmd_target
                    + weights[1] * dq_actual_target
                    + weights[2] * dq_ref_leader
                )
        else:
            q_mirror_target = q_ref_leader.copy()
            dq_mirror_target = dq_ref_leader.copy()

        tau_force_feedback = np.zeros(n, dtype=float)
        wrench_ur5e = np.zeros(6, dtype=float)
        if force_feedback_gain > 0.0:
            try:
                wrench_ur5e, tcp_jt = system.get_follower_tcp_force()
                if system.map_index is not None and system.map_signs is not None:
                    lim = min(len(system.map_index), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            tau_force_feedback[leader_idx] = (
                                force_feedback_gain * tcp_jt[i] / sign
                            )
            except Exception:
                pass

        tau_gravity = system.gravity_compensation(q_leader, dq_leader)
        tau_friction = system.friction_compensation(dq_leader)
        tau_limit, torque_gripper = system.joint_limit_barrier(
            q_leader, dq_leader, float(grip), float(grip_vel)
        )
        tau_damping = np.zeros(n, dtype=float)
        if system.gravity_comp_velocity_damping != 0.0:
            tau_damping = -system.gravity_comp_velocity_damping * dq_leader

        tau_mirror = (
            kp_mirror * (q_mirror_target - q_leader)
            + kd_mirror * (dq_mirror_target - dq_leader)
        )
        tau_mirror *= _impedance_ramp_scale(args, t_mono - t_start)

        tau_cmd_total = (
            tau_gravity
            + tau_friction
            + tau_limit
            + tau_damping
            + tau_mirror
            + tau_force_feedback
        )
        system.set_leader_joint_torque(tau_cmd_total, float(torque_gripper))

        tau_ext_human = yamane.update(q=q_leader, dq=dq_leader, tau_cmd=tau_cmd_total, dt=dt)
        tau_components = system.separate_contact_torque_components(
            tau_model=tau_cmd_total,
            tau_ext_shi=tau_ext_human,
        )

        tau_ext_db = np.where(
            np.abs(tau_ext_human) > deadband_tau,
            tau_ext_human - np.sign(tau_ext_human) * deadband_tau,
            0.0,
        )
        if velocity_gate_threshold > 1e-6:
            speed = float(np.linalg.norm(dq_leader))
            v_gate = 1.0 / np.sqrt(1.0 + (speed / velocity_gate_threshold) ** 2)
        else:
            v_gate = 1.0
        tau_ext_gated = tau_ext_db * v_gate

        k_virtual_safe = K_virtual if abs(K_virtual) > 1e-6 else 1.0
        delta_leader_new = tau_ext_gated / k_virtual_safe
        if system.map_signs is not None and system.map_index is not None:
            delta_follower = system.map_signs * delta_leader_new[system.map_index] * correction_gain
        else:
            delta_follower = delta_leader_new * correction_gain

        q_cmd_ur5e = q_ref_follower + delta_follower
        follower.command_joint_state_impedance(
            target_joints=q_cmd_ur5e,
            target_velocities=dq_ref_follower,
            kp=float(args.follower_kp),
            kd=float(args.follower_kd),
        )

        if has_gripper:
            gripper_action = system._build_follower_action(q_leader, float(grip))
            if len(gripper_action) > 6:
                with gripper_lock:
                    gripper_state["latest"] = int(
                        np.clip(gripper_action[-1] * 255, 0, 255)
                    )

        q_follower, dq_follower = system.get_follower_arm_state()
        q_follower = np.asarray(q_follower[:n], dtype=float)
        dq_follower = np.asarray(dq_follower[:n], dtype=float)

        epsilon = q_follower - q_cmd_ur5e
        recorder.record(
            t_mono=t_mono,
            q_ref=q_cmd_ur5e,
            dq_ref=dq_ref_follower,
            ddq_ref=ddq_ref,
            q_leader=q_leader,
            dq_leader=dq_leader,
            q_cmd_leader=q_mirror_target,
            q_c_leader=q_mirror_target,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=q_cmd_ur5e,
            delta_corr=delta_follower,
            tau_cmd=tau_cmd_total,
            tau_ext_shi=tau_ext_human,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=float(np.clip(np.linalg.norm(tau_ext_gated) / 0.45, 0.0, 1.0)),
            contact_state=(
                ContactGate.CORRECTION_ACTIVE
                if np.linalg.norm(delta_follower) > 1e-6
                else ContactGate.NO_CONTACT
            ),
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q_leader,
            dq=dq_leader,
            grip=float(grip),
            tau_ext=tau_ext_human,
            wrench=wrench_ur5e,
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)

    try:
        follower.robot.stopJ(2.0)
    except Exception:
        pass


def main() -> int:
    args = _parse_args()
    arch_mode = _resolve_arch_mode(args)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    running = True
    system: Optional[FACTRGravityCompensation] = None
    traj_buf: SharedTrajectoryBuffer | None = None
    obs_snap: SharedObservationSnapshot | None = None
    policy_proc: mp.Process | None = None
    stop_event: Any | None = None

    def _sig(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")

        n = int(system.num_arm_joints)
        policy_impedance_cfg = system.config.get("controller", {}).get("policy_impedance", {})
        kp_value = args.kp if args.kp is not None else policy_impedance_cfg.get("kp", system.policy_impedance_kp)
        kd_value = args.kd if args.kd is not None else policy_impedance_cfg.get("kd", system.policy_impedance_kd)
        system.policy_impedance_kp = _joint_array_from_value(kp_value, n, 5.0)
        system.policy_impedance_kd = _joint_array_from_value(kd_value, n, 0.5)
        print(
            "[IMPEDANCE] leader policy "
            f"Kp={_format_joint_array(system.policy_impedance_kp)} "
            f"Kd={_format_joint_array(system.policy_impedance_kd)}"
        )

        if system.driver is not None:
            system.driver.set_torque_mode(False)
            time.sleep(0.05)
            system.driver.set_operating_mode(0)
            time.sleep(0.05)
            system.driver.set_torque_mode(True)
            time.sleep(0.05)

        startup_settle_time = max(float(args.settle_time), 0.0)
        if startup_settle_time > 0.0:
            print("[SETTLE] gravity comp warmup before trajectory capture")
            t_settle = time.perf_counter()
            while time.perf_counter() - t_settle < startup_settle_time:
                _leader_gravity_hold(system, float(system.dt))

        q0 = _prime_policy_impedance_from_hold(system)
        args.settle_time = 0.0

        horizon = int(args.horizon)
        traj_buf = SharedTrajectoryBuffer(
            name="epsilon_traj",
            horizon=horizon,
            n_joints=n,
            create=True,
        )
        obs_snap = SharedObservationSnapshot(
            name="epsilon_obs",
            n_joints=n,
            img_height=480,
            img_width=640,
            create=True,
        )

        initial_traj = np.tile(q0, (horizon, 1))
        traj_buf.write(initial_traj, time.monotonic())

        traj_interp = TrajectoryInterpolator(
            traj_buf=traj_buf,
            action_dt=float(args.action_dt),
            stale_threshold_s=5.0,
            fallback_q=q0.copy(),
        )

        policy_type, policy_config = _build_policy(args, n, q0, config_path, system.config)

        stop_event = mp.Event()
        policy_proc = mp.Process(
            target=policy_worker,
            args=(
                "epsilon_traj",
                "epsilon_obs",
                horizon,
                n,
                float(args.action_dt),
                policy_type,
                policy_config,
                stop_event,
            ),
            daemon=True,
        )
        policy_proc.start()
        print(f"[POLICY] started pid={policy_proc.pid} ({policy_type})")

        rate_hz = max(1.0, 1.0 / float(system.dt))
        max_duration = float(args.duration) if float(args.duration) > 0 else float(args.max_duration)
        recorder = EpsilonRecorder(
            n_joints=n,
            max_duration_s=max_duration,
            rate_hz=rate_hz,
        )
        ddq_computer = FilteredDDQComputer(n_joints=n)

        shi: Optional[MinimalistTorqueEstimator] = None
        needs_shi = arch_mode == "leader_tracking" or (
            arch_mode == "leader_observer"
            and str(args.admittance_observer).lower() == "shi"
        ) or (
            arch_mode in ("leader_admittance_impedance", "leader_admittance_position")
            and str(args.admittance_observer).lower() == "shi"
        )
        if not args.no_shi and needs_shi:
            try:
                shi = _build_shi(system, config_path, n)
            except Exception as exc:
                print(f"[WARN] Shi estimator unavailable: {exc}")

        target_label = arch_mode
        mirror_source_used = None
        if arch_mode == "four_channel":
            if args.target != "follower":
                print("[WARN] four_channel records follower epsilon; ignoring --target leader")
            mirror_source_used = str(
                args.mirror_source
                or system.config.get("teleop", {}).get("four_channel", {}).get("mirror_source", "cmd")
            )
            _run_four_channel_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap)
        elif arch_mode in ("leader_tracking", "leader_observer"):
            _run_leader_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap, shi)
        elif arch_mode == "leader_admittance_impedance":
            _run_leader_admittance_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap, shi)
        elif arch_mode == "leader_admittance_position":
            _run_leader_admittance_position_mode(
                system,
                args,
                traj_interp,
                recorder,
                ddq_computer,
                obs_snap,
                shi,
            )
        elif arch_mode == "follower_tracking":
            _run_follower_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap)
        else:
            raise ValueError(f"Unknown mode: {arch_mode}")

        out_dir = Path(args.log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.monotonic())
        fname = f"epsilon_{target_label}_{args.test_mode}_{timestamp}.npz"
        out_path = out_dir / fname
        metadata = {
            "mode": arch_mode,
            "target": target_label,
            "architecture": arch_mode,
            "leader_command_mode": (
                "position"
                if arch_mode == "leader_admittance_position"
                else "current"
            ),
            "wrench_source": ("bota" if str(args.admittance_source).startswith("bota_") else "none"),
            "test_mode": args.test_mode,
            "compensation": args.compensation,
            "joint_index": int(args.joint_index),
            "amplitude": float(args.amplitude),
            "frequency": float(args.frequency),
            "multi_amplitudes": _parse_float_list(args.multi_amplitudes),
            "multi_frequencies": _parse_float_list(args.multi_frequencies),
            "ramp_delta": float(args.ramp_delta),
            "ramp_duration": float(args.ramp_duration),
            "kp": np.asarray(system.policy_impedance_kp, dtype=float).tolist(),
            "kd": np.asarray(system.policy_impedance_kd, dtype=float).tolist(),
            "follower_kp": float(args.follower_kp),
            "follower_kd": float(args.follower_kd),
            "four_channel": bool(arch_mode == "four_channel"),
            "mirror_source": mirror_source_used,
            "mirror_blend_cmd": float(args.mirror_blend_cmd),
            "mirror_blend_actual": float(args.mirror_blend_actual),
            "mirror_blend_policy": float(args.mirror_blend_policy),
            "adm_mass": float(args.adm_mass),
            "adm_damp": float(args.adm_damp),
            "adm_stiff": float(args.adm_stiff),
            "adm_leak": float(args.adm_leak),
            "adm_delta_max": float(args.adm_delta_max),
            "admittance_source": str(args.admittance_source),
            "admittance_observer": str(args.admittance_observer),
            "leader_position_operating_mode": int(args.leader_position_operating_mode),
            "leader_position_handoff_settle_seconds": float(args.leader_position_handoff_settle_seconds),
            "leader_position_arm_settle_seconds": float(args.leader_position_arm_settle_seconds),
            "leader_position_recover_pre_handoff": bool(args.leader_position_recover_pre_handoff),
            "leader_position_recover_seconds": float(args.leader_position_recover_seconds),
            "leader_position_recover_max_delta": float(args.leader_position_recover_max_delta),
            "leader_current_limit": _int_array_metadata(args.leader_current_limit),
            "leader_goal_current": _int_array_metadata(args.leader_goal_current),
            "leader_position_p_gains": _int_array_metadata(args.leader_position_p_gains),
            "leader_position_i_gains": _int_array_metadata(args.leader_position_i_gains),
            "leader_position_d_gains": _int_array_metadata(args.leader_position_d_gains),
            "leader_velocity_p_gains": _int_array_metadata(args.leader_velocity_p_gains),
            "leader_velocity_i_gains": _int_array_metadata(args.leader_velocity_i_gains),
            "bota_enable": bool(args.bota_enable or str(args.admittance_source).startswith("bota_")),
            "bota_config": str((REPO_ROOT / args.bota_config).resolve() if not Path(args.bota_config).is_absolute() else Path(args.bota_config).resolve()),
            "bota_driver_tare": bool(args.bota_driver_tare),
            "bota_warmup_seconds": float(args.bota_warmup_seconds),
            "bota_bias_seconds": float(args.bota_bias_seconds),
            "bota_filter_cutoff": float(args.bota_filter_cutoff),
            "bota_filter_alpha": float(args.bota_filter_alpha),
            "bota_deadband": _parse_float_list(args.bota_deadband),
            "bota_saturation": _parse_float_list(args.bota_saturation),
            "bota_gravity_comp": bool(args.bota_gravity_comp),
            "bota_payload_mass": float(args.bota_payload_mass),
            "bota_payload_com": _parse_float_list(args.bota_payload_com),
            "bota_gravity_sign": float(args.bota_gravity_sign),
            "bota_frame": str(args.bota_frame),
            "bota_axis_mask": _parse_float_list(args.bota_axis_mask),
            "bota_cart_mass": _parse_float_list(args.bota_cart_mass),
            "bota_cart_damp": _parse_float_list(args.bota_cart_damp),
            "bota_cart_stiff": _parse_float_list(args.bota_cart_stiff),
            "bota_cart_max": _parse_float_list(args.bota_cart_max),
            "bota_hfvc_force_dims": int(args.bota_hfvc_force_dims),
            "bota_hfvc_stiction": _parse_float_list(args.bota_hfvc_stiction),
            "bota_wrench_sign": float(args.bota_wrench_sign),
            "bota_se3_output_mode": str(args.bota_se3_output_mode),
            "bota_se3_pose_feedback": bool(str(args.admittance_source).lower() == "bota_se3"),
            "bota_joint_gain": float(args.bota_joint_gain),
            "bota_dls_damping": float(args.bota_dls_damping),
            "bota_contact_force_scale": float(args.bota_contact_force_scale),
            "bota_contact_torque_scale": float(args.bota_contact_torque_scale),
            "admittance_inner_loop": (
                "dynamixel_position"
                if arch_mode == "leader_admittance_position"
                else (
                    "software_impedance"
                    if arch_mode == "leader_admittance_impedance"
                    else "none"
                )
            ),
            "settle_time": startup_settle_time,
            "impedance_ramp_time": float(args.impedance_ramp_time),
            "action_dt": float(args.action_dt),
            "horizon": int(args.horizon),
            "config": str(config_path),
        }
        if args.test_mode == "taskspace":
            metadata.update(
                {
                    "taskspace_motion": str(args.taskspace_motion),
                    "taskspace_amplitude": float(args.taskspace_amplitude),
                    "taskspace_axis": str(args.taskspace_axis),
                    "frequency": float(args.frequency),
                    "urdf_path": str(policy_config.get("urdf_path", "")),
                }
            )
        recorder.save(out_path, metadata=metadata)
        print(f"Saved: {out_path}")
        if recorder.dropped_samples:
            print("[WARN] recorder buffer overflowed; increase --max-duration")

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Run failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if stop_event is not None:
            stop_event.set()
        if policy_proc is not None:
            try:
                policy_proc.join(timeout=2.0)
                if policy_proc.is_alive():
                    policy_proc.terminate()
            except Exception:
                pass

        if traj_buf is not None:
            try:
                traj_buf.close()
                traj_buf.unlink()
            except Exception:
                pass
        if obs_snap is not None:
            try:
                obs_snap.close()
                obs_snap.unlink()
            except Exception:
                pass

        if system is not None:
            try:
                system.set_leader_joint_torque(np.zeros(system.num_arm_joints), 0.0)
                time.sleep(0.05)
                if system.driver is not None:
                    system.driver.set_operating_mode(3)
                time.sleep(0.05)
                system.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
