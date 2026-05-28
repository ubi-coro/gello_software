from __future__ import annotations

import math
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

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


class BotaMiniOneReader:
    """Lifecycle wrapper around the Bota MiniOne driver used for Phase-B corrections."""

    def __init__(self, config_path: Path, driver_tare: bool = False):
        try:
            import bota_driver
        except ImportError as exc:
            raise RuntimeError(
                "bota_driver is not importable. Activate the environment with the "
                "Bota Systems Python driver before running Phase B collection."
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


class BotaLinearWrenchCalibration:
    """Linear orientation-based BOTA residual compensation loaded from YAML."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.feature_mode = str(payload.get("feature_mode", "relative_rotation"))
        self.target_field = str(payload.get("target_field", "wrench_base_static_ema"))
        self.coefficients = np.asarray(payload["coefficients"], dtype=float)
        if self.coefficients.shape != (10, 6):
            raise ValueError(
                f"Expected BOTA calibration coefficients shape (10, 6), got {self.coefficients.shape}"
            )

    def features(self, r_base_sensor: np.ndarray, r_ref_runtime: np.ndarray) -> np.ndarray:
        rot = np.asarray(r_base_sensor, dtype=float).reshape(3, 3)
        if self.feature_mode == "relative_rotation":
            body = (rot - np.asarray(r_ref_runtime, dtype=float).reshape(3, 3)).reshape(9)
        elif self.feature_mode == "absolute_rotation":
            body = rot.reshape(9)
        else:
            raise ValueError(f"Unsupported BOTA calibration feature_mode: {self.feature_mode}")
        return np.concatenate([[1.0], body])

    def predict(self, r_base_sensor: np.ndarray, r_ref_runtime: np.ndarray) -> np.ndarray:
        return self.features(r_base_sensor, r_ref_runtime) @ self.coefficients


def load_bota_wrench_calibration(path: Path) -> BotaLinearWrenchCalibration:
    if yaml is None:
        raise RuntimeError("PyYAML is required for BOTA wrench calibration")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"BOTA wrench calibration not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid BOTA wrench calibration YAML: {resolved}")
    if payload.get("kind") != "bota_minione_linear_wrench_compensation":
        raise ValueError(f"Unsupported BOTA wrench calibration kind: {payload.get('kind')!r}")
    return BotaLinearWrenchCalibration(resolved, payload)


class BotaWrenchConditioner:
    """Apply software bias, frame transform, calibrated compensation, LPF, deadband, and saturation."""

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
        base_axis_map: np.ndarray | None = None,
        base_axis_signs: np.ndarray | None = None,
        wrench_calibration: BotaLinearWrenchCalibration | None = None,
    ):
        self.cutoff_hz = max(float(cutoff_hz), 0.0)
        self.alpha = float(np.clip(float(alpha), 0.0, 1.0))
        self.deadband = np.asarray(deadband, dtype=float).reshape(6)
        self.saturation = np.asarray(saturation, dtype=float).reshape(6)
        self.gravity_comp = bool(gravity_comp)
        self.payload_mass_kg = max(float(payload_mass_kg), 0.0)
        self.payload_com_sensor = np.asarray(payload_com_sensor, dtype=float).reshape(3)
        self.gravity_sign = float(gravity_sign)
        self.base_axis_map = self._validate_axis_map(base_axis_map)
        self.base_axis_signs = self._validate_axis_signs(base_axis_signs)
        self.bias_sensor = np.zeros(6, dtype=float)
        self.filtered_base = np.zeros(6, dtype=float)
        self.wrench_calibration = wrench_calibration
        self.calibration_reference_r_base_sensor = np.eye(3, dtype=float)
        self.last_calibration_prediction = np.zeros(6, dtype=float)
        self.last_calibrated_base = np.zeros(6, dtype=float)
        self.initialized = False
        self.last_filter_timestamp_us: int | None = None

    @staticmethod
    def _validate_axis_map(axis_map: np.ndarray | None) -> np.ndarray:
        if axis_map is None:
            return np.arange(6, dtype=int)
        arr = np.asarray(axis_map, dtype=int).reshape(-1)
        if arr.size != 6 or sorted(arr.tolist()) != list(range(6)):
            raise ValueError("bota.base_axis_map must be a permutation of [0,1,2,3,4,5]")
        return arr

    @staticmethod
    def _validate_axis_signs(axis_signs: np.ndarray | None) -> np.ndarray:
        if axis_signs is None:
            return np.ones(6, dtype=float)
        arr = np.asarray(axis_signs, dtype=float).reshape(-1)
        if arr.size != 6:
            raise ValueError("bota.base_axis_signs must contain 6 values")
        signs = np.sign(arr)
        signs[signs == 0.0] = 1.0
        return signs

    def _apply_base_axis_correction(self, wrench_base: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench_base, dtype=float).reshape(6)
        return self.base_axis_signs * wrench[self.base_axis_map]

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

    def set_calibration_reference(self, r_base_sensor: np.ndarray) -> None:
        self.calibration_reference_r_base_sensor = np.asarray(r_base_sensor, dtype=float).reshape(3, 3).copy()
        if self.wrench_calibration is not None:
            print(
                "[BOTA] calibrated wrench compensation reference set "
                f"feature_mode={self.wrench_calibration.feature_mode} "
                f"target={self.wrench_calibration.target_field}"
            )

    def _condition(self, wrench_base: np.ndarray) -> np.ndarray:
        conditioned = np.asarray(wrench_base, dtype=float).reshape(6).copy()
        conditioned = np.where(
            np.abs(conditioned) > self.deadband,
            conditioned - np.sign(conditioned) * self.deadband,
            0.0,
        )
        return np.clip(conditioned, -self.saturation, self.saturation)

    def update(
        self,
        wrench_sensor_raw: np.ndarray,
        r_base_sensor: np.ndarray,
        dt: float,
        timestamp_us: int | None = None,
        is_new_frame: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wrench_sensor = np.asarray(wrench_sensor_raw, dtype=float).reshape(6) - self.bias_sensor
        force_base = r_base_sensor @ wrench_sensor[:3]
        torque_base = r_base_sensor @ wrench_sensor[3:]
        wrench_base = self._apply_base_axis_correction(np.concatenate([force_base, torque_base]))

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

        calibration_prediction = np.zeros(6, dtype=float)
        calibrated_base = self.filtered_base.copy()
        if self.wrench_calibration is not None:
            calibration_prediction = self.wrench_calibration.predict(
                r_base_sensor, self.calibration_reference_r_base_sensor
            )
            calibrated_base = self.filtered_base - calibration_prediction
        self.last_calibration_prediction = calibration_prediction.copy()
        self.last_calibrated_base = calibrated_base.copy()
        conditioned = self._condition(calibrated_base)
        return wrench_base, conditioned, calibrated_base, calibration_prediction


class GelloTaskspaceKinematics:
    """Pinocchio FK/Jacobian helper using base-aligned frame coordinates."""

    def __init__(self, system: Any, n_joints: int, frame_name: str | None = None):
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


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=float)
    if theta < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * math.sin(theta)) * vee


def _pose_error_base(pos_current: np.ndarray, rot_current: np.ndarray, pos_ref: np.ndarray, rot_ref: np.ndarray) -> np.ndarray:
    pos_err = np.asarray(pos_current, dtype=float).reshape(3) - np.asarray(pos_ref, dtype=float).reshape(3)
    rot_err = _so3_log(np.asarray(rot_current, dtype=float).reshape(3, 3) @ np.asarray(rot_ref, dtype=float).reshape(3, 3).T)
    return np.concatenate([pos_err, rot_err])


class SE3PoseAdmittance6D:
    """6D wrench admittance producing a bounded SE(3) task-space offset."""

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
        pose_error = _pose_error_base(pos_current, rot_current, pos_ref, rot_ref) * self.axis_mask
        wrench = self.wrench_sign * np.asarray(wrench_base, dtype=float).reshape(6) * self.axis_mask

        if self.output_mode == "pose_error":
            spring_state = pose_error
            wrench_all = wrench - self.stiffness * spring_state - self.damping * self.velocity
            command_base = pose_error
        else:
            wrench_all = wrench - self.stiffness * self.offset - self.damping * self.velocity
            command_base = self.offset

        wrench_all *= self.axis_mask
        wrench_all = np.where(
            np.abs(wrench_all) > self.stiction,
            wrench_all - np.sign(wrench_all) * self.stiction,
            0.0,
        )
        ddx = wrench_all / self.mass
        self.velocity = (1.0 - self.leak * dt_s) * (self.velocity + ddx * dt_s)
        self.velocity *= self.axis_mask
        command_error = command_base + self.velocity * dt_s
        command_error = np.clip(command_error, -self.max_offset, self.max_offset) * self.axis_mask
        if self.output_mode == "offset":
            self.offset = command_error.copy()
        elif dt_s > 0.0:
            self.velocity = (command_error - pose_error) / dt_s * self.axis_mask
        return command_error.copy(), pose_error.copy()


def damped_least_squares_delta_q(jacobian: np.ndarray, task_delta: np.ndarray, damping: float) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    delta = np.asarray(task_delta, dtype=float).reshape(6)
    lam2 = max(float(damping), 0.0) ** 2
    lhs = jac @ jac.T + lam2 * np.eye(6)
    return jac.T @ np.linalg.solve(lhs, delta)


def bota_contact_probability(wrench_base: np.ndarray, force_scale: float, torque_scale: float) -> tuple[float, float]:
    force_norm = float(np.linalg.norm(wrench_base[:3])) / max(float(force_scale), 1e-6)
    torque_norm = float(np.linalg.norm(wrench_base[3:])) / max(float(torque_scale), 1e-6)
    probability = float(np.clip(max(force_norm, torque_norm), 0.0, 1.0))
    if probability >= 0.6:
        return probability, 2.0
    if probability >= 0.25:
        return probability, 1.0
    return probability, 0.0


def intervention_flags(
    contact_probability: float,
    delta: np.ndarray,
    contact_threshold: float,
    delta_threshold: float,
) -> tuple[float, float]:
    contact_active = float(contact_threshold) > 0.0 and float(contact_probability) >= float(contact_threshold)
    delta_active = float(delta_threshold) > 0.0 and float(np.linalg.norm(np.asarray(delta, dtype=float))) >= float(delta_threshold)
    source = 0.0
    if contact_active:
        source += 1.0
    if delta_active:
        source += 2.0
    return (1.0 if source > 0.0 else 0.0), source


class DeltaResidualLimiter:
    """Post-filter for bounded human residual corrections."""

    def __init__(
        self,
        n: int,
        delta_max: np.ndarray,
        rate_max: np.ndarray,
        release_rate_max: np.ndarray,
        release_tau_s: float,
        release_contact_threshold: float,
    ):
        self.n = int(n)
        self.delta_max = np.maximum(np.asarray(delta_max, dtype=float).reshape(self.n), 1e-9)
        self.rate_max = np.maximum(np.asarray(rate_max, dtype=float).reshape(self.n), 0.0)
        self.release_rate_max = np.maximum(np.asarray(release_rate_max, dtype=float).reshape(self.n), 0.0)
        self.release_tau_s = max(float(release_tau_s), 0.0)
        self.release_contact_threshold = max(float(release_contact_threshold), 0.0)
        self.delta = np.zeros(self.n, dtype=float)

    def reset(self, value: np.ndarray | None = None) -> None:
        if value is None:
            self.delta.fill(0.0)
        else:
            self.delta = np.clip(np.asarray(value, dtype=float).reshape(self.n), -self.delta_max, self.delta_max)

    def step(self, raw_delta: np.ndarray, dt: float, contact_probability: float) -> np.ndarray:
        target = np.clip(np.asarray(raw_delta, dtype=float).reshape(self.n), -self.delta_max, self.delta_max)
        dt_s = max(float(dt), 0.0)
        if self.release_tau_s > 0.0 and float(contact_probability) < self.release_contact_threshold:
            alpha = 1.0 - math.exp(-dt_s / max(self.release_tau_s, 1e-9))
            target = (1.0 - alpha) * self.delta
        if dt_s > 0.0:
            delta_step = target - self.delta
            rate = self.rate_max.copy()
            relaxing = np.abs(target) < np.abs(self.delta)
            release_rate = np.where(self.release_rate_max > 0.0, self.release_rate_max, rate)
            rate = np.where(relaxing, release_rate, rate)
            limited = rate > 0.0
            if np.any(limited):
                delta_step = np.where(limited, np.clip(delta_step, -rate * dt_s, rate * dt_s), delta_step)
            self.delta = self.delta + delta_step
        else:
            self.delta = target
        self.delta = np.clip(self.delta, -self.delta_max, self.delta_max)
        return self.delta.copy()


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def map_leader_delta_to_follower_delta(system: Any, delta_leader: np.ndarray, n: int) -> np.ndarray:
    delta = np.asarray(delta_leader[:n], dtype=float)
    if getattr(system, "map_index", None) is None or getattr(system, "map_signs", None) is None:
        return delta.copy()
    lim = min(len(system.map_index), len(system.map_signs), n)
    delta_follower = np.zeros(n, dtype=float)
    for follower_idx in range(lim):
        leader_idx = int(system.map_index[follower_idx])
        if 0 <= leader_idx < len(delta):
            delta_follower[follower_idx] = float(system.map_signs[follower_idx]) * delta[leader_idx]
    return delta_follower

def make_safe_leader_mirror_target(
    system: Any,
    q_cmd_leader_raw: np.ndarray,
    q_leader: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a turn-safe, limit-aware GELLO mirror target.

    The UR command path remains independent. This only prevents the haptic
    mirror from chasing an equivalent but bad GELLO turn branch. Limits are
    applied only when the current decoded joint is already plausibly inside
    that configured limit interval; this avoids clamping joints whose static
    limits are on a different 2pi branch than the current boot calibration.
    """
    target_raw = np.asarray(q_cmd_leader_raw[:n], dtype=float)
    reference = np.asarray(q_leader[:n], dtype=float)
    target_safe = reference + wrap_angle_delta(target_raw - reference)

    active = np.zeros(n, dtype=bool)
    lower = getattr(system, "arm_joint_limits_min", None)
    upper = getattr(system, "arm_joint_limits_max", None)
    margin = float(getattr(system, "safety_margin", 0.0) or 0.0)
    if lower is not None and upper is not None:
        lo_arr = np.asarray(lower, dtype=float).reshape(-1)
        hi_arr = np.asarray(upper, dtype=float).reshape(-1)
        lim = min(n, lo_arr.size, hi_arr.size)
        for i in range(lim):
            lo = float(lo_arr[i]) + margin
            hi = float(hi_arr[i]) - margin
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                continue
            # Only trust the static limit interval if the current decoded
            # joint is on the same configured branch. Some GELLO limits are
            # intentionally broader/multi-turn and not all are branch-aligned.
            if lo - margin <= reference[i] <= hi + margin:
                clipped = float(np.clip(target_safe[i], lo, hi))
                if abs(clipped - target_safe[i]) > 1e-9:
                    active[i] = True
                target_safe[i] = clipped

    safety_delta = target_safe - target_raw
    active |= np.abs(safety_delta) > 1e-6
    return target_safe, safety_delta, active

def _as_list(value: Any, n: int, fill: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n, fill, dtype=float)
    if isinstance(value, str):
        raw = [float(v.strip()) for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw = [float(v) for v in value]
    else:
        raw = [float(value)]
    if not raw:
        return np.full(n, fill, dtype=float)
    if len(raw) == 1:
        return np.full(n, raw[0], dtype=float)
    if len(raw) < n:
        raw = raw + [fill] * (n - len(raw))
    return np.asarray(raw[:n], dtype=float)


def _as_int_list(value: Any, n: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        raw = [int(round(float(v.strip()))) for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw = [int(round(float(v))) for v in value]
    else:
        raw = [int(round(float(value)))]
    if not raw:
        return None
    if len(raw) == 1:
        raw = raw * n
    if len(raw) < n:
        raw = raw + [raw[-1]] * (n - len(raw))
    return np.asarray(raw[:n], dtype=np.int32)


def _readback_2byte(driver: Any, address: int, signed: bool = False) -> list[int]:
    if not hasattr(driver, "read_control_table_2byte"):
        return []
    try:
        return [int(v) for v in driver.read_control_table_2byte(address, signed=signed).tolist()]
    except Exception as exc:
        print(f"[DXL] readback failed for register {address}: {exc}")
        return []



def _recover_leader_position_after_handoff(
    system: FACTRGravityCompensation,
    target_raw_pre_handoff: np.ndarray,
    current_raw: np.ndarray,
    phase_b_cfg: dict[str, Any],
) -> np.ndarray:
    leader_cfg = phase_b_cfg.get("leader_position", {})
    n = int(system.num_arm_joints)
    duration = max(float(leader_cfg.get("recover_s", 0.0)), 0.0)
    max_delta = max(float(leader_cfg.get("recover_max_delta", 0.0)), 0.0)
    if duration <= 0.0:
        return np.asarray(current_raw, dtype=float).copy()

    target_raw_pre_handoff = np.asarray(target_raw_pre_handoff, dtype=float)
    current_raw = np.asarray(current_raw, dtype=float)
    if target_raw_pre_handoff.shape[0] != current_raw.shape[0]:
        print("[DXL] pre-handoff recovery skipped: target/current length mismatch")
        return current_raw.copy()

    delta_raw = np.zeros_like(current_raw)
    delta_raw[:n] = wrap_angle_delta(target_raw_pre_handoff[:n] - current_raw[:n])
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
        system.driver.set_joints((start_raw + alpha * (target_raw - start_raw)).tolist())
        time.sleep(max(float(system.dt), 0.001))

    raw_after = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_after.shape[0] != current_raw.shape[0]:
        raw_after = target_raw
    system.driver.set_joints(raw_after.tolist())
    return raw_after


def _switch_leader_to_position_hold(system: FACTRGravityCompensation, phase_b_cfg: dict[str, Any]) -> np.ndarray:
    """Switch GELLO to the validated position-hold handoff and return the rebased raw pose."""
    if system.driver is None:
        return np.zeros(int(system.num_motors), dtype=float)

    leader_cfg = phase_b_cfg.get("leader_position", {})
    raw_pos_now = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_now.shape[0] != int(system.num_motors):
        raise RuntimeError(
            "Unexpected Dynamixel state length during position-mode handoff "
            f"(got {raw_pos_now.shape[0]}, expected {system.num_motors})"
        )

    num_motors = int(system.num_motors)
    operating_mode = int(leader_cfg.get("operating_mode", 3))
    if operating_mode not in (3, 5):
        raise ValueError("Phase-B leader_position.operating_mode must be 3 or 5")

    current_limit = _as_int_list(leader_cfg.get("current_limit"), num_motors)
    goal_current = _as_int_list(leader_cfg.get("goal_current"), num_motors)
    pos_p = _as_int_list(leader_cfg.get("position_p_gains"), num_motors)
    pos_i = _as_int_list(leader_cfg.get("position_i_gains"), num_motors)
    pos_d = _as_int_list(leader_cfg.get("position_d_gains"), num_motors)
    vel_p = _as_int_list(leader_cfg.get("velocity_p_gains"), num_motors)
    vel_i = _as_int_list(leader_cfg.get("velocity_i_gains"), num_motors)

    setattr(system, "_leader_position_pre_handoff_raw", raw_pos_now.copy())
    system.driver.set_torque_mode(False)
    time.sleep(0.02)
    system.driver.set_operating_mode(operating_mode)
    time.sleep(0.02)
    if hasattr(system.driver, "verify_operating_mode"):
        system.driver.verify_operating_mode(operating_mode)
    if current_limit is not None and hasattr(system.driver, "set_current_limits"):
        system.driver.set_current_limits(current_limit.tolist())
        time.sleep(0.01)
    if operating_mode == 5 and goal_current is None and current_limit is not None:
        goal_current = current_limit.copy()
    if goal_current is not None and hasattr(system.driver, "set_goal_currents_raw"):
        system.driver.set_goal_currents_raw(goal_current.tolist())
        time.sleep(0.01)
    if any(v is not None for v in (pos_p, pos_i, pos_d)) and hasattr(system.driver, "set_position_pid_gains"):
        system.driver.set_position_pid_gains(
            p_gains=None if pos_p is None else pos_p.tolist(),
            i_gains=None if pos_i is None else pos_i.tolist(),
            d_gains=None if pos_d is None else pos_d.tolist(),
        )
        time.sleep(0.01)
    if any(v is not None for v in (vel_p, vel_i)) and hasattr(system.driver, "set_velocity_pi_gains"):
        system.driver.set_velocity_pi_gains(
            p_gains=None if vel_p is None else vel_p.tolist(),
            i_gains=None if vel_i is None else vel_i.tolist(),
        )
        time.sleep(0.01)

    print(f"[DXL] leader position handoff operating_mode={operating_mode}")
    for name, values in {
        "current_limit": _readback_2byte(system.driver, ADDR_CURRENT_LIMIT),
        "goal_current": _readback_2byte(system.driver, ADDR_GOAL_CURRENT, signed=True),
        "pos_p": _readback_2byte(system.driver, ADDR_POSITION_P_GAIN),
        "pos_i": _readback_2byte(system.driver, ADDR_POSITION_I_GAIN),
        "pos_d": _readback_2byte(system.driver, ADDR_POSITION_D_GAIN),
        "vel_p": _readback_2byte(system.driver, ADDR_VELOCITY_P_GAIN),
        "vel_i": _readback_2byte(system.driver, ADDR_VELOCITY_I_GAIN),
    }.items():
        if values:
            print(f"[DXL] {name}: {values}")

    raw_pos_post_mode = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_post_mode.shape[0] != int(system.num_motors):
        raw_pos_post_mode = raw_pos_now
    handoff_motion = wrap_angle_delta(raw_pos_post_mode[: system.num_arm_joints] - raw_pos_now[: system.num_arm_joints])
    print(
        "[DXL] handoff pre-torque drift "
        f"max={float(np.max(np.abs(handoff_motion))):.4f}rad "
        f"mean={float(np.mean(np.abs(handoff_motion))):.4f}rad"
    )

    # Preload the observed post-switch pose before torque is re-enabled. This
    # prevents the position controller from pulling the arm back through a
    # synthetic correction during collection startup.
    if hasattr(system.driver, "write_goal_positions_unchecked"):
        system.driver.write_goal_positions_unchecked(raw_pos_post_mode.tolist())
    else:
        system.driver.set_torque_mode(True)
        time.sleep(0.005)
        system.driver.set_joints(raw_pos_post_mode.tolist())
        time.sleep(0.005)
        raw_pos_post_mode = np.asarray(system.driver.get_joints(), dtype=float)
        system.driver.set_torque_mode(False)
        time.sleep(0.005)
    system.driver.set_torque_mode(True)
    time.sleep(0.02)

    settle_s = max(float(leader_cfg.get("handoff_settle_s", 0.0)), 0.0)
    if settle_s > 0.0:
        print(f"[DXL] settling position hold for {settle_s:.2f}s before rebase")
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < settle_s:
            time.sleep(min(0.02, max(0.0, settle_s - (time.perf_counter() - t_settle))))

    raw_pos_after = np.asarray(system.driver.get_joints(), dtype=float)
    if raw_pos_after.shape[0] != int(system.num_motors):
        return raw_pos_post_mode
    post_enable_motion = wrap_angle_delta(raw_pos_after[: system.num_arm_joints] - raw_pos_post_mode[: system.num_arm_joints])
    print(
        "[DXL] handoff post-enable drift "
        f"max={float(np.max(np.abs(post_enable_motion))):.4f}rad "
        f"mean={float(np.mean(np.abs(post_enable_motion))):.4f}rad"
    )
    system.driver.set_joints(raw_pos_after.tolist())
    time.sleep(0.02)
    return raw_pos_after


def _start_phase_b_follower_command_thread(
    system: FACTRGravityCompensation,
    initial_target: np.ndarray,
    kp: float,
    kd: float,
    handoff_ramp_s: float = 0.0,
) -> tuple[Event, Lock, dict[str, Any], Thread]:
    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("Phase B requires teleop.enable=true with direct RTDE follower control")

    follower = system._direct_follower_robot
    stop_event = Event()
    state_lock = Lock()
    q_target_initial = np.asarray(initial_target, dtype=float).copy()
    q_start = q_target_initial.copy()
    q_error = np.zeros_like(q_target_initial)
    can_ramp = False
    state: dict[str, Any] = {
        "q": q_target_initial.copy(),
        "failed": False,
        "failure_reason": "",
        "failures": 0,
    }

    try:
        q_follower_now, _ = system.get_follower_arm_state()
        q_start = np.asarray(q_follower_now[: len(q_target_initial)], dtype=float).copy()
        q_error = q_target_initial - q_start
        can_ramp = q_start.shape == q_target_initial.shape
        print(
            "[FOLLOWER] initial command error "
            f"max={float(np.max(np.abs(q_error))):.4f}rad "
            f"mean={float(np.mean(np.abs(q_error))):.4f}rad"
        )
        if can_ramp and float(handoff_ramp_s) > 0.0:
            state["q"] = q_start.copy()
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
                    kp=float(kp),
                    kd=float(kd),
                )
                if ok is False:
                    failures += 1
                    with state_lock:
                        state["failed"] = True
                        state["failure_reason"] = "directTorque returned False"
                        state["failures"] = int(failures)
                    now = time.monotonic()
                    if now - last_warn > 1.0:
                        last_warn = now
                        print(f"[FOLLOWER] directTorque returned False ({failures} total)")
                    stop_event.set()
                    break
            except Exception as exc:
                failures += 1
                with state_lock:
                    state["failed"] = True
                    state["failure_reason"] = f"follower command failed: {exc}"
                    state["failures"] = int(failures)
                now = time.monotonic()
                if now - last_warn > 1.0:
                    last_warn = now
                    print(f"[FOLLOWER] command failed ({failures} total): {exc}")
                stop_event.set()
                break
            sleep_s = dt_thread - (time.perf_counter() - loop_t0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    thread = Thread(target=_worker, daemon=True, name="phase-b-follower-command")
    thread.start()
    print(f"[FOLLOWER] direct command thread started (500Hz, Kp={float(kp):.1f}, Kd={float(kd):.1f})")

    ramp_s = max(float(handoff_ramp_s), 0.0)
    if can_ramp and ramp_s > 0.0 and np.max(np.abs(q_error)) > 1e-6:
        print(f"[FOLLOWER] ramping initial target over {ramp_s:.2f}s before arming")
        t0 = time.perf_counter()
        while not stop_event.is_set():
            alpha = min((time.perf_counter() - t0) / ramp_s, 1.0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            with state_lock:
                state["q"] = q_start + smooth * (q_target_initial - q_start)
            if alpha >= 1.0:
                break
            time.sleep(0.002)
    with state_lock:
        state["q"] = q_target_initial.copy()
    return stop_event, state_lock, state, thread


def run_phase_b_bota_se3_loop(
    system: FACTRGravityCompensation,
    traj_interp: TrajectoryInterpolator,
    state_cache: dict[str, Any],
    stop_event: Event,
    phase_b_cfg: dict[str, Any],
    repo_root: Path,
) -> None:
    """Validated Phase-B loop: q_cmd_ur5e = q_ref_policy + delta_human."""
    n = int(system.num_arm_joints)
    dt = float(system.dt)
    bota_cfg = phase_b_cfg.get("bota", {})
    residual_cfg = phase_b_cfg.get("residual", {})
    follower_cfg = phase_b_cfg.get("follower", {})
    intervention_cfg = phase_b_cfg.get("intervention", {})

    bota_path = Path(str(bota_cfg.get("config", "configs/bota_binary.json")))
    if not bota_path.is_absolute():
        bota_path = (repo_root / bota_path).resolve()

    wrench_calibration: BotaLinearWrenchCalibration | None = None
    if bool(bota_cfg.get("wrench_calibration_enable", False)):
        calibration_path = Path(str(bota_cfg.get("wrench_calibration", "")))
        if not str(calibration_path):
            raise ValueError("phase_b.bota.wrench_calibration_enable requires bota.wrench_calibration")
        if not calibration_path.is_absolute():
            calibration_path = (repo_root / calibration_path).resolve()
        wrench_calibration = load_bota_wrench_calibration(calibration_path)
        print(f"[BOTA] loaded calibrated wrench compensation: {wrench_calibration.path}")

    bota_reader: BotaMiniOneReader | None = None
    follower_stop: Event | None = None
    follower_thread: Thread | None = None
    latest_raw = np.zeros(int(system.num_motors), dtype=float)

    try:
        kin = GelloTaskspaceKinematics(system, n, str(bota_cfg.get("frame", "") or "") or None)
        admittance = SE3PoseAdmittance6D(
            mass=_as_list(bota_cfg.get("cart_mass"), 6, 2.0),
            damping=_as_list(bota_cfg.get("cart_damp"), 6, 18.0),
            stiffness=_as_list(bota_cfg.get("cart_stiff"), 6, 8.0),
            leak=0.02,
            max_offset=_as_list(bota_cfg.get("cart_max"), 6, 0.03),
            axis_mask=_as_list(bota_cfg.get("axis_mask"), 6, 1.0),
            stiction=_as_list(bota_cfg.get("stiction"), 6, 0.0),
            wrench_sign=float(bota_cfg.get("wrench_sign", 1.0)),
            output_mode=str(bota_cfg.get("output_mode", "offset")),
        )
        limiter = DeltaResidualLimiter(
            n=n,
            delta_max=_as_list(residual_cfg.get("delta_max"), n, 0.06),
            rate_max=_as_list(residual_cfg.get("delta_rate_max"), n, 0.0),
            release_rate_max=_as_list(residual_cfg.get("delta_release_rate_max"), n, 0.0),
            release_tau_s=float(residual_cfg.get("delta_release_tau_s", 0.0)),
            release_contact_threshold=float(residual_cfg.get("delta_release_contact_threshold", 0.05)),
        )
        conditioner = BotaWrenchConditioner(
            cutoff_hz=float(bota_cfg.get("filter_cutoff_hz", 25.0)),
            alpha=float(bota_cfg.get("filter_alpha", 0.0)),
            deadband=_as_list(bota_cfg.get("deadband"), 6, 0.0),
            saturation=_as_list(bota_cfg.get("saturation"), 6, 25.0),
            gravity_comp=bool(bota_cfg.get("gravity_comp", False)),
            payload_mass_kg=float(bota_cfg.get("payload_mass_kg", 0.070)),
            payload_com_sensor=_as_list(bota_cfg.get("payload_com_sensor_m"), 3, 0.0),
            gravity_sign=float(bota_cfg.get("gravity_sign", 1.0)),
            base_axis_map=_as_int_list(bota_cfg.get("base_axis_map"), 6),
            base_axis_signs=_as_list(bota_cfg.get("base_axis_signs"), 6, 1.0),
            wrench_calibration=wrench_calibration,
        )
        print(
            "[BOTA] base-frame axis correction "
            f"map={conditioner.base_axis_map.tolist()} signs={conditioner.base_axis_signs.tolist()}"
        )

        bota_reader = BotaMiniOneReader(bota_path, driver_tare=bool(bota_cfg.get("driver_tare", False)))
        bota_reader.start()

        warmup_s = max(float(bota_cfg.get("warmup_s", 1.0)), 0.0)
        if warmup_s > 0.0:
            print(f"[BOTA] warmup/discarding frames for {warmup_s:.2f}s")
            t_warm = time.perf_counter()
            while not stop_event.is_set() and time.perf_counter() - t_warm < warmup_s:
                bota_reader.read_latest()
                time.sleep(min(dt, 0.0025))
            print(f"[BOTA] warmup frames new={bota_reader.new_frames} duplicate={bota_reader.duplicate_frames}")

        bias_s = max(float(bota_cfg.get("bias_s", 2.0)), 0.0)
        if bias_s > 0.0:
            print(f"[BOTA] collecting software bias for {bias_s:.2f}s")
            samples: list[np.ndarray] = []
            t_bias = time.perf_counter()
            while not stop_event.is_set() and time.perf_counter() - t_bias < bias_s:
                wrench_raw, *_ = bota_reader.read_latest()
                samples.append(wrench_raw.copy())
                time.sleep(min(dt, 0.0025))
            conditioner.set_bias(samples)

        if wrench_calibration is not None:
            pos_ref_calib, _vel_ref_calib, _cur_ref_calib = system.driver.get_positions_velocities_and_currents()
            q_raw_ref_calib = np.asarray(pos_ref_calib[:n], dtype=float)
            q_ref_calib = (q_raw_ref_calib - system.joint_offsets[:n]) * system.joint_signs[:n]
            _pos_calib, rot_calib, _jac_calib = kin.pose_and_jacobian(q_ref_calib)
            conditioner.set_calibration_reference(rot_calib)

        raw_hold = _switch_leader_to_position_hold(system, phase_b_cfg)
        if bool(phase_b_cfg.get("leader_position", {}).get("recover_pre_handoff", False)):
            pre = getattr(system, "_leader_position_pre_handoff_raw", raw_hold)
            raw_hold = _recover_leader_position_after_handoff(system, pre, raw_hold, phase_b_cfg)
        latest_raw = raw_hold.copy()
        q_hold = (raw_hold[:n] - system.joint_offsets[:n]) * system.joint_signs[:n]
        traj_interp.traj_buf.write(np.tile(q_hold, (traj_interp.traj_buf.horizon, 1)), time.monotonic())
        traj_interp.fallback_q = q_hold.copy()

        policy_action_hold = system._build_follower_action(q_hold, 0.0)
        q_follower_target = np.asarray(policy_action_hold[:n], dtype=float)
        follower_stop, follower_lock, follower_state, follower_thread = _start_phase_b_follower_command_thread(
            system=system,
            initial_target=q_follower_target,
            kp=float(follower_cfg.get("kp", 190.0)),
            kd=float(follower_cfg.get("kd", 15.0)),
            handoff_ramp_s=float(follower_cfg.get("handoff_ramp_s", 0.6)),
        )

        arm_settle_s = max(float(phase_b_cfg.get("leader_position", {}).get("arm_settle_s", 0.5)), 0.0)
        if arm_settle_s > 0.0:
            print(f"[ARMING] settling {arm_settle_s:.2f}s after handoff/follower start before recording")
            t_arm = time.perf_counter()
            while not stop_event.is_set() and time.perf_counter() - t_arm < arm_settle_s:
                time.sleep(0.01)

        with follower_lock:
            follower_failed = bool(follower_state.get("failed", False))
            follower_failure_reason = str(follower_state.get("failure_reason", ""))
            follower_failures = int(follower_state.get("failures", 0))
        if follower_failed:
            reason = follower_failure_reason or "follower directTorque command failed during arming"
            with state_cache["lock"]:
                state_cache["phase_b_error"] = reason
                state_cache["phase_b_fatal"] = True
                state_cache["phase_b_fatal_reason"] = reason
                state_cache["follower_command_failures"] = int(follower_failures)
            print(f"[PHASE B] fatal follower command failure before arming: {reason}")
            stop_event.set()
            return

        with state_cache["lock"]:
            state_cache["phase_b_ready"] = True
        print("[PHASE B] BOTA-SE3 residual loop armed")

        last_warn = 0.0
        last_loop_t0: float | None = None
        while not stop_event.is_set():
            loop_t0 = time.perf_counter()
            phase_b_loop_dt_s = float("nan") if last_loop_t0 is None else loop_t0 - last_loop_t0
            last_loop_t0 = loop_t0
            t_mono = time.monotonic()

            try:
                pos, vel, cur = system.driver.get_positions_velocities_and_currents()
                q_raw = np.asarray(pos[:n], dtype=float)
                dq_raw = np.asarray(vel[:n], dtype=float)
                latest_raw = np.asarray(pos, dtype=float)
                q_leader = (q_raw - system.joint_offsets[:n]) * system.joint_signs[:n]
                dq_leader = dq_raw * system.joint_signs[:n]
                gripper_raw = float(pos[n]) if len(pos) > n else 0.0
                gripper = (gripper_raw - system.joint_offsets[n]) * system.joint_signs[n] if len(pos) > n else 0.0
            except Exception as exc:
                now = time.monotonic()
                if now - last_warn > 1.0:
                    last_warn = now
                    print(f"[DXL] read failed in Phase B loop: {exc}")
                time.sleep(dt)
                continue

            q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
            q_ref = np.asarray(q_ref[:n], dtype=float)
            dq_ref = np.asarray(dq_ref[:n], dtype=float)

            pos_ref, rot_ref, _ = kin.pose_and_jacobian(q_ref)
            pos_cur, rot_cur, jac_cur = kin.pose_and_jacobian(q_leader)
            wrench_raw, status_flags, timestamp_us, temp_c, _, is_new = bota_reader.read_latest()
            wrench_base_raw, wrench_base, wrench_base_calibrated, wrench_calibration_prediction = conditioner.update(
                wrench_raw,
                rot_cur,
                dt,
                timestamp_us=timestamp_us,
                is_new_frame=is_new,
            )
            contact_prob, contact_gate = bota_contact_probability(
                wrench_base,
                force_scale=float(bota_cfg.get("contact_force_scale", 8.0)),
                torque_scale=float(bota_cfg.get("contact_torque_scale", 0.35)),
            )
            task_delta, pose_error = admittance.step(pos_ref, rot_ref, pos_cur, rot_cur, wrench_base, dt)
            delta_raw = damped_least_squares_delta_q(jac_cur, task_delta, float(bota_cfg.get("dls_damping", 0.10)))
            delta_raw = np.clip(delta_raw[:n], -limiter.delta_max, limiter.delta_max)
            delta_leader = limiter.step(delta_raw, dt, contact_prob)

            q_cmd_leader_raw = q_ref + delta_leader
            q_cmd_leader, leader_mirror_safety_delta, leader_mirror_safety_active = make_safe_leader_mirror_target(
                system, q_cmd_leader_raw, q_leader, n
            )
            target_hw = latest_raw.copy()
            target_hw[:n] = q_cmd_leader / system.joint_signs[:n] + system.joint_offsets[:n]
            if target_hw.shape[0] > n:
                target_hw[n] = gripper_raw
            try:
                system.driver.set_joints(target_hw.tolist())
            except Exception as exc:
                now = time.monotonic()
                if now - last_warn > 1.0:
                    last_warn = now
                    print(f"[DXL] leader position command failed: {exc}")

            policy_action = system._build_follower_action(q_ref, gripper)
            compliant_action = np.asarray(policy_action, dtype=float).copy()
            delta_follower = map_leader_delta_to_follower_delta(system, delta_leader, n)
            compliant_action[:n] = np.asarray(policy_action[:n], dtype=float) + delta_follower
            with follower_lock:
                follower_state["q"] = compliant_action[:n].copy()
                follower_failed = bool(follower_state.get("failed", False))
                follower_failure_reason = str(follower_state.get("failure_reason", ""))
                follower_failures = int(follower_state.get("failures", 0))

            if follower_failed:
                reason = follower_failure_reason or "follower directTorque command failed"
                with state_cache["lock"]:
                    state_cache["phase_b_error"] = reason
                    state_cache["phase_b_fatal"] = True
                    state_cache["phase_b_fatal_reason"] = reason
                    state_cache["follower_command_failures"] = int(follower_failures)
                print(f"[PHASE B] fatal follower command failure: {reason}")
                stop_event.set()
                break

            try:
                q_follower, dq_follower = system.get_follower_arm_state()
                q_follower = np.asarray(q_follower[:n], dtype=float)
                dq_follower = np.asarray(dq_follower[:n], dtype=float)
            except Exception:
                q_follower = np.zeros(n, dtype=float)
                dq_follower = np.zeros(n, dtype=float)

            active, source = intervention_flags(
                contact_probability=contact_prob,
                delta=delta_leader,
                contact_threshold=float(intervention_cfg.get("contact_threshold", 0.25)),
                delta_threshold=float(intervention_cfg.get("delta_threshold", 0.01)),
            )
            tau_ext = jac_cur.T @ wrench_base
            phase_b_compute_dt_s = time.perf_counter() - loop_t0
            phase_b_sleep_s = max(0.0, dt - phase_b_compute_dt_s)
            phase_b_overrun = phase_b_compute_dt_s > dt
            policy_action_age_s = float(getattr(traj_interp, "last_action_age_s", float("nan")))
            policy_inference_dt_s = float(getattr(traj_interp, "last_policy_inference_dt_s", float("nan")))
            policy_trajectory_t_write = float(getattr(traj_interp, "last_t_write", 0.0))
            policy_trajectory_is_new = bool(getattr(traj_interp, "last_is_new", False))
            diag = {
                "votes": {
                    "wrench": bool(source in (1.0, 3.0)),
                    "delta_q": bool(source in (2.0, 3.0)),
                    "energy": False,
                    "torque": False,
                },
                "timing": {
                    "phase_b_loop_dt_s": float(phase_b_loop_dt_s),
                    "phase_b_compute_dt_s": float(phase_b_compute_dt_s),
                    "phase_b_sleep_s": float(phase_b_sleep_s),
                    "phase_b_overrun": bool(phase_b_overrun),
                    "phase_b_target_dt_s": float(dt),
                    "policy_action_age_s": float(policy_action_age_s),
                    "policy_inference_dt_s": float(policy_inference_dt_s),
                    "control_loop_hz": float(1.0 / phase_b_loop_dt_s) if phase_b_loop_dt_s > 0.0 else float("nan"),
                    "policy_trajectory_t_write": float(policy_trajectory_t_write),
                    "policy_trajectory_is_new": bool(policy_trajectory_is_new),
                    "reference_stale": bool(is_stale),
                    "leader_mirror_safety_any": bool(np.any(leader_mirror_safety_active)),
                },
                "contact_probability": float(contact_prob),
                "contact_gate": float(contact_gate),
                "intervention_source": float(source),
                "bota_status_flags": status_flags.copy(),
                "bota_temperature_c": float(temp_c),
                "bota_wrench_calibrated": bool(wrench_calibration is not None),
            }

            with state_cache["lock"]:
                state_cache.update(
                    {
                        "updated": True,
                        "t_mono": t_mono,
                        "q_leader": q_leader.copy(),
                        "dq_leader": dq_leader.copy(),
                        "grip": float(gripper),
                        "grip_raw": float(gripper_raw),
                        "tau_ext": tau_ext.copy(),
                        "q_ref": q_ref.copy(),
                        "dq_ref": dq_ref.copy(),
                        "is_stale": bool(is_stale),
                        "is_correction": bool(active > 0.0),
                        "intervention_active": float(active),
                        "intervention_source": float(source),
                        "detector_diag": diag,
                        "policy_action": np.asarray(policy_action, dtype=float).copy(),
                        "compliant_action": compliant_action.copy(),
                        "q_follower": q_follower.copy(),
                        "dq_follower": dq_follower.copy(),
                        "gripper_follower": float(gripper),
                        "wrench_ur5e": wrench_base.copy(),
                        "wrench_sensor_raw": wrench_raw.copy(),
                        "wrench_base_raw": wrench_base_raw.copy(),
                        "wrench_base": wrench_base.copy(),
                        "wrench_base_calibrated": wrench_base_calibrated.copy(),
                        "wrench_calibration_prediction": wrench_calibration_prediction.copy(),
                        "wrench_bota_raw": wrench_base_raw.copy(),
                        "task_delta": task_delta.copy(),
                        "task_pose_error": pose_error.copy(),
                        "delta_leader": delta_leader.copy(),
                        "delta_leader_raw": delta_raw.copy(),
                        "q_cmd_leader": q_cmd_leader.copy(),
                        "q_cmd_leader_raw": q_cmd_leader_raw.copy(),
                        "leader_mirror_safety_delta": leader_mirror_safety_delta.copy(),
                        "leader_mirror_safety_active": leader_mirror_safety_active.copy(),
                        "delta_human": delta_leader.copy(),
                        "delta_human_raw": delta_raw.copy(),
                        "q_cmd_ur5e": compliant_action[:n].copy(),
                        "epsilon_leader": wrap_angle_delta(q_leader - q_cmd_leader),
                        "epsilon_ur5e": q_follower - compliant_action[:n],
                        "phase_b_loop_dt_s": float(phase_b_loop_dt_s),
                        "phase_b_compute_dt_s": float(phase_b_compute_dt_s),
                        "phase_b_sleep_s": float(phase_b_sleep_s),
                        "phase_b_overrun": bool(phase_b_overrun),
                        "policy_action_age_s": float(policy_action_age_s),
                        "policy_inference_dt_s": float(policy_inference_dt_s),
                        "control_loop_hz": float(1.0 / phase_b_loop_dt_s) if phase_b_loop_dt_s > 0.0 else float("nan"),
                        "policy_trajectory_t_write": float(policy_trajectory_t_write),
                        "policy_trajectory_is_new": bool(policy_trajectory_is_new),
                    }
                )

            sleep_s = dt - (time.perf_counter() - loop_t0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    except Exception as exc:
        with state_cache["lock"]:
            state_cache["phase_b_error"] = repr(exc)
        print(f"[PHASE B] BOTA-SE3 residual loop failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if follower_stop is not None:
            follower_stop.set()
        if follower_thread is not None:
            follower_thread.join(timeout=1.0)
        try:
            follower = getattr(system, "_direct_follower_robot", None)
            robot = getattr(follower, "robot", None)
            if robot is not None and hasattr(robot, "stopJ"):
                robot.stopJ(2.0)
                print("[FOLLOWER] stopJ sent during Phase-B shutdown")
        except Exception as exc:
            print(f"[FOLLOWER] stopJ failed during Phase-B shutdown: {exc}")
        if bota_reader is not None:
            bota_reader.close()
        try:
            if system.driver is not None:
                system.driver.set_joints(latest_raw.tolist())
        except Exception:
            pass


