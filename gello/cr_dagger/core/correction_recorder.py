"""Correction data recorder for CR-DAgger."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CorrectionFrame:
    """Single timestep of correction data."""

    timestamp: float                # Monotonic clock [s].
    q_ref: np.ndarray               # Policy output/reference in follower joint frame.
    dq_ref: np.ndarray              # Policy velocity reference from finite differences.
    q_actual: np.ndarray            # GELLO measured position from encoders.
    dq_actual: np.ndarray           # GELLO measured velocity.
    q_compliant: np.ndarray         # Compliant command sent to the follower.
    dq_compliant: np.ndarray        # Compliant velocity state if available.
    delta_q: np.ndarray             # q_compliant - q_ref with latency compensation.
    delta_q_raw: np.ndarray         # q_compliant - q_ref without latency compensation.
    tau_ext_gello: np.ndarray       # External torque estimate in leader joint frame.
    q_follower: np.ndarray          # UR5e joint positions.
    wrench_ur5e: np.ndarray         # Conditioned base-frame wrench [Fx,Fy,Fz,Tx,Ty,Tz].
    is_correction: bool             # Fused detector output.
    detector_diagnostics: dict      # Per-detector votes and scalar diagnostics.
    q_ref_leader: np.ndarray | None = None         # Policy reference in leader joint frame.
    q_cmd_leader: np.ndarray | None = None         # Leader command q_ref_leader + delta_leader.
    q_cmd_ur5e: np.ndarray | None = None           # Final UR5e command in follower joint frame.
    epsilon_leader: np.ndarray | None = None       # Leader tracking error q_actual - q_cmd_leader.
    epsilon_ur5e: np.ndarray | None = None         # Follower tracking error q_follower - q_cmd_ur5e.
    wrench_sensor_raw: np.ndarray | None = None    # Raw BOTA sensor-frame wrench before software bias.
    wrench_base_raw: np.ndarray | None = None      # Base-frame wrench before LPF/deadband/saturation.
    wrench_base: np.ndarray | None = None          # Conditioned base-frame wrench used by admittance.
    task_delta: np.ndarray | None = None           # SE(3) admittance command before joint projection.
    task_pose_error: np.ndarray | None = None      # SE(3) pose error used by admittance.
    delta_leader: np.ndarray | None = None         # Limited human residual in leader joint frame.
    delta_leader_raw: np.ndarray | None = None     # Raw projected residual before residual limiter.
    images: dict[str, np.ndarray] | None = None


class CorrectionRecorder:
    def __init__(
        self,
        n_joints: int,
        log_dir: str = "cr_dagger_data",
        latency_compensation_s: float = 0.008,
        q_ref_history_length: int = 100,
        metadata: dict[str, Any] | None = None,
    ):
        self.n_joints = int(n_joints)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.latency_compensation_s = float(latency_compensation_s)
        self.metadata = dict(metadata or {})

        self.q_ref_history = deque(maxlen=q_ref_history_length)
        self.frames: list[CorrectionFrame] = []
        self.episode_id = None

    def start_episode(self, episode_id: str | None = None) -> None:
        self.episode_id = episode_id or f"episode_{int(time.monotonic())}"
        self.frames = []
        self.q_ref_history.clear()

    def record(
        self,
        timestamp: float,
        q_ref: np.ndarray,
        q_actual: np.ndarray,
        dq_actual: np.ndarray,
        q_compliant: np.ndarray,
        dq_compliant: np.ndarray,
        tau_ext_gello: np.ndarray,
        q_follower: np.ndarray,
        wrench_ur5e: np.ndarray,
        is_correction: bool,
        detector_diagnostics: dict,
        images: dict[str, np.ndarray] | None = None,
        q_ref_leader: np.ndarray | None = None,
        q_cmd_leader: np.ndarray | None = None,
        q_cmd_ur5e: np.ndarray | None = None,
        epsilon_leader: np.ndarray | None = None,
        epsilon_ur5e: np.ndarray | None = None,
        wrench_sensor_raw: np.ndarray | None = None,
        wrench_base_raw: np.ndarray | None = None,
        wrench_base: np.ndarray | None = None,
        task_delta: np.ndarray | None = None,
        task_pose_error: np.ndarray | None = None,
        delta_leader: np.ndarray | None = None,
        delta_leader_raw: np.ndarray | None = None,
    ) -> None:
        q_ref = np.asarray(q_ref, dtype=float).copy()
        q_compliant = np.asarray(q_compliant, dtype=float).copy()
        self.q_ref_history.append((float(timestamp), q_ref.copy()))

        if len(self.q_ref_history) > 1:
            dt = float(timestamp) - self.q_ref_history[-2][0]
            if dt > 0.0:
                dq_ref = (q_ref - self.q_ref_history[-2][1]) / dt
            else:
                dq_ref = np.zeros_like(q_ref)
        else:
            dq_ref = np.zeros_like(q_ref)

        delta_q_raw = q_compliant - q_ref
        delta_q = self._compute_delta_q_compensated(q_compliant, float(timestamp))

        frame = CorrectionFrame(
            timestamp=float(timestamp),
            q_ref=q_ref.copy(),
            dq_ref=dq_ref.copy(),
            q_actual=np.asarray(q_actual, dtype=float).copy(),
            dq_actual=np.asarray(dq_actual, dtype=float).copy(),
            q_compliant=q_compliant.copy(),
            dq_compliant=np.asarray(dq_compliant, dtype=float).copy(),
            delta_q=delta_q.copy(),
            delta_q_raw=delta_q_raw.copy(),
            tau_ext_gello=np.asarray(tau_ext_gello, dtype=float).copy(),
            q_follower=np.asarray(q_follower, dtype=float).copy(),
            wrench_ur5e=np.asarray(wrench_ur5e, dtype=float).copy(),
            is_correction=bool(is_correction),
            detector_diagnostics=dict(detector_diagnostics or {}),
            q_ref_leader=self._copy_optional(q_ref_leader),
            q_cmd_leader=self._copy_optional(q_cmd_leader),
            q_cmd_ur5e=self._copy_optional(q_cmd_ur5e),
            epsilon_leader=self._copy_optional(epsilon_leader),
            epsilon_ur5e=self._copy_optional(epsilon_ur5e),
            wrench_sensor_raw=self._copy_optional(wrench_sensor_raw),
            wrench_base_raw=self._copy_optional(wrench_base_raw),
            wrench_base=self._copy_optional(wrench_base),
            task_delta=self._copy_optional(task_delta),
            task_pose_error=self._copy_optional(task_pose_error),
            delta_leader=self._copy_optional(delta_leader),
            delta_leader_raw=self._copy_optional(delta_leader_raw),
            images=images,
        )
        self.frames.append(frame)

    def end_episode(self) -> Path | None:
        if not self.frames:
            return None

        T = len(self.frames)
        timestamps = np.array([f.timestamp for f in self.frames])
        q_ref = np.array([f.q_ref for f in self.frames])
        dq_ref = np.array([f.dq_ref for f in self.frames])
        q_actual = np.array([f.q_actual for f in self.frames])
        dq_actual = np.array([f.dq_actual for f in self.frames])
        q_compliant = np.array([f.q_compliant for f in self.frames])
        dq_compliant = np.array([f.dq_compliant for f in self.frames])
        delta_q = np.array([f.delta_q for f in self.frames])
        delta_q_raw = np.array([f.delta_q_raw for f in self.frames])
        tau_ext = np.array([f.tau_ext_gello for f in self.frames])
        q_follower = np.array([f.q_follower for f in self.frames])
        wrench = np.array([f.wrench_ur5e for f in self.frames])
        is_correction = np.array([f.is_correction for f in self.frames])

        q_ref_leader = self._stack_optional("q_ref_leader", self.n_joints)
        q_cmd_leader = self._stack_optional("q_cmd_leader", self.n_joints)
        q_cmd_ur5e = self._stack_optional("q_cmd_ur5e", self.n_joints)
        epsilon_leader = self._stack_optional("epsilon_leader", self.n_joints)
        epsilon_ur5e = self._stack_optional("epsilon_ur5e", self.n_joints)
        wrench_sensor_raw = self._stack_optional("wrench_sensor_raw", 6)
        wrench_base_raw = self._stack_optional("wrench_base_raw", 6)
        wrench_base = self._stack_optional("wrench_base", 6)
        task_delta = self._stack_optional("task_delta", 6)
        task_pose_error = self._stack_optional("task_pose_error", 6)
        delta_leader = self._stack_optional("delta_leader", self.n_joints)
        delta_leader_raw = self._stack_optional("delta_leader_raw", self.n_joints)

        detector_votes = np.zeros((T, 4), dtype=bool)
        contact_probability = np.zeros(T, dtype=np.float32)
        contact_gate = np.zeros(T, dtype=np.float32)
        intervention_source = np.zeros(T, dtype=np.float32)
        bota_temperature_c = np.zeros(T, dtype=np.float32)
        bota_status_flags = np.zeros((T, 8), dtype=np.float32)
        for i, f in enumerate(self.frames):
            diag = f.detector_diagnostics or {}
            v = diag.get("votes", {})
            detector_votes[i] = [
                v.get("torque", False),
                v.get("delta", v.get("delta_q", False)),
                v.get("energy", False),
                v.get("wrench", False),
            ]
            contact_probability[i] = float(diag.get("contact_probability", 0.0))
            contact_gate[i] = float(diag.get("contact_gate", 0.0))
            intervention_source[i] = float(diag.get("intervention_source", 0.0))
            bota_temperature_c[i] = float(diag.get("bota_temperature_c", 0.0))
            flags = np.asarray(diag.get("bota_status_flags", []), dtype=np.float32).ravel()
            if flags.size:
                bota_status_flags[i, : min(flags.size, bota_status_flags.shape[1])] = flags[: bota_status_flags.shape[1]]

        metadata_json = json.dumps(self.metadata, sort_keys=True, default=self._metadata_json_default)
        filename = self.log_dir / f"{self.episode_id}.npz"
        if filename.exists():
            stamp = int(time.monotonic())
            filename = self.log_dir / f"{self.episode_id}_{stamp}.npz"
        np.savez_compressed(
            filename,
            timestamps=timestamps,
            q_ref=q_ref,
            q_ref_follower=q_ref,
            q_ref_leader=q_ref_leader,
            dq_ref=dq_ref,
            q_actual=q_actual,
            q_leader=q_actual,
            dq_actual=dq_actual,
            dq_leader=dq_actual,
            q_compliant=q_compliant,
            q_cmd_ur5e=q_cmd_ur5e,
            q_cmd_follower=q_cmd_ur5e,
            q_cmd_leader=q_cmd_leader,
            dq_compliant=dq_compliant,
            delta_q=delta_q,
            delta_q_raw=delta_q_raw,
            delta_leader=delta_leader,
            delta_leader_raw=delta_leader_raw,
            tau_ext=tau_ext,
            tau_ext_shi=tau_ext,
            q_follower=q_follower,
            epsilon=epsilon_ur5e,
            epsilon_ur5e=epsilon_ur5e,
            epsilon_follower=epsilon_ur5e,
            epsilon_leader=epsilon_leader,
            wrench=wrench,
            wrench_sensor_raw=wrench_sensor_raw,
            wrench_base_raw=wrench_base_raw,
            wrench_base=wrench_base,
            bota_wrench_raw=wrench_sensor_raw,
            bota_wrench_base=wrench_base_raw,
            bota_wrench_conditioned=wrench_base,
            task_delta=task_delta,
            task_pose_error=task_pose_error,
            bota_task_offset_se3=task_delta,
            bota_pose_error_se3=task_pose_error,
            bota_delta_q_se3=delta_leader_raw,
            bota_delta_q_se3_limited=delta_leader,
            is_correction=is_correction,
            detector_votes=detector_votes,
            contact_probability=contact_probability,
            contact_gate=contact_gate,
            contact_state=contact_gate,
            intervention_active=is_correction.astype(np.float32),
            intervention_source=intervention_source,
            bota_temperature_c=bota_temperature_c,
            bota_status_flags=bota_status_flags,
            bota_status=bota_status_flags[:, :4],
            metadata_json=metadata_json,
        )
        print(f"[CorrectionRecorder] Episode saved: {filename}")

        # LeRobot owns image/video storage; NPZ keeps synchronized scalar diagnostics.
        return filename

    @staticmethod
    def _copy_optional(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=float).copy()

    def _stack_optional(self, field_name: str, width: int) -> np.ndarray:
        rows = []
        for frame in self.frames:
            value = getattr(frame, field_name)
            if value is None:
                rows.append(np.full(int(width), np.nan, dtype=float))
            else:
                rows.append(np.asarray(value, dtype=float).reshape(-1)[: int(width)])
                if rows[-1].size < int(width):
                    rows[-1] = np.pad(rows[-1], (0, int(width) - rows[-1].size), constant_values=np.nan)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _metadata_json_default(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _compute_delta_q_compensated(
        self, q_compliant: np.ndarray, timestamp: float
    ) -> np.ndarray:
        if self.latency_compensation_s <= 0.0 or len(self.q_ref_history) < 2:
            return q_compliant - self.q_ref_history[-1][1]

        t_lookback = timestamp - self.latency_compensation_s

        for i in range(len(self.q_ref_history) - 1, 0, -1):
            t1, q1 = self.q_ref_history[i]
            t0, q0 = self.q_ref_history[i - 1]
            if t0 <= t_lookback <= t1:
                if t1 == t0:
                    q_interp = q1
                else:
                    ratio = (t_lookback - t0) / (t1 - t0)
                    q_interp = q0 + ratio * (q1 - q0)
                return q_compliant - q_interp

        if t_lookback < self.q_ref_history[0][0]:
            return q_compliant - self.q_ref_history[0][1]
        return q_compliant - self.q_ref_history[-1][1]
