"""High-frequency epsilon recorder for tracking tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EpsilonRecorder:
    """Record epsilon test signals at a fixed rate."""

    def __init__(
        self,
        n_joints: int = 6,
        max_duration_s: float = 60.0,
        rate_hz: float = 500.0,
    ):
        max_steps = int(max_duration_s * rate_hz) + 1000
        self.n_joints = int(n_joints)
        self.rate_hz = float(rate_hz)
        self.max_steps = int(max_steps)

        self.fields: dict[str, np.ndarray] = {
            "t_mono": np.zeros(self.max_steps, dtype=float),
            "q_ref": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "dq_ref": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "ddq_ref": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "q_leader": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "dq_leader": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "q_cmd_leader": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "q_c_leader": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "q_follower": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "dq_follower": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "q_cmd_follower": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "delta_corr": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "tau_cmd": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "tau_ext_shi": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "tau_model": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "tau_meas": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "tau_residual": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "wrench": np.zeros((self.max_steps, 6), dtype=float),
            "bota_wrench_raw": np.zeros((self.max_steps, 6), dtype=float),
            "bota_wrench_base": np.zeros((self.max_steps, 6), dtype=float),
            "bota_wrench_conditioned": np.zeros((self.max_steps, 6), dtype=float),
            "bota_tau_joint": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "bota_delta_q_cartesian": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "bota_delta_q_hfvc": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "bota_delta_q_se3": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "bota_delta_q_joint": np.zeros((self.max_steps, self.n_joints), dtype=float),
            "bota_task_offset": np.zeros((self.max_steps, 6), dtype=float),
            "bota_task_offset_hfvc": np.zeros((self.max_steps, 6), dtype=float),
            "bota_task_offset_se3": np.zeros((self.max_steps, 6), dtype=float),
            "bota_pose_error_se3": np.zeros((self.max_steps, 6), dtype=float),
            "bota_status": np.zeros((self.max_steps, 4), dtype=float),
            "contact_probability": np.zeros(self.max_steps, dtype=float),
            "contact_state": np.zeros(self.max_steps, dtype=float),
            "epsilon": np.zeros((self.max_steps, self.n_joints), dtype=float),
        }
        self._idx = 0
        self._dropped = False

    def record(self, **kwargs: Any) -> None:
        if self._idx >= self.max_steps:
            self._dropped = True
            return

        i = self._idx
        for key, value in kwargs.items():
            if key not in self.fields:
                continue
            self.fields[key][i] = np.asarray(value, dtype=float)
        self._idx += 1

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        trimmed = {key: val[: self._idx] for key, val in self.fields.items()}
        np.savez_compressed(path, **trimmed, metadata_json=metadata_json)
        return path

    @property
    def dropped_samples(self) -> bool:
        return self._dropped
