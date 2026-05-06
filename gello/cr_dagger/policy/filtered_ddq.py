"""Filtered ddq reference computation."""
from __future__ import annotations

import numpy as np


class FilteredDDQComputer:
    """Compute ddq from dq history with EMA smoothing."""

    def __init__(self, n_joints: int, alpha: float = 0.2):
        self.n_joints = int(n_joints)
        self.alpha = float(alpha)
        self._prev_dq = np.zeros(self.n_joints, dtype=float)
        self._prev_ddq = np.zeros(self.n_joints, dtype=float)
        self._prev_t: float | None = None

    def update(self, dq_ref: np.ndarray, t_now: float) -> np.ndarray:
        dq_ref = np.asarray(dq_ref, dtype=float)
        if dq_ref.shape[0] < self.n_joints:
            raise ValueError(
                f"dq_ref must have at least {self.n_joints} elements, got {dq_ref.shape}"
            )

        if self._prev_t is None:
            self._prev_t = float(t_now)
            self._prev_dq = dq_ref[: self.n_joints].copy()
            self._prev_ddq = np.zeros(self.n_joints, dtype=float)
            return self._prev_ddq.copy()

        dt = float(t_now - self._prev_t)
        if dt < 1e-6:
            return self._prev_ddq.copy()

        ddq_raw = (dq_ref[: self.n_joints] - self._prev_dq) / dt
        ddq_filt = self.alpha * ddq_raw + (1.0 - self.alpha) * self._prev_ddq

        self._prev_dq = dq_ref[: self.n_joints].copy()
        self._prev_ddq = ddq_filt.copy()
        self._prev_t = float(t_now)
        return ddq_filt.copy()
