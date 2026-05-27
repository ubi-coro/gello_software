"""Trajectory interpolator with staleness detection."""
from __future__ import annotations

import time

import numpy as np

from gello.cr_dagger.ipc.shared_trajectory_buffer import SharedTrajectoryBuffer

class TrajectoryInterpolator:
    def __init__(
        self,
        traj_buf: SharedTrajectoryBuffer,
        action_dt: float = 0.1,
        stale_threshold_s: float = 5.0,
        fallback_q: np.ndarray | None = None,
    ):
        self.traj_buf = traj_buf
        self.action_dt = action_dt
        self.stale_threshold_s = stale_threshold_s
        self.fallback_q = fallback_q
        
        self._last_q = None
        self._last_t = None
        self.last_t_write: float = 0.0
        self.last_action_age_s: float = float("nan")
        self.last_is_new: bool = False

    def get_reference(self, t_now: float) -> tuple[np.ndarray, np.ndarray, bool]:
        traj, t_write, is_new = self.traj_buf.read()
        self.last_t_write = float(t_write)
        self.last_action_age_s = float(t_now - t_write)
        self.last_is_new = bool(is_new)
        
        is_stale = (t_now - t_write > self.stale_threshold_s)
        
        if is_stale and self.fallback_q is not None:
            q_ref = self.fallback_q.copy()
        else:
            dt = t_now - t_write
            idx = max(0, min(self.traj_buf.horizon - 1, int(dt / self.action_dt)))
            idx_next = min(self.traj_buf.horizon - 1, idx + 1)

            t_idx = idx * self.action_dt
            t_next = idx_next * self.action_dt

            if t_next > t_idx:
                alpha = (dt - t_idx) / (t_next - t_idx)
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = 0.0
            q_ref = traj[idx] + alpha * (traj[idx_next] - traj[idx])
            
        if self._last_q is not None and self._last_t is not None and t_now > self._last_t:
            dq_ref = (q_ref - self._last_q) / (t_now - self._last_t)
        else:
            dq_ref = np.zeros_like(q_ref)
            
        self._last_q = q_ref.copy()
        self._last_t = t_now
        
        return q_ref, dq_ref, is_stale
