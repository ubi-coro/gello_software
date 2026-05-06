"""Dummy policies for testing the CR-DAgger pipeline."""
from __future__ import annotations

import numpy as np

class DummySinePolicy:
    """Generate sinusoidal joint trajectories."""

    def __init__(
        self,
        n_joints: int = 6,
        horizon: int = 32,
        action_dt: float = 0.1,
        amplitude: np.ndarray | None = None,
        frequency: np.ndarray | None = None,
        center: np.ndarray | None = None,
    ):
        self.n_joints = n_joints
        self.horizon = horizon
        self.action_dt = action_dt
        
        self.amplitude = np.array(amplitude) if amplitude is not None else np.ones(n_joints) * 0.1
        self.frequency = np.array(frequency) if frequency is not None else np.ones(n_joints) * 0.2
        self.center = np.array(center) if center is not None else np.zeros(n_joints)

    def predict(self, t_now: float) -> np.ndarray:
        traj = np.zeros((self.horizon, self.n_joints))
        for k in range(self.horizon):
            t_k = t_now + k * self.action_dt
            traj[k] = self.center + self.amplitude * np.sin(2 * np.pi * self.frequency * t_k)
        return traj

class StaticHoldPolicy:
    """Hold a fixed position."""

    def __init__(self, q_hold: np.ndarray, horizon: int = 32, n_joints: int = 6):
        self.q_hold = np.array(q_hold)
        self.horizon = horizon
        self.n_joints = n_joints

    def predict(self, t_now: float) -> np.ndarray:
        return np.tile(self.q_hold, (self.horizon, 1))


class DummyRampPolicy:
    """Linear ramp for constant-velocity tracking tests."""

    def __init__(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        duration: float = 5.0,
        horizon: int = 32,
        action_dt: float = 0.1,
    ):
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_end = np.asarray(q_end, dtype=float)
        self.duration = float(max(duration, 1e-6))
        self.horizon = horizon
        self.action_dt = action_dt
        self._t0: float | None = None

    def predict(self, t_now: float) -> np.ndarray:
        if self._t0 is None:
            self._t0 = float(t_now)

        traj = np.zeros((self.horizon, self.q_start.shape[0]))
        for k in range(self.horizon):
            t_k = float(t_now + k * self.action_dt)
            t_rel = max(0.0, t_k - float(self._t0))
            alpha = min(1.0, t_rel / self.duration)
            traj[k] = self.q_start + alpha * (self.q_end - self.q_start)
        return traj


class DummyChirpPolicy:
    """Frequency sweep on a single joint."""

    def __init__(
        self,
        center: np.ndarray,
        amplitude: float = 0.1,
        f_start: float = 0.1,
        f_end: float = 2.0,
        duration: float = 30.0,
        horizon: int = 32,
        action_dt: float = 0.1,
        joint_index: int = 1,
    ):
        self.center = np.asarray(center, dtype=float)
        self.amplitude = float(amplitude)
        self.f_start = float(f_start)
        self.f_end = float(f_end)
        self.duration = float(max(duration, 1e-6))
        self.horizon = horizon
        self.action_dt = action_dt
        self.joint_index = int(joint_index)
        self._t0: float | None = None

    def _chirp_phase(self, t_rel: float) -> float:
        k = (self.f_end - self.f_start) / self.duration
        return 2.0 * np.pi * (self.f_start * t_rel + 0.5 * k * t_rel * t_rel)

    def predict(self, t_now: float) -> np.ndarray:
        if self._t0 is None:
            self._t0 = float(t_now)

        traj = np.zeros((self.horizon, self.center.shape[0]))
        for k in range(self.horizon):
            t_k = float(t_now + k * self.action_dt)
            t_rel = max(0.0, min(self.duration, t_k - float(self._t0)))
            phase = self._chirp_phase(t_rel)
            q = self.center.copy()
            if 0 <= self.joint_index < q.shape[0]:
                q[self.joint_index] = self.center[self.joint_index] + self.amplitude * np.sin(phase)
            traj[k] = q
        return traj


class DummyMultiJointSinePolicy:
    """Multiple joints with distinct sine frequencies."""

    def __init__(
        self,
        center: np.ndarray,
        amplitudes: np.ndarray,
        frequencies: np.ndarray,
        horizon: int = 32,
        action_dt: float = 0.1,
    ):
        self.center = np.asarray(center, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.horizon = horizon
        self.action_dt = action_dt

    def predict(self, t_now: float) -> np.ndarray:
        traj = np.zeros((self.horizon, self.center.shape[0]))
        for k in range(self.horizon):
            t_k = float(t_now + k * self.action_dt)
            traj[k] = self.center + self.amplitudes * np.sin(2.0 * np.pi * self.frequencies * t_k)
        return traj
