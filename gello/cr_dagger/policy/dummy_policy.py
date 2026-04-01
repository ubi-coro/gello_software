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
