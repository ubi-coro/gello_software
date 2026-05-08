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


class DummyTaskSpacePolicy:
    """Generate correlated joint references from task-space primitives via IK."""

    def __init__(
        self,
        pin_model: object,
        pin_data: object,
        q_home: np.ndarray,
        motion_type: str = "line",
        amplitude: float = 0.08,
        frequency: float = 0.3,
        horizon: int = 32,
        action_dt: float = 0.1,
        axis: str = "xy",
    ):
        import pinocchio as pin

        self.pin_model = pin_model
        self.pin_data = pin_data
        self.q_home = np.asarray(q_home, dtype=float)
        self.motion_type = str(motion_type)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.horizon = int(horizon)
        self.action_dt = float(action_dt)
        self.axis = str(axis)

        self._n = int(self.q_home.shape[0])
        if self.pin_model.nq < self._n:
            raise ValueError(f"Pinocchio model nq={self.pin_model.nq} is smaller than n_joints={self._n}")

        self._ee_frame_id = int(self.pin_model.nframes - 1)

        q_full = np.zeros(self.pin_model.nq, dtype=float)
        q_full[: self._n] = self.q_home
        pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        home_pose = self.pin_data.oMf[self._ee_frame_id].copy()
        self._home_pos = home_pose.translation.copy()

        self._q_prev = self.q_home.copy()

    def _task_space_offset(self, t: float) -> np.ndarray:
        omega = 2.0 * np.pi * self.frequency
        amp = self.amplitude

        if self.motion_type == "line":
            dx = amp * np.sin(omega * t)
            dy = 0.0
        elif self.motion_type == "circle":
            dx = amp * np.cos(omega * t) - amp
            dy = amp * np.sin(omega * t)
        elif self.motion_type == "figure8":
            dx = amp * np.sin(omega * t)
            dy = 0.5 * amp * np.sin(2.0 * omega * t)
        else:
            dx = 0.0
            dy = 0.0

        offset = np.zeros(3, dtype=float)
        if self.axis == "xy":
            offset[0], offset[1] = dx, dy
        elif self.axis == "xz":
            offset[0], offset[2] = dx, dy
        elif self.axis == "yz":
            offset[1], offset[2] = dx, dy
        else:
            offset[0], offset[1] = dx, dy
        return offset

    def _ik_solve(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        import pinocchio as pin

        q = np.asarray(q_init, dtype=float).copy()
        q_full = np.zeros(self.pin_model.nq, dtype=float)

        for _ in range(8):
            q_full[: self._n] = q
            pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            current_pos = self.pin_data.oMf[self._ee_frame_id].translation
            pos_error = target_pos - current_pos
            if np.linalg.norm(pos_error) < 1e-5:
                break

            jacobian = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q_full,
                self._ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            j_pos = jacobian[:3, : self._n]
            damping = 1e-4
            jj_t = j_pos @ j_pos.T + damping * np.eye(3)
            dq = j_pos.T @ np.linalg.solve(jj_t, pos_error)
            q = q + 0.8 * dq

        return q

    def predict(self, t_now: float) -> np.ndarray:
        traj = np.zeros((self.horizon, self._n), dtype=float)
        q_seed = self._q_prev.copy()

        for k in range(self.horizon):
            t_k = float(t_now + k * self.action_dt)
            target_pos = self._home_pos + self._task_space_offset(t_k)
            q_seed = self._ik_solve(target_pos, q_seed)
            traj[k] = q_seed
            if k == 0:
                self._q_prev = q_seed.copy()

        return traj
