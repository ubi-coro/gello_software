"""Policy inference worker process for CR-DAgger."""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np


class InverseMapBridge:
    """Convert follower-frame policy output back to leader-frame commands."""

    def __init__(self, map_signs: np.ndarray, map_offsets: np.ndarray, map_index: np.ndarray):
        self.signs = np.asarray(map_signs, dtype=float)
        self.offsets = np.asarray(map_offsets, dtype=float)
        self.index = np.asarray(map_index, dtype=int)

        if not (len(self.signs) == len(self.offsets) == len(self.index)):
            raise ValueError("map_signs, map_offsets, and map_index must have the same length")
        if np.any(np.abs(self.signs) <= 1e-6):
            raise ValueError("signs must be non-zero for inverse mapping")

    def follower_to_leader(self, action_follower: np.ndarray) -> np.ndarray:
        """Convert UR5e-frame action to GELLO-frame reference."""
        action_follower = np.asarray(action_follower, dtype=float)
        q_leader = np.zeros_like(action_follower)
        if action_follower.shape[0] < len(self.index):
            raise ValueError(
                f"action_follower must have at least {len(self.index)} elements, got {action_follower.shape[0]}"
            )
        q_leader[self.index] = (action_follower[: len(self.index)] - self.offsets) / self.signs
        return q_leader

    def leader_to_follower(self, q_leader: np.ndarray) -> np.ndarray:
        """Convert GELLO-frame reference to UR5e-frame action."""
        q_leader = np.asarray(q_leader, dtype=float)
        action_follower = np.zeros_like(q_leader)
        action_follower[: len(self.index)] = self.signs * q_leader[self.index] + self.offsets
        return action_follower


def _maybe_build_inverse_bridge(policy_config: dict) -> InverseMapBridge | None:
    map_index = policy_config.get("map_index")
    map_signs = policy_config.get("map_signs")
    map_offsets = policy_config.get("map_offsets")
    if map_index is None or map_signs is None or map_offsets is None:
        return None
    return InverseMapBridge(
        map_signs=np.asarray(map_signs, dtype=float),
        map_offsets=np.asarray(map_offsets, dtype=float),
        map_index=np.asarray(map_index, dtype=int),
    )

def policy_worker(
    traj_shm_name: str,
    obs_shm_name: str,
    horizon: int,
    n_joints: int,
    action_dt: float,
    policy_type: str,
    policy_config: dict,
    stop_event: Any,
) -> None:
    
    from gello.cr_dagger.ipc.shared_trajectory_buffer import SharedTrajectoryBuffer
    from gello.cr_dagger.ipc.shared_observation_snapshot import SharedObservationSnapshot

    traj_buf = SharedTrajectoryBuffer(
        name=traj_shm_name, horizon=horizon,
        n_joints=n_joints, create=False,
    )
    obs_snap = SharedObservationSnapshot(
        name=obs_shm_name, n_joints=n_joints, create=False,
    )

    if policy_type == "dummy_sine":
        from gello.cr_dagger.policy.dummy_policy import DummySinePolicy
        center = np.array(policy_config.get("center", [0.0] * n_joints))
        policy = DummySinePolicy(
            n_joints=n_joints, horizon=horizon, action_dt=action_dt,
            center=center,
            amplitude=np.array(policy_config.get("amplitude", [0.1] * n_joints)),
            frequency=np.array(policy_config.get("frequency", [0.2] * n_joints)),
        )
    elif policy_type == "dummy_hold":
        from gello.cr_dagger.policy.dummy_policy import StaticHoldPolicy
        q_hold = np.array(policy_config.get("q_hold", [0.0] * n_joints))
        policy = StaticHoldPolicy(q_hold=q_hold, horizon=horizon, n_joints=n_joints)
    elif policy_type == "dummy_chirp":
        from gello.cr_dagger.policy.dummy_policy import DummyChirpPolicy
        center = np.array(policy_config.get("center", [0.0] * n_joints))
        policy = DummyChirpPolicy(
            center=center,
            amplitude=float(policy_config.get("amplitude", 0.1)),
            f_start=float(policy_config.get("f_start", 0.1)),
            f_end=float(policy_config.get("f_end", 2.0)),
            duration=float(policy_config.get("duration", 30.0)),
            horizon=horizon,
            action_dt=action_dt,
            joint_index=int(policy_config.get("joint_index", 1)),
        )
    elif policy_type == "dummy_ramp":
        from gello.cr_dagger.policy.dummy_policy import DummyRampPolicy
        q_start = np.array(policy_config.get("q_start", [0.0] * n_joints))
        q_end = np.array(policy_config.get("q_end", q_start.tolist()))
        policy = DummyRampPolicy(
            q_start=q_start,
            q_end=q_end,
            duration=float(policy_config.get("duration", 5.0)),
            horizon=horizon,
            action_dt=action_dt,
        )
    elif policy_type == "dummy_multi_sine":
        from gello.cr_dagger.policy.dummy_policy import DummyMultiJointSinePolicy
        center = np.array(policy_config.get("center", [0.0] * n_joints))
        amplitudes = np.array(policy_config.get("amplitudes", [0.05] * n_joints))
        frequencies = np.array(policy_config.get("frequencies", [0.2] * n_joints))
        policy = DummyMultiJointSinePolicy(
            center=center,
            amplitudes=amplitudes,
            frequencies=frequencies,
            horizon=horizon,
            action_dt=action_dt,
        )
    elif policy_type == "dummy_taskspace":
        import pinocchio as pin

        from gello.cr_dagger.policy.dummy_policy import DummyTaskSpacePolicy

        urdf_path = str(policy_config.get("urdf_path", ""))
        if not urdf_path:
            raise ValueError("dummy_taskspace requires 'urdf_path' in policy_config")

        pin_model = pin.buildModelFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        center = np.array(policy_config.get("center", [0.0] * n_joints))
        policy = DummyTaskSpacePolicy(
            pin_model=pin_model,
            pin_data=pin_data,
            q_home=center,
            motion_type=str(policy_config.get("motion_type", "circle")),
            amplitude=float(policy_config.get("amplitude", 0.08)),
            frequency=float(policy_config.get("frequency", 0.3)),
            horizon=horizon,
            action_dt=action_dt,
            axis=str(policy_config.get("axis", "xy")),
        )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    inverse_bridge = _maybe_build_inverse_bridge(policy_config)

    print(f"[PolicyWorker] Policy '{policy_type}' loaded. Running inference loop.")

    while not stop_event.is_set():
        t_loop_start = time.monotonic()

        obs = obs_snap.read()
        t_now = time.monotonic()
        
        actions_follower = np.asarray(policy.predict(t_now), dtype=float)
        if inverse_bridge is not None:
            actions_leader = np.zeros_like(actions_follower)
            for i in range(actions_follower.shape[0]):
                actions_leader[i] = inverse_bridge.follower_to_leader(actions_follower[i])
        else:
            actions_leader = actions_follower

        traj_buf.write(trajectory=actions_leader, t_write=t_now)

        inference_time = time.monotonic() - t_loop_start
        sleep_time = max(0.0, float(action_dt) - inference_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    traj_buf.close()
    obs_snap.close()
    print("[PolicyWorker] Stopped.")
