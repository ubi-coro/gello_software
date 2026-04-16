"""Policy inference worker process for CR-DAgger."""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np

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
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    print(f"[PolicyWorker] Policy '{policy_type}' loaded. Running inference loop.")

    while not stop_event.is_set():
        t_loop_start = time.monotonic()

        obs = obs_snap.read()
        t_now = time.monotonic()
        
        actions = policy.predict(t_now)
        traj_buf.write(trajectory=actions, t_write=t_now)

        inference_time = time.monotonic() - t_loop_start
        sleep_time = max(0.0, float(action_dt) - inference_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    traj_buf.close()
    obs_snap.close()
    print("[PolicyWorker] Stopped.")
