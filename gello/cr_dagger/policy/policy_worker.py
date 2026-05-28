"""Policy inference worker process for CR-DAgger."""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class LeRobotACTPolicy:
    """LeRobot ACT rollout adapter that publishes follower-frame joint targets."""

    def __init__(
        self,
        policy_config: dict,
        horizon: int,
        n_joints: int,
    ) -> None:
        self.horizon = int(horizon)
        self.n_joints = int(n_joints)
        self.task = str(policy_config.get("task", ""))
        self.robot_type = str(policy_config.get("robot_type", ""))
        self.camera_names = list(policy_config.get("camera_names") or [])
        self.dataset_features = dict(policy_config.get("dataset_features") or {})
        self._last_missing_print_t = 0.0
        self._last_action: np.ndarray | None = None

        policy_path = str(policy_config.get("policy_path") or "")
        if not policy_path:
            raise ValueError("lerobot_act requires policy_config['policy_path']")

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.control_utils import predict_action

        import torch

        self._torch = torch
        self._predict_action = predict_action

        self.policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        self.policy_cfg.pretrained_path = policy_path
        if policy_config.get("device"):
            self.policy_cfg.device = str(policy_config["device"])

        dataset_repo = str(policy_config.get("dataset_repo") or "")
        dataset_root = policy_config.get("dataset_root")
        ds_meta = None
        if dataset_repo:
            try:
                ds_meta = LeRobotDatasetMetadata(
                    dataset_repo,
                    root=Path(dataset_root) if dataset_root else None,
                )
                print(f"[PolicyWorker] Loaded LeRobot metadata from {dataset_repo}")
            except Exception as exc:
                print(
                    f"[PolicyWorker] Could not load dataset metadata '{dataset_repo}': {exc}. "
                    "Falling back to runtime dataset features."
                )
        if ds_meta is None:
            if not self.dataset_features:
                raise RuntimeError(
                    "No LeRobot dataset metadata available. Pass --policy-dataset-repo "
                    "or provide runtime dataset features."
                )
            ds_meta = SimpleNamespace(features=self.dataset_features, stats={})

        self.policy = make_policy(self.policy_cfg, ds_meta=ds_meta)
        self.policy = self.policy.eval()
        dataset_stats = getattr(ds_meta, "stats", None)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=self.policy_cfg.pretrained_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": self.policy_cfg.device},
            },
        )

        for obj in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()

        self.input_keys = list(getattr(self.policy_cfg, "input_features", {}) or [])
        if not self.input_keys:
            self.input_keys = [
                key for key in getattr(ds_meta, "features", {})
                if str(key).startswith("observation.")
            ]
        print(
            f"[PolicyWorker] LeRobot ACT loaded from {policy_path} "
            f"device={self.policy_cfg.device} inputs={self.input_keys}"
        )

    def _build_observation(self, obs: dict) -> dict[str, np.ndarray] | None:
        images = obs.get("images") or {}
        state = np.concatenate(
            [
                np.asarray(obs["q"][: self.n_joints], dtype=np.float32),
                np.asarray(obs["dq"][: self.n_joints], dtype=np.float32),
                np.asarray([obs.get("grip", 0.0)], dtype=np.float32),
            ]
        )
        frame: dict[str, np.ndarray] = {}
        missing: list[str] = []

        for key in self.input_keys:
            if key == "observation.state":
                frame[key] = state
            elif key == "observation.effort":
                frame[key] = np.asarray(obs.get("tau_ext", np.zeros(self.n_joints))[: self.n_joints], dtype=np.float32)
            elif key == "observation.wrench":
                frame[key] = np.asarray(obs.get("wrench", np.zeros(6))[:6], dtype=np.float32)
            elif key == "observation.delta_q":
                frame[key] = np.zeros(self.n_joints, dtype=np.float32)
            elif key == "observation.is_correction":
                frame[key] = np.zeros(1, dtype=np.float32)
            elif key == "observation.detector_votes":
                frame[key] = np.zeros(4, dtype=np.float32)
            elif key.startswith("observation.images."):
                cam_name = key.rsplit(".", 1)[-1]
                img = images.get(cam_name)
                if img is None and len(images) == 1:
                    img = next(iter(images.values()))
                if img is None:
                    missing.append(key)
                    continue
                frame[key] = np.asarray(img, dtype=np.uint8)
            else:
                # Unknown optional observation feature: supply zeros if the shape is known.
                feature = self.dataset_features.get(key, {})
                shape = tuple(feature.get("shape", (1,))) if isinstance(feature, dict) else (1,)
                frame[key] = np.zeros(shape, dtype=np.float32)

        if missing:
            now = time.monotonic()
            if now - self._last_missing_print_t > 2.0:
                self._last_missing_print_t = now
                print(f"[PolicyWorker] Waiting for observation keys: {missing}")
            return None
        return frame

    def _action_to_trajectory(self, action: Any) -> np.ndarray:
        if hasattr(action, "detach"):
            action_arr = action.detach().cpu().numpy()
        else:
            action_arr = np.asarray(action)
        action_arr = np.asarray(action_arr, dtype=float).squeeze()
        if action_arr.ndim == 0:
            raise ValueError(f"Policy returned scalar action {action_arr}")
        if action_arr.ndim == 1:
            q = action_arr[: self.n_joints]
            return np.tile(q, (self.horizon, 1))
        traj = action_arr[:, : self.n_joints]
        if traj.shape[0] >= self.horizon:
            return traj[: self.horizon].copy()
        pad = np.tile(traj[-1], (self.horizon - traj.shape[0], 1))
        return np.vstack([traj, pad])

    def predict(self, obs: dict | None) -> np.ndarray | None:
        if obs is None:
            return self._last_action
        frame = self._build_observation(obs)
        if frame is None:
            return self._last_action
        action = self._predict_action(
            observation=frame,
            policy=self.policy,
            device=self._torch.device(self.policy_cfg.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=bool(getattr(self.policy_cfg, "use_amp", False)),
            task=self.task,
            robot_type=self.robot_type,
        )
        self._last_action = self._action_to_trajectory(action)
        return self._last_action


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
        name=obs_shm_name,
        n_joints=n_joints,
        camera_names=policy_config.get("camera_names"),
        create=False,
    )

    if policy_type in ("lerobot_act", "act"):
        policy = LeRobotACTPolicy(
            policy_config=policy_config,
            horizon=horizon,
            n_joints=n_joints,
        )
    elif policy_type == "dummy_sine":
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

    output_frame = str(policy_config.get("output_frame", "leader")).strip().lower()
    if output_frame not in ("leader", "follower"):
        raise ValueError(f"output_frame must be 'leader' or 'follower', got {output_frame!r}")
    inverse_bridge = None
    if output_frame == "follower":
        inverse_bridge = _maybe_build_inverse_bridge(policy_config)
        if inverse_bridge is None:
            raise ValueError("follower-frame policy output requires map_index/map_signs/map_offsets")

    print(
        f"[PolicyWorker] Policy '{policy_type}' loaded "
        f"(output_frame={output_frame}). Running inference loop."
    )

    while not stop_event.is_set():
        t_loop_start = time.monotonic()

        obs = obs_snap.read()
        t_now = time.monotonic()
        
        t_predict_start = time.perf_counter()
        if isinstance(policy, LeRobotACTPolicy):
            actions_pred = policy.predict(obs)
        else:
            actions_pred = policy.predict(t_now)
        policy_inference_dt_s = time.perf_counter() - t_predict_start
        if actions_pred is None:
            time.sleep(min(float(action_dt), 0.01))
            continue

        actions_follower = np.asarray(actions_pred, dtype=float)
        if inverse_bridge is not None:
            actions_leader = np.zeros_like(actions_follower)
            for i in range(actions_follower.shape[0]):
                actions_leader[i] = inverse_bridge.follower_to_leader(actions_follower[i])
        else:
            actions_leader = actions_follower

        traj_buf.write(
            trajectory=actions_leader,
            t_write=t_now,
            policy_inference_dt_s=policy_inference_dt_s,
        )

        inference_time = time.monotonic() - t_loop_start
        sleep_time = max(0.0, float(action_dt) - inference_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    traj_buf.close()
    obs_snap.close()
    print("[PolicyWorker] Stopped.")
