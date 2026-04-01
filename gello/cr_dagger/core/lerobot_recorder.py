"""LeRobot v3.0 dataset recorder for CR-DAgger correction data."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

class LeRobotCorrectionRecorder:
    DEFAULT_FEATURES = {
        "observation.state": {
            "dtype": "float32",
            "shape": (13,),
            "names": [
                "q_0", "q_1", "q_2", "q_3", "q_4", "q_5",
                "dq_0", "dq_1", "dq_2", "dq_3", "dq_4", "dq_5",
                "gripper",
            ],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["tau_ext_0", "tau_ext_1", "tau_ext_2", "tau_ext_3", "tau_ext_4", "tau_ext_5"],
        },
        "observation.wrench": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        },
        "observation.delta_q": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dq_0", "dq_1", "dq_2", "dq_3", "dq_4", "dq_5"],
        },
        "observation.is_correction": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_correction"],
        },
        "observation.detector_votes": {
            "dtype": "bool",
            "shape": (4,),
            "names": ["vote_torque", "vote_delta", "vote_energy", "vote_wrench"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["q_ref_0", "q_ref_1", "q_ref_2", "q_ref_3", "q_ref_4", "q_ref_5", "gripper_ref"],
        },
        "action.compliant": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["q_c_0", "q_c_1", "q_c_2", "q_c_3", "q_c_4", "q_c_5", "gripper_c"],
        },
    }

    def __init__(
        self,
        repo_id: str,
        fps: int = 330,
        task_description: str = "CR-DAgger correction episode",
        n_joints: int = 6,
        camera_names: list[str] | None = None,
        camera_shapes: dict[str, tuple[int, int, int]] | None = None,
        latency_compensation_s: float = 0.008,
        q_ref_history_length: int = 100,
        root: str | None = None,
    ):
        if not HAS_LEROBOT:
            raise ImportError(
                "lerobot >= 0.4.0 is required for LeRobot v3.0 recording. "
                "Install with: pip install lerobot"
            )

        self.repo_id = repo_id
        self.fps = fps
        self.task_description = task_description
        self.n_joints = n_joints
        self.latency_compensation_s = latency_compensation_s
        self.camera_names = camera_names or []

        self.camera_shapes = camera_shapes or {}
        for cam in self.camera_names:
            if cam not in self.camera_shapes:
                self.camera_shapes[cam] = (480, 640, 3)

        self.features = dict(self.DEFAULT_FEATURES)
        for cam in self.camera_names:
            h, w, c = self.camera_shapes[cam]
            self.features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": 30,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            }

        self._q_ref_history: deque[tuple[float, np.ndarray]] = deque(maxlen=q_ref_history_length)
        self._episode_start_t: float = 0.0
        self._frame_count: int = 0
        self._episode_count: int = 0

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self.features,
            root=Path(root) if root else None,
        )

    def start_episode(self, task_description: str | None = None) -> None:
        self._q_ref_history.clear()
        self._episode_start_t = time.monotonic()
        self._frame_count = 0
        if task_description is not None:
            self.task_description = task_description

    def add_frame(
        self,
        timestamp: float,
        q: np.ndarray,
        dq: np.ndarray,
        gripper: float,
        tau_ext: np.ndarray,
        wrench_ur5e: np.ndarray,
        q_ref: np.ndarray,
        q_compliant: np.ndarray,
        dq_compliant: np.ndarray,
        gripper_ref: float = 0.0,
        gripper_compliant: float = 0.0,
        is_correction: bool = False,
        detector_votes: np.ndarray | None = None,
        images: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._q_ref_history.append((timestamp, q_ref.copy()))

        delta_q = self._compute_delta_q_compensated(q_compliant, timestamp)

        frame = {
            "observation.state": torch.tensor(
                np.concatenate([q[:self.n_joints], dq[:self.n_joints], [gripper]]),
                dtype=torch.float32,
            ),
            "observation.effort": torch.tensor(
                tau_ext[:self.n_joints], dtype=torch.float32,
            ),
            "observation.wrench": torch.tensor(
                wrench_ur5e[:6], dtype=torch.float32,
            ),
            "observation.delta_q": torch.tensor(
                delta_q[:self.n_joints], dtype=torch.float32,
            ),
            "observation.is_correction": torch.tensor(
                [is_correction], dtype=torch.bool,
            ),
            "observation.detector_votes": torch.tensor(
                detector_votes if detector_votes is not None else [False, False, False, False],
                dtype=torch.bool,
            ),
            "action": torch.tensor(
                np.concatenate([q_ref[:self.n_joints], [gripper_ref]]),
                dtype=torch.float32,
            ),
            "action.compliant": torch.tensor(
                np.concatenate([q_compliant[:self.n_joints], [gripper_compliant]]),
                dtype=torch.float32,
            ),
        }

        if images is not None:
            for cam_name, img_array in images.items():
                key = f"observation.images.{cam_name}"
                if key in self.features:
                    frame[key] = torch.from_numpy(img_array.transpose(2, 0, 1).copy())

        self.dataset.add_frame(frame)
        self._frame_count += 1

    def end_episode(self) -> int:
        self.dataset.save_episode(task=self.task_description)
        episode_idx = self._episode_count
        self._episode_count += 1
        print(f"[LeRobotRecorder] Episode {episode_idx} saved: {self._frame_count} frames")
        return episode_idx

    def finalize(self) -> Path:
        self.dataset.finalize()
        local_path = Path(self.dataset.root)
        print(f"[LeRobotRecorder] Dataset finalized at: {local_path}")
        print(f"  Episodes: {self._episode_count}")
        print(f"  Repo ID: {self.repo_id}")
        return local_path

    def push_to_hub(self, private: bool = True) -> str:
        self.dataset.push_to_hub(private=private)
        url = f"https://huggingface.co/datasets/{self.repo_id}"
        print(f"[LeRobotRecorder] Pushed to Hub: {url}")
        return url

    def _compute_delta_q_compensated(
        self, q_compliant: np.ndarray, timestamp: float,
    ) -> np.ndarray:
        if len(self._q_ref_history) == 0:
            return np.zeros(self.n_joints)

        t_lookback = timestamp - self.latency_compensation_s
        history = self._q_ref_history

        if t_lookback <= history[0][0]:
            q_ref_delayed = history[0][1]
        elif t_lookback >= history[-1][0]:
            q_ref_delayed = history[-1][1]
        else:
            q_ref_delayed = None
            for i in range(len(history) - 1):
                t0, q0 = history[i]
                t1, q1 = history[i + 1]
                if t0 <= t_lookback <= t1:
                    alpha = (t_lookback - t0) / max(t1 - t0, 1e-9)
                    q_ref_delayed = (1.0 - alpha) * q0 + alpha * q1
                    break
            if q_ref_delayed is None:
                q_ref_delayed = history[-1][1]

        return q_compliant[:self.n_joints] - q_ref_delayed[:self.n_joints]

    def get_episode_count(self) -> int:
        return self._episode_count
