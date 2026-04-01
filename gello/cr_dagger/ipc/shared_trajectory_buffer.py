"""Lock-free double-buffered trajectory exchange via shared memory."""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Optional

import numpy as np

class SharedTrajectoryBuffer:
    HEADER_BYTES = 8 + 8 + 8  # version(int64), active_slot(int64), t_write(float64)

    def __init__(
        self,
        name: str,
        horizon: int = 32,
        n_joints: int = 6,
        create: bool = False,
    ):
        self.name = name
        self.horizon = horizon
        self.n_joints = n_joints
        
        self.data_bytes = horizon * n_joints * 8
        self.slot_bytes = self.HEADER_BYTES + self.data_bytes
        self.total_bytes = self.slot_bytes * 2
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_bytes)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=name)
        else:
            self.shm = shared_memory.SharedMemory(name=name)

    def write(self, trajectory: np.ndarray, t_write: float) -> None:
        
        active_slot_bytes = self.shm.buf[8:16]
        active_slot = np.frombuffer(active_slot_bytes, dtype=np.int64)[0]
        
        next_slot = 1 - active_slot
        offset = next_slot * self.slot_bytes
        
        # write data header
        header = np.array([1, next_slot], dtype=np.int64)
        t_w = np.array([t_write], dtype=np.float64)
        
        self.shm.buf[offset+16:offset+24] = t_w.tobytes()
        self.shm.buf[offset+24:offset+self.slot_bytes] = trajectory.tobytes()
        
        # switch active
        self.shm.buf[8:16] = np.array([next_slot], dtype=np.int64).tobytes()

    def read(self) -> tuple[np.ndarray, float, bool]:
        
        active_slot = np.frombuffer(self.shm.buf[8:16], dtype=np.int64)[0]
        offset = active_slot * self.slot_bytes
        
        t_write = np.frombuffer(self.shm.buf[offset+16:offset+24], dtype=np.float64)[0]
        traj_data = np.frombuffer(self.shm.buf[offset+24:offset+self.slot_bytes], dtype=np.float64)
        
        traj = traj_data.reshape((self.horizon, self.n_joints)).copy()
        
        return traj, float(t_write), True

    def get_reference(self, t_now: float, action_dt: float = 0.1) -> np.ndarray:
        traj, t_write, _ = self.read()
        
        dt = t_now - t_write
        idx = max(0, min(self.horizon - 1, int(dt / action_dt)))
        idx_next = min(self.horizon - 1, idx + 1)
        
        t_idx = idx * action_dt
        t_next = idx_next * action_dt
        
        if t_next > t_idx:
            alpha = (dt - t_idx) / (t_next - t_idx)
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = 0.0
            
        return traj[idx] + alpha * (traj[idx_next] - traj[idx])

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()
