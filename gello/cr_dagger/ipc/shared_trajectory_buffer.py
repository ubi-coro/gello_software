"""Lock-free double-buffered trajectory exchange via shared memory."""
from __future__ import annotations

from multiprocessing import shared_memory

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
            self.shm = self._create_fresh_segment()
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            if self.shm.size != self.total_bytes:
                size = self.shm.size
                self.shm.close()
                raise ValueError(
                    f"SharedTrajectoryBuffer '{name}' size mismatch: "
                    f"expected {self.total_bytes}, got {size}"
                )

        self._last_version = -1

    def _create_fresh_segment(self) -> shared_memory.SharedMemory:
        """Create a clean shared-memory segment, replacing stale leftovers."""
        try:
            shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self.total_bytes
            )
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=self.name, create=False)
            try:
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self.total_bytes
            )

        shm.buf[:] = b"\x00" * self.total_bytes
        return shm

    def write(self, trajectory: np.ndarray, t_write: float) -> None:
        traj = np.asarray(trajectory, dtype=np.float64)
        expected_shape = (self.horizon, self.n_joints)
        if traj.shape != expected_shape:
            raise ValueError(
                f"trajectory must have shape {expected_shape}, got {traj.shape}"
            )

        active_slot_bytes = self.shm.buf[8:16]
        active_slot = int(np.frombuffer(active_slot_bytes, dtype=np.int64)[0])
        if active_slot not in (0, 1):
            active_slot = 0

        next_slot = 1 - active_slot
        offset = next_slot * self.slot_bytes

        version = int(np.frombuffer(self.shm.buf[0:8], dtype=np.int64)[0]) + 1
        t_w = np.array([t_write], dtype=np.float64)

        # Write inactive slot first, then atomically switch active slot index.
        self.shm.buf[offset:offset + 8] = np.array([version], dtype=np.int64).tobytes()
        self.shm.buf[offset + 8:offset + 16] = np.array([next_slot], dtype=np.int64).tobytes()
        self.shm.buf[offset + 16:offset + 24] = t_w.tobytes()
        self.shm.buf[offset + 24:offset + self.slot_bytes] = traj.tobytes(order="C")

        self.shm.buf[0:8] = np.array([version], dtype=np.int64).tobytes()
        self.shm.buf[8:16] = np.array([next_slot], dtype=np.int64).tobytes()

    def read(self) -> tuple[np.ndarray, float, bool]:
        active_slot = int(np.frombuffer(self.shm.buf[8:16], dtype=np.int64)[0])
        if active_slot not in (0, 1):
            active_slot = 0

        offset = active_slot * self.slot_bytes
        version = int(np.frombuffer(self.shm.buf[offset:offset + 8], dtype=np.int64)[0])
        t_write = np.frombuffer(self.shm.buf[offset + 16:offset + 24], dtype=np.float64)[0]
        traj_data = np.frombuffer(
            self.shm.buf[offset + 24:offset + self.slot_bytes],
            dtype=np.float64,
            count=self.horizon * self.n_joints,
        )

        traj = traj_data.reshape((self.horizon, self.n_joints)).copy()
        is_new = version != self._last_version
        self._last_version = version

        return traj, float(t_write), is_new

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
