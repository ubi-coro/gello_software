"""Shared memory observation snapshot for control loop → policy IPC."""
from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

class SharedObservationSnapshot:
    """Latest-only observation exchange via shared memory."""

    def __init__(
        self,
        name: str,
        n_joints: int = 6,
        img_height: int = 480,
        img_width: int = 640,
        create: bool = False,
    ):
        self.name = name
        self.n_joints = n_joints
        self.img_height = img_height
        self.img_width = img_width
        
        self.offsets = {}
        curr = 0
        
        self.offsets['version'] = curr; curr += 8
        self.offsets['timestamp'] = curr; curr += 8
        self.offsets['q'] = curr; curr += 8 * n_joints
        self.offsets['dq'] = curr; curr += 8 * n_joints
        self.offsets['grip'] = curr; curr += 8
        self.offsets['tau_ext'] = curr; curr += 8 * n_joints
        self.offsets['wrench'] = curr; curr += 48
        
        self.img_size = img_height * img_width * 3
        self._image_shape = (self.img_height, self.img_width, 3)
        self.offsets['image'] = curr; curr += self.img_size
        
        self.total_bytes = curr
        
        if create:
            self.shm = self._create_fresh_segment()
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            if self.shm.size != self.total_bytes:
                size = self.shm.size
                self.shm.close()
                raise ValueError(
                    f"SharedObservationSnapshot '{name}' size mismatch: "
                    f"expected {self.total_bytes}, got {size}"
                )
            
        self.last_read_version = -1

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

    def write(
        self,
        timestamp: float,
        q: np.ndarray,
        dq: np.ndarray,
        grip: float,
        tau_ext: np.ndarray,
        wrench: np.ndarray,
        image: np.ndarray | None = None,
    ) -> None:
        buf = self.shm.buf
        
        version = int(
            np.frombuffer(
                buf[self.offsets['version']:self.offsets['version'] + 8],
                dtype=np.int64,
            )[0]
        ) + 1
        buf[self.offsets['version']:self.offsets['version']+8] = np.array([version], dtype=np.int64).tobytes()
        
        buf[self.offsets['timestamp']:self.offsets['timestamp']+8] = np.array([timestamp], dtype=np.float64).tobytes()
        buf[self.offsets['q']:self.offsets['q']+(8*self.n_joints)] = np.array(q[:self.n_joints], dtype=np.float64).tobytes()
        buf[self.offsets['dq']:self.offsets['dq']+(8*self.n_joints)] = np.array(dq[:self.n_joints], dtype=np.float64).tobytes()
        buf[self.offsets['grip']:self.offsets['grip']+8] = np.array([grip], dtype=np.float64).tobytes()
        buf[self.offsets['tau_ext']:self.offsets['tau_ext']+(8*self.n_joints)] = np.array(tau_ext[:self.n_joints], dtype=np.float64).tobytes()
        buf[self.offsets['wrench']:self.offsets['wrench']+48] = np.array(wrench[:6], dtype=np.float64).tobytes()

        img_view = np.ndarray(
            self._image_shape,
            dtype=np.uint8,
            buffer=buf,
            offset=self.offsets['image'],
        )
        if image is None:
            img_view.fill(0)
            return

        image_arr = np.asarray(image, dtype=np.uint8)
        if image_arr.shape != self._image_shape:
            raise ValueError(
                f"image must have shape {self._image_shape}, got {image_arr.shape}"
            )
        np.copyto(img_view, image_arr)

    def read(self) -> dict | None:
        buf = self.shm.buf
        version = int(
            np.frombuffer(
                buf[self.offsets['version']:self.offsets['version'] + 8],
                dtype=np.int64,
            )[0]
        )
        
        if version == self.last_read_version:
            return None
            
        self.last_read_version = version
        
        timestamp = np.frombuffer(buf[self.offsets['timestamp']:self.offsets['timestamp']+8], dtype=np.float64)[0]
        q = np.frombuffer(buf[self.offsets['q']:self.offsets['q']+(8*self.n_joints)], dtype=np.float64).copy()
        dq = np.frombuffer(buf[self.offsets['dq']:self.offsets['dq']+(8*self.n_joints)], dtype=np.float64).copy()
        grip = np.frombuffer(buf[self.offsets['grip']:self.offsets['grip']+8], dtype=np.float64)[0]
        tau_ext = np.frombuffer(buf[self.offsets['tau_ext']:self.offsets['tau_ext']+(8*self.n_joints)], dtype=np.float64).copy()
        wrench = np.frombuffer(buf[self.offsets['wrench']:self.offsets['wrench']+48], dtype=np.float64).copy()
        
        image = np.ndarray(
            self._image_shape,
            dtype=np.uint8,
            buffer=buf,
            offset=self.offsets['image'],
        ).copy()
        
        return {
            'timestamp': float(timestamp),
            'q': q,
            'dq': dq,
            'grip': float(grip),
            'tau_ext': tau_ext,
            'wrench': wrench,
            'image': image,
        }

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()
