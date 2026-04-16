"""Correction data recorder for CR-DAgger."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

@dataclass
class CorrectionFrame:
    """Single timestep of correction data."""
    timestamp: float                # monotonic clock [s]
    q_ref: np.ndarray               # policy output (reference)
    dq_ref: np.ndarray              # policy velocity reference (finite diff)
    q_actual: np.ndarray            # GELLO measured position (from encoder)
    dq_actual: np.ndarray           # GELLO measured velocity
    q_compliant: np.ndarray         # admittance output q_c (sent to hardware)
    dq_compliant: np.ndarray        # admittance velocity state
    delta_q: np.ndarray             # q_compliant - q_ref (latency-compensated)
    delta_q_raw: np.ndarray         # q_compliant - q_ref (no compensation)
    tau_ext_gello: np.ndarray       # Shi observer on GELLO
    q_follower: np.ndarray          # UR5e joint positions
    wrench_ur5e: np.ndarray         # F/T sensor [Fx,Fy,Fz,Tx,Ty,Tz]
    is_correction: bool             # fused detector output
    detector_diagnostics: dict      # per-detector votes
    images: dict[str, np.ndarray] | None = None

class CorrectionRecorder:
    def __init__(
        self,
        n_joints: int,
        log_dir: str = "cr_dagger_data",
        latency_compensation_s: float = 0.008,
        q_ref_history_length: int = 100,
    ):
        self.n_joints = n_joints
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.latency_compensation_s = latency_compensation_s
        
        self.q_ref_history = deque(maxlen=q_ref_history_length)
        self.frames = []
        self.episode_id = None
        
    def start_episode(self, episode_id: str | None = None) -> None:
        self.episode_id = episode_id or f"episode_{int(time.time())}"
        self.frames = []
        self.q_ref_history.clear()
        
    def record(
        self,
        timestamp: float,
        q_ref: np.ndarray,
        q_actual: np.ndarray,
        dq_actual: np.ndarray,
        q_compliant: np.ndarray,
        dq_compliant: np.ndarray,
        tau_ext_gello: np.ndarray,
        q_follower: np.ndarray,
        wrench_ur5e: np.ndarray,
        is_correction: bool,
        detector_diagnostics: dict,
        images: dict[str, np.ndarray] | None = None,
    ) -> None:
        
        self.q_ref_history.append((timestamp, q_ref.copy()))
        
        if len(self.q_ref_history) > 1:
            dt = timestamp - self.q_ref_history[-2][0]
            if dt > 0:
                dq_ref = (q_ref - self.q_ref_history[-2][1]) / dt
            else:
                dq_ref = np.zeros_like(q_ref)
        else:
            dq_ref = np.zeros_like(q_ref)
            
        delta_q_raw = q_compliant - q_ref
        delta_q = self._compute_delta_q_compensated(q_compliant, timestamp)
        
        frame = CorrectionFrame(
            timestamp=timestamp,
            q_ref=q_ref.copy(),
            dq_ref=dq_ref,
            q_actual=q_actual.copy(),
            dq_actual=dq_actual.copy(),
            q_compliant=q_compliant.copy(),
            dq_compliant=dq_compliant.copy(),
            delta_q=delta_q.copy(),
            delta_q_raw=delta_q_raw.copy(),
            tau_ext_gello=tau_ext_gello.copy(),
            q_follower=q_follower.copy(),
            wrench_ur5e=wrench_ur5e.copy(),
            is_correction=is_correction,
            detector_diagnostics=detector_diagnostics.copy(),
            images=images
        )
        self.frames.append(frame)

    def end_episode(self) -> Path | None:
        if not self.frames:
            return None
            
        T = len(self.frames)
        timestamps = np.array([f.timestamp for f in self.frames])
        q_ref = np.array([f.q_ref for f in self.frames])
        q_actual = np.array([f.q_actual for f in self.frames])
        q_compliant = np.array([f.q_compliant for f in self.frames])
        delta_q = np.array([f.delta_q for f in self.frames])
        tau_ext = np.array([f.tau_ext_gello for f in self.frames])
        wrench = np.array([f.wrench_ur5e for f in self.frames])
        is_correction = np.array([f.is_correction for f in self.frames])
        
        detector_votes = np.zeros((T, 4), dtype=bool)
        for i, f in enumerate(self.frames):
            v = f.detector_diagnostics.get('votes', {})
            detector_votes[i] = [v.get('torque', False), v.get('delta', False), v.get('energy', False), v.get('wrench', False)]
            
        filename = self.log_dir / f"{self.episode_id}.npz"
        np.savez_compressed(
            filename,
            timestamps=timestamps,
            q_ref=q_ref,
            q_actual=q_actual,
            q_compliant=q_compliant,
            delta_q=delta_q,
            tau_ext=tau_ext,
            wrench=wrench,
            is_correction=is_correction,
            detector_votes=detector_votes
        )
        
        # OMITTING detailed image saving step for brevity since it uses dummy
        
        return filename

    def _compute_delta_q_compensated(
        self, q_compliant: np.ndarray, timestamp: float
    ) -> np.ndarray:
        
        if self.latency_compensation_s <= 0.0 or len(self.q_ref_history) < 2:
            return q_compliant - self.q_ref_history[-1][1]
            
        t_lookback = timestamp - self.latency_compensation_s
        
        for i in range(len(self.q_ref_history)-1, 0, -1):
            t1, q1 = self.q_ref_history[i]
            t0, q0 = self.q_ref_history[i-1]
            if t0 <= t_lookback <= t1:
                if t1 == t0:
                    q_interp = q1
                else:
                    ratio = (t_lookback - t0) / (t1 - t0)
                    q_interp = q0 + ratio * (q1 - q0)
                return q_compliant - q_interp
                
        if t_lookback < self.q_ref_history[0][0]:
            return q_compliant - self.q_ref_history[0][1]
        return q_compliant - self.q_ref_history[-1][1]
