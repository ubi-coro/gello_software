"""Multi-signal intervention detector for CR-DAgger.

Combines four independent detection signals via majority voting:
1. Torque threshold (Schmitt trigger on ‖τ_ext‖)
2. Delta position deviation (‖q_c − q_ref‖ from admittance)
3. Energy injection (P = τ_ext^T · dq_c > 0)
4. F/T wrench change (optional, from UR5e)

References:
    - CR-DAgger (Xu et al. 2025), Section 3.1 (correction recording with buttons)
    - This module provides a button-free alternative

"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage


@dataclass
class TorqueThresholdParams:
    threshold_high: float = 0.3
    threshold_low: float = 0.15
    onset_steps: int = 3
    offset_steps: int = 10


@dataclass
class DeltaPositionParams:
    position_threshold: float = 0.02
    velocity_threshold: float = 0.05
    onset_time: float = 0.03
    offset_time: float = 0.2


@dataclass
class EnergyInjectionParams:
    energy_threshold: float = 0.005
    decay_rate: float = 10.0
    onset_steps: int = 5
    offset_steps: int = 15


@dataclass
class WrenchChangeParams:
    force_threshold: float = 3.0


@dataclass
class FusedDetectorParams:
    torque: TorqueThresholdParams = field(default_factory=TorqueThresholdParams)
    delta: DeltaPositionParams = field(default_factory=DeltaPositionParams)
    energy: EnergyInjectionParams = field(default_factory=EnergyInjectionParams)
    wrench: WrenchChangeParams = field(default_factory=WrenchChangeParams)
    min_votes: int = 2


class TorqueThresholdDetector:
    """Schmitt trigger on ‖τ_ext‖ with hysteresis."""

    def __init__(self, params: TorqueThresholdParams):
        self.params = params
        self.state = False
        self.active_steps = 0
        self.inactive_steps = 0

    def update(self, tau_ext: np.ndarray) -> bool:
        mag = np.linalg.norm(tau_ext)
        
        if mag > self.params.threshold_high:
            self.active_steps += 1
            self.inactive_steps = 0
        elif mag < self.params.threshold_low:
            self.inactive_steps += 1
            self.active_steps = 0
        else:
            self.active_steps = 0
            self.inactive_steps = 0
            
        if not self.state and self.active_steps >= self.params.onset_steps:
            self.state = True
        elif self.state and self.inactive_steps >= self.params.offset_steps:
            self.state = False
            
        return self.state

    def reset(self) -> None:
        self.state = False
        self.active_steps = 0
        self.inactive_steps = 0


class DeltaPositionDetector:
    """Detect intervention from admittance position deviation."""

    def __init__(self, params: DeltaPositionParams, dt: float):
        self.params = params
        self.dt = dt
        self.state = False
        self.active_time = 0.0
        self.inactive_time = 0.0

    def update(self, q_c: np.ndarray, q_ref: np.ndarray, dq_c: np.ndarray) -> bool:
        pos_err = np.linalg.norm(q_c - q_ref)
        vel_err = np.linalg.norm(dq_c)
        
        if pos_err > self.params.position_threshold or vel_err > self.params.velocity_threshold:
            self.active_time += self.dt
            self.inactive_time = 0.0
        else:
            self.inactive_time += self.dt
            self.active_time = 0.0
            
        if not self.state and self.active_time >= self.params.onset_time:
            self.state = True
        elif self.state and self.inactive_time >= self.params.offset_time:
            self.state = False
            
        return self.state

    def reset(self) -> None:
        self.state = False
        self.active_time = 0.0
        self.inactive_time = 0.0


class EnergyInjectionDetector:
    """Detect intervention from mechanical power injection."""

    def __init__(self, params: EnergyInjectionParams, dt: float):
        self.params = params
        self.dt = dt
        self.state = False
        self.energy_accumulator = 0.0
        self.active_steps = 0
        self.inactive_steps = 0

    def update(self, tau_ext: np.ndarray, dq_c: np.ndarray) -> bool:
        power = np.dot(tau_ext, dq_c)
        power_in = max(0.0, power)
        
        self.energy_accumulator += power_in * self.dt
        self.energy_accumulator -= self.params.decay_rate * self.energy_accumulator * self.dt
        self.energy_accumulator = max(0.0, self.energy_accumulator)
        
        if self.energy_accumulator > self.params.energy_threshold:
            self.active_steps += 1
            self.inactive_steps = 0
        else:
            self.inactive_steps += 1
            self.active_steps = 0
            
        if not self.state and self.active_steps >= self.params.onset_steps:
            self.state = True
        elif self.state and self.inactive_steps >= self.params.offset_steps:
            self.state = False
            
        return self.state

    def reset(self) -> None:
        self.state = False
        self.energy_accumulator = 0.0
        self.active_steps = 0
        self.inactive_steps = 0


class FusedInterventionDetector:
    """Multi-signal fusion with majority voting."""

    def __init__(self, params: FusedDetectorParams, n_joints: int, dt: float):
        self.params = params
        self.n_joints = n_joints
        self.dt = dt
        
        self.torque_det = TorqueThresholdDetector(params.torque)
        self.delta_det = DeltaPositionDetector(params.delta, dt)
        self.energy_det = EnergyInjectionDetector(params.energy, dt)
        self.is_intervening = False
        
        self.last_votes = {}

    def update(
        self,
        tau_ext: np.ndarray,
        q_c: np.ndarray,
        q_ref: np.ndarray,
        dq_c: np.ndarray,
        wrench: np.ndarray | None = None,
    ) -> bool:
        
        v_torque = self.torque_det.update(tau_ext)
        v_delta = self.delta_det.update(q_c, q_ref, dq_c)
        v_energy = self.energy_det.update(tau_ext, dq_c)
        
        v_wrench = False
        if wrench is not None:
            if np.linalg.norm(wrench[:3]) > self.params.wrench.force_threshold:
                v_wrench = True
                
        self.last_votes = {
            'torque': v_torque,
            'delta': v_delta,
            'energy': v_energy,
            'wrench': v_wrench,
        }
        
        votes = sum([v_torque, v_delta, v_energy, v_wrench])
        self.is_intervening = bool(votes >= self.params.min_votes)
        return self.is_intervening

    def get_diagnostics(self) -> dict:
        votes = sum(list(self.last_votes.values()))
        confidence = votes / 4.0
        return {
            'is_intervening': self.is_intervening,
            'confidence': confidence,
            'votes': self.last_votes.copy(),
            'energy_level': self.energy_det.energy_accumulator,
        }

    def reset(self) -> None:
        self.torque_det.reset()
        self.delta_det.reset()
        self.energy_det.reset()
        self.is_intervening = False


class PostHocLabeler:
    """Offline intervention labeling from recorded data."""

    @staticmethod
    def label_episode(
        timestamps: np.ndarray,
        tau_ext_history: np.ndarray,
        delta_q_history: np.ndarray,
        threshold_sigma: float = 2.5,
    ) -> np.ndarray:
        
        tau_mag = np.linalg.norm(tau_ext_history, axis=1)
        smooth_tau = scipy.ndimage.gaussian_filter1d(tau_mag, sigma=5)
        
        mean_tau = np.mean(smooth_tau)
        std_tau = np.std(smooth_tau)
        threshold = mean_tau + threshold_sigma * std_tau
        
        raw_labels = smooth_tau > threshold
        
        labels_closed = scipy.ndimage.binary_closing(raw_labels, structure=np.ones(10))
        labels_final = scipy.ndimage.binary_opening(labels_closed, structure=np.ones(5))
        
        return labels_final
