"""Joint-space admittance controller for CR-DAgger on GELLO.

Implements:
    M·q̈_c = τ_ext_human + τ_wrench_fb − D·dq_c − K·(q_c − q_ref)

References:
    CR-DAgger (Xu et al. 2025), Eq. 1-2
    Shi et al. 2026, Eq. 11-12 (semi-implicit Euler)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdmittanceParams:
    """Admittance dynamics parameters (constant per run).

    Attributes:
        mass:      Virtual inertia M [kg·m²], scalar applied per joint.
        damping:   Virtual damping D [Nm·s/rad], scalar applied per joint.
        stiffness: Virtual stiffness K [Nm/rad], scalar applied per joint.
    """
    mass: float = 1.0
    damping: float = 5.0
    stiffness: float = 20.0


class JointSpaceAdmittanceController:
    """Admittance controller producing compliant position references."""

    def __init__(
        self,
        params: AdmittanceParams,
        n_joints: int,
        q_init: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
    ):
        self.params = params
        self.n_joints = n_joints
        
        self.q_min = q_min.copy()
        self.q_max = q_max.copy()
        
        self.q_c = np.clip(q_init.copy(), self.q_min, self.q_max)
        self.dq_c = np.zeros(self.n_joints)
        
        self._q_ref_last = self.q_c.copy()

    def step(
        self,
        q_ref: np.ndarray,
        tau_ext_human: np.ndarray,
        dt: float,
        tau_wrench_fb: np.ndarray | None = None,
    ) -> np.ndarray:
        
        tau_total = tau_ext_human - self.params.damping * self.dq_c - self.params.stiffness * (self.q_c - q_ref)
        if tau_wrench_fb is not None:
            tau_total += tau_wrench_fb
            
        ddq_c = tau_total / self.params.mass

        # Semi-implicit Euler
        self.dq_c = self.dq_c + ddq_c * dt
        self.q_c = self.q_c + self.dq_c * dt
        
        self.q_c = np.clip(self.q_c, self.q_min, self.q_max)
        self._q_ref_last = q_ref.copy()
        
        return self.q_c.copy()

    def get_delta_q(self) -> np.ndarray:
        return self.q_c - self._q_ref_last

    def get_state(self) -> dict:
        return {
            'q_c': self.q_c.copy(),
            'dq_c': self.dq_c.copy(),
            'q_ref_last': self._q_ref_last.copy(),
            'delta_q': self.get_delta_q(),
        }

    def reset(self, q_init: np.ndarray) -> None:
        self.q_c = np.clip(q_init.copy(), self.q_min, self.q_max)
        self.dq_c = np.zeros(self.n_joints)
        self._q_ref_last = self.q_c.copy()


class WrenchFeedbackMapper:
    """Map UR5e F/T wrench to GELLO joint-space torques."""

    def __init__(
        self,
        n_leader_joints: int,
        map_index: np.ndarray,
        map_signs: np.ndarray,
        feedback_gain: float = 0.5,
    ):
        self.n_leader_joints = n_leader_joints
        self.map_index = map_index
        self.map_signs = map_signs
        self.feedback_gain = feedback_gain
        self._tare_torque = np.zeros(6)

    def update(
        self,
        wrench_ur5e: np.ndarray,
        jacobian_ur5e: np.ndarray,
    ) -> np.ndarray:
        
        tau_ur5e = jacobian_ur5e.T @ (wrench_ur5e - self._tare_torque)
        tau_gello = np.zeros(self.n_leader_joints)
        
        for follower_idx in range(min(len(self.map_index), 6)):
            leader_idx = self.map_index[follower_idx]
            if leader_idx < self.n_leader_joints:
                tau_gello[leader_idx] = self.feedback_gain * self.map_signs[follower_idx] * tau_ur5e[follower_idx]
                
        return tau_gello

    def set_tare(self, tare_wrench: np.ndarray, tare_jacobian: np.ndarray) -> None:
        self._tare_torque = tare_wrench.copy()
