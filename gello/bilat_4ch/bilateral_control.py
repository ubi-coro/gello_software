"""
4-Channel Bilateral Control for GELLO + UR5e Teleoperation

Based on Yamane et al. (2025): "Fast Bilateral Teleoperation and Imitation Learning
Using Sensorless Force Control via Accurate Dynamics Model"

This module provides sensorless bilateral control using:
- Velocity Observer (VOB) for velocity estimation
- Force Observer (FOB) for external torque estimation
- 4-channel control law for position/force transparency

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from gello.factr.gello_ur5e_observer import LeaderObserver


@dataclass
class BilateralControllerConfig:
    """Configuration for 4-channel bilateral controller.
    
    Based on Table 3 from Yamane et al. (2025).
    """
    # Position control gains (Eq. 8: Kp = 1/2 * M_d^-1 * K_d)
    kp: float = 800.0  # Position P-gain
    kd: float = 40.0   # Position D-gain (velocity damping)
    
    # Force control: Kf = {2M(θ)}^(-1) computed dynamically
    # This ensures ideal kinesthetic coupling (Eq. 7)
    
    # Observer parameters
    omega_c: float = 50.0  # Cut-off frequency [rad/s]
    zeta: float = 1.0      # Damping coefficient (1.0 = double pole, no vibration)
    
    # Viscous friction coefficients D (from parameter identification)
    # These are per-joint values in Nm*s/rad
    viscous_friction: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.051, 0.088, 0.021, 0.076, 0.029, 0.040])
    )
    
    # Safety limits
    max_torque_per_joint: float = 5.0  # Nm, safety clamp
    
    # Control mode
    use_variable_inertia_gain: bool = True  # Use M(θ)-dependent Kf


class BilateralController:
    """4-Channel Bilateral Controller for GELLO Leader + UR5e Follower.
    
    Implements the control law from Yamane et al. (2025) Equation 8:
    
        τ_ref = M(θ)[Kp(θd-θ) + Kd(dθd-dθ̂) + Kf(τd+τ̂_ext)] 
                - τ̂_ext + C(θ,dθ̂)dθ̂ + Ddθ̂ + g(θ)
    
    Where:
        - θd, dθd, τd are the opposite side's states (leader↔follower)
        - θ̂, dθ̂, τ̂_ext are estimated via observers
        - M, C, g are from identified dynamics model
        - Kp, Kd are position control gains
        - Kf = {2M(θ)}^(-1) for ideal force transparency
    
    The 4-channel architecture achieves:
        - Position synchronization: θ_leader ≈ θ_follower
        - Force transparency: τ_ext_leader ≈ -τ_ext_follower

    """
    
    def __init__(
        self,
        leader_urdf: str,
        follower_urdf: str,
        dt: float,
        config: Optional[BilateralControllerConfig] = None,
        leader_q0: Optional[npt.NDArray[np.float64]] = None,
        follower_q0: Optional[npt.NDArray[np.float64]] = None,
    ):
        """Initialize bilateral controller.
        
        Args:
            leader_urdf: Path to GELLO leader URDF file
            follower_urdf: Path to UR5e follower URDF file
            dt: Control loop timestep [s]
            config: Controller configuration (uses defaults if None)
            leader_q0: Initial leader joint positions [rad]
            follower_q0: Initial follower joint positions [rad]
        """
        self.config = config or BilateralControllerConfig()
        self.dt = dt
        
        # Default initial positions (typical home pose)
        if leader_q0 is None:
            leader_q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        if follower_q0 is None:
            follower_q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        
        self.nq = len(leader_q0)
        
        # Initialize observers for both sides
        self.leader_obs = LeaderObserver(
            urdf_path=leader_urdf,
            omega_c=self.config.omega_c,
            dt=dt,
            q0=leader_q0,
        )
        
        self.follower_obs = LeaderObserver(
            urdf_path=follower_urdf,
            omega_c=self.config.omega_c,
            dt=dt,
            q0=follower_q0,
        )
        
        # Gain matrices (diagonal for joint-space control)
        self.Kp = self.config.kp * np.eye(self.nq)
        self.Kd = self.config.kd * np.eye(self.nq)
        self.D = np.diag(self.config.viscous_friction[:self.nq])
        
        # Cache for debugging/logging
        self._last_leader_tau_ext = np.zeros(self.nq)
        self._last_follower_tau_ext = np.zeros(self.nq)
        self._last_leader_dq_hat = np.zeros(self.nq)
        self._last_follower_dq_hat = np.zeros(self.nq)
    
    def reset(
        self,
        leader_q0: npt.NDArray[np.float64],
        follower_q0: npt.NDArray[np.float64],
    ) -> None:
        """Reset observer states to new initial configurations.
        
        Call this when teleop session restarts or after E-stop.
        """
        self.leader_obs.reset(leader_q0)
        self.follower_obs.reset(follower_q0)
        self._last_leader_tau_ext = np.zeros(self.nq)
        self._last_follower_tau_ext = np.zeros(self.nq)
    
    def _compute_kf(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute force gain Kf = {2M(θ)}^(-1).
        
        This ensures ideal kinesthetic coupling where the operator
        feels the original inertia of the follower (Eq. 7 in paper).
        """
        if self.config.use_variable_inertia_gain:
            return np.linalg.inv(2.0 * M)
        else:
            # Fallback: fixed gain (less accurate but more stable)
            return 0.5 * np.eye(self.nq)
    
    def _get_nonlinear_compensation(
        self,
        obs: LeaderObserver,
        q: npt.NDArray[np.float64],
        dq_hat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute nonlinear dynamics compensation: C(θ,dθ̂)dθ̂ + Ddθ̂ + g(θ).
        
        This is the feedforward term that cancels known dynamics,
        leaving only the "acceleration control" input τ_u.
        """
        # Update Coriolis with estimated velocity (not measured!)
        # This is a key insight from the paper - use dq_hat for consistency
        pin.computeCoriolis(obs.model, obs.data, q, dq_hat)
        coriolis = obs.data.C @ dq_hat
        
        # Gravity
        g = pin.computeGeneralizedGravity(obs.model, obs.data, q)
        
        # Viscous friction
        friction = self.D @ dq_hat
        
        return coriolis + friction + g
    
    def compute_control(
        self,
        # Leader (GELLO) state
        q_leader: npt.NDArray[np.float64],
        dq_leader: npt.NDArray[np.float64],
        tau_cmd_leader: npt.NDArray[np.float64],
        # Follower (UR5e) state
        q_follower: npt.NDArray[np.float64],
        dq_follower: npt.NDArray[np.float64],
        tau_cmd_follower: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Dict]:
        """Compute bilateral control torques for leader and follower.
        
        Implements Equation 8 from Yamane et al. (2025):
        
        τ_ref = M(θ)[Kp(θd-θ) + Kd(dθd-dθ̂) + Kf(τd+τ̂_ext)] 
                - τ̂_ext + C(θ,dθ̂)dθ̂ + Ddθ̂ + g(θ)
        
        Args:
            q_leader: Leader joint positions [rad]
            dq_leader: Leader joint velocities (measured) [rad/s]
            tau_cmd_leader: Last commanded torque to leader [Nm]
            q_follower: Follower joint positions [rad]
            dq_follower: Follower joint velocities (measured) [rad/s]
            tau_cmd_follower: Last commanded torque to follower [Nm]
        
        Returns:
            tau_ref_leader: Reference torque for leader [Nm]
            tau_ref_follower: Reference torque for follower [Nm]
            info: Debug information dict
        """
        # === Observer Updates ===
        # Reference velocity is zero for quasi-static bilateral control
        dq_ref = np.zeros(self.nq)
        
        dq_hat_l, tau_ext_l = self.leader_obs.update(
            q_measured=q_leader,
            dq_measured=dq_leader,
            tau_cmd=tau_cmd_leader,
            dq_ref=dq_ref,
        )
        
        dq_hat_f, tau_ext_f = self.follower_obs.update(
            q_measured=q_follower,
            dq_measured=dq_follower,
            tau_cmd=tau_cmd_follower,
            dq_ref=dq_ref,
        )
        
        # Cache for debugging
        self._last_leader_tau_ext = tau_ext_l.copy()
        self._last_follower_tau_ext = tau_ext_f.copy()
        self._last_leader_dq_hat = dq_hat_l.copy()
        self._last_follower_dq_hat = dq_hat_f.copy()
        
        # === Get Inertia Matrices ===
        pin.crba(self.leader_obs.model, self.leader_obs.data, q_leader)
        M_l = self.leader_obs.data.M.copy()
        
        pin.crba(self.follower_obs.model, self.follower_obs.data, q_follower)
        M_f = self.follower_obs.data.M.copy()
        
        # === Compute Variable Force Gains ===
        Kf_l = self._compute_kf(M_l)
        Kf_f = self._compute_kf(M_f)
        
        # === 4-Channel Control Law (Eq. 8) ===
        # Leader: desires to track follower position and feel follower's external force
        #   θd = θ_follower, dθd = dθ̂_follower, τd = τ_ext_follower
        tau_u_l = M_l @ (
            self.Kp @ (q_follower - q_leader) +
            self.Kd @ (dq_hat_f - dq_hat_l) +
            Kf_l @ (tau_ext_f + tau_ext_l)
        ) - tau_ext_l
        
        # Follower: desires to track leader position and feel leader's external force
        #   θd = θ_leader, dθd = dθ̂_leader, τd = τ_ext_leader
        tau_u_f = M_f @ (
            self.Kp @ (q_leader - q_follower) +
            self.Kd @ (dq_hat_l - dq_hat_f) +
            Kf_f @ (tau_ext_l + tau_ext_f)
        ) - tau_ext_f
        
        # === Add Nonlinear Compensation ===
        tau_ref_l = tau_u_l + self._get_nonlinear_compensation(
            self.leader_obs, q_leader, dq_hat_l
        )
        tau_ref_f = tau_u_f + self._get_nonlinear_compensation(
            self.follower_obs, q_follower, dq_hat_f
        )
        
        # === Safety Clamp ===
        max_tau = self.config.max_torque_per_joint
        tau_ref_l = np.clip(tau_ref_l, -max_tau, max_tau)
        tau_ref_f = np.clip(tau_ref_f, -max_tau, max_tau)
        
        # === Debug Info ===
        info = {
            "dq_hat_leader": dq_hat_l.copy(),
            "dq_hat_follower": dq_hat_f.copy(),
            "tau_ext_leader": tau_ext_l.copy(),
            "tau_ext_follower": tau_ext_f.copy(),
            "position_error": np.linalg.norm(q_leader - q_follower),
            "force_error": np.linalg.norm(tau_ext_l + tau_ext_f),
        }
        
        return tau_ref_l, tau_ref_f, info
    
    def compute_leader_only(
        self,
        q_leader: npt.NDArray[np.float64],
        dq_leader: npt.NDArray[np.float64],
        tau_cmd_leader: npt.NDArray[np.float64],
        q_follower: npt.NDArray[np.float64],
        dq_follower: npt.NDArray[np.float64],
        tau_ext_follower: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], Dict]:
        """Compute only leader torque when follower provides direct torque sensing.
        
        Use this when the follower (e.g., UR5e) has direct joint torque sensing
        via `getActualJointTorques()`, eliminating the need for a follower observer.
        
        Args:
            q_leader: Leader joint positions [rad]
            dq_leader: Leader joint velocities [rad/s]
            tau_cmd_leader: Last commanded torque to leader [Nm]
            q_follower: Follower joint positions [rad]
            dq_follower: Follower joint velocities [rad/s]
            tau_ext_follower: Follower external torques (from sensor) [Nm]
        
        Returns:
            tau_ref_leader: Reference torque for leader [Nm]
            info: Debug information dict
        """
        dq_ref = np.zeros(self.nq)
        
        # Only run leader observer
        dq_hat_l, tau_ext_l = self.leader_obs.update(
            q_measured=q_leader,
            dq_measured=dq_leader,
            tau_cmd=tau_cmd_leader,
            dq_ref=dq_ref,
        )
        
        # Get leader inertia
        pin.crba(self.leader_obs.model, self.leader_obs.data, q_leader)
        M_l = self.leader_obs.data.M.copy()
        Kf_l = self._compute_kf(M_l)
        
        # Control law for leader
        tau_u_l = M_l @ (
            self.Kp @ (q_follower - q_leader) +
            self.Kd @ (dq_follower - dq_hat_l) +
            Kf_l @ (tau_ext_follower + tau_ext_l)
        ) - tau_ext_l
        
        tau_ref_l = tau_u_l + self._get_nonlinear_compensation(
            self.leader_obs, q_leader, dq_hat_l
        )
        
        # Safety clamp
        max_tau = self.config.max_torque_per_joint
        tau_ref_l = np.clip(tau_ref_l, -max_tau, max_tau)
        
        info = {
            "dq_hat_leader": dq_hat_l.copy(),
            "tau_ext_leader": tau_ext_l.copy(),
            "tau_ext_follower": tau_ext_follower.copy(),
            "position_error": np.linalg.norm(q_leader - q_follower),
        }
        
        return tau_ref_l, info


# =============================================================================

# Integration Helper for gravity_compensation.py

# =============================================================================

class BilateralGravityCompensation:
    """Drop-in replacement for force feedback in FACTRGravityCompensation.
    
    This class wraps BilateralController to provide the same interface
    as the existing torque_feedback() method, but uses observer-based
    estimation instead of direct torque sensing.
    
    Usage in gravity_compensation.py:
    
        # In __init__:
        if self.enable_bilateral_control:
            self.bilateral = BilateralGravityCompensation(
                leader_urdf=self.config["arm_teleop"]["leader_urdf"],
                follower_urdf="path/to/ur5e.urdf",
                dt=self.dt,
                q0=self.calibration_joint_pos[:self.num_arm_joints],
            )
        
        # In control_loop_step:
        if self.enable_bilateral_control:
            torque_arm += self.bilateral.compute_feedback(
                q_leader=leader_arm_pos,
                dq_leader=leader_arm_vel,
                tau_cmd_leader=self._last_tau_cmd,
                q_follower=follower_pos,
                dq_follower=follower_vel,
                tau_ext_follower=follower_torques,  # from UR5e
            )
    """
    
    def __init__(
        self,
        leader_urdf: str,
        follower_urdf: str,
        dt: float,
        q0: npt.NDArray[np.float64],
        config: Optional[BilateralControllerConfig] = None,
    ):
        self.controller = BilateralController(
            leader_urdf=leader_urdf,
            follower_urdf=follower_urdf,
            dt=dt,
            config=config,
            leader_q0=q0,
            follower_q0=q0,
        )
        self._last_tau_cmd = np.zeros(len(q0))
    
    def compute_feedback(
        self,
        q_leader: npt.NDArray[np.float64],
        dq_leader: npt.NDArray[np.float64],
        tau_cmd_leader: npt.NDArray[np.float64],
        q_follower: npt.NDArray[np.float64],
        dq_follower: npt.NDArray[np.float64],
        tau_ext_follower: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute bilateral force feedback torque for leader.
        
        Returns only the feedback component (not gravity compensation).
        """
        tau_ref, info = self.controller.compute_leader_only(
            q_leader=q_leader,
            dq_leader=dq_leader,
            tau_cmd_leader=tau_cmd_leader,
            q_follower=q_follower,
            dq_follower=dq_follower,
            tau_ext_follower=tau_ext_follower,
        )
        
        # Subtract gravity compensation (already handled by GC system)
        # Return only the bilateral feedback component
        g = pin.computeGeneralizedGravity(
            self.controller.leader_obs.model,
            self.controller.leader_obs.data,
            q_leader,
        )
        
        return tau_ref - g


# =============================================================================

# Standalone Test

# =============================================================================

if __name__ == "__main__":
    import time
    
    # Test configuration
    leader_urdf = "gello/factr/urdf/GELLO_Assembly_URDF_V3_1/GELLO_Assembly_URDF_V3.urdf"
    follower_urdf = leader_urdf  # Use same URDF for testing
    
    q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    dt = 0.001  # 1kHz
    
    print("Initializing BilateralController...")
    controller = BilateralController(
        leader_urdf=leader_urdf,
        follower_urdf=follower_urdf,
        dt=dt,
        leader_q0=q0,
        follower_q0=q0,
    )
    
    print("\nRunning simulation (press Ctrl+C to stop)...")
    
    # Simulate small position offset
    q_l = q0.copy()
    q_f = q0.copy()
    q_f[0] += 0.1  # 0.1 rad offset on joint 1
    
    dq_l = np.zeros(6)
    dq_f = np.zeros(6)
    tau_cmd_l = np.zeros(6)
    tau_cmd_f = np.zeros(6)
    
    try:
        for i in range(100):
            tau_l, tau_f, info = controller.compute_control(
                q_leader=q_l,
                dq_leader=dq_l,
                tau_cmd_leader=tau_cmd_l,
                q_follower=q_f,
                dq_follower=dq_f,
                tau_cmd_follower=tau_cmd_f,
            )
            
            if i % 10 == 0:
                print(f"\n[Step {i}]")
                print(f"  Position error: {info['position_error']:.4f} rad")
                print(f"  Force error: {info['force_error']:.4f} Nm")
                print(f"  τ_leader: {[f'{t:.3f}' for t in tau_l]}")
                print(f"  τ_follower: {[f'{t:.3f}' for t in tau_f]}")
            
            # Update for next iteration
            tau_cmd_l = tau_l
            tau_cmd_f = tau_f
            
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nTest stopped.")
    
    print("\nTest complete!")