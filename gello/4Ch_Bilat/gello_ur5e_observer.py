"""Leader-side disturbance observer for torque estimation without an F/T sensor.
This module is self-contained so you can prototype external torque estimation on
the GELLO leader without modifying the rest of the stack yet. It follows a
discrete DOB scheme (velocity observer + force observer) using the Pinocchio
model of the leader to supply mass/inertia terms.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pinocchio as pin

class LeaderObserver:
	"""Estimate joint velocities and external joint torques on the leader.

	The observer implements a two-stage filter (velocity and force observers)
	using Tustin-discretized LPF/HPF blocks. It expects the commanded torques
	to be provided with model-based terms included; it subtracts non-linear
	dynamics internally to form the input term tau_u.
	"""
        def __init__(self, urdf_path: str, omega_c: float, dt: float, q0: np.ndarray):
            # Load Pinocchio model for leader dynamics
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            self.nq = self.model.nq
            
            # Observer parameters
            self.omega_c = float(omega_c)
            self.dt = float(dt)
            
            # Persistent filter states
            self.q_prev = np.array(q0, dtype=float)
            self.dtheta_ref_prev = np.zeros(self.nq)
            # Velocity observer states
            self.dtheta_int_hpf_prev = np.zeros(self.nq)
            self.theta_lpf_vob_prev = np.array(q0, dtype=float)
            # Force observer states
            self.tau_u_lpf_prev = np.zeros(self.nq)
            self.theta_lpf_fob_prev = np.array(q0, dtype=float)
            self.temp_lpf_prev = np.zeros(self.nq)
            # Outputs
            self.dtheta_hat = np.zeros(self.nq)
            self.tau_ext_hat = np.zeros(self.nq)
    
        def reset(self, q0: np.ndarray) -> None:
            """Reset filter states to a new starting configuration."""
            self.q_prev = np.array(q0, dtype=float)
            self.dtheta_ref_prev = np.zeros(self.nq)
            self.dtheta_int_hpf_prev.fill(0.0)
            self.theta_lpf_vob_prev = np.array(q0, dtype=float)
            self.tau_u_lpf_prev.fill(0.0)
            self.theta_lpf_fob_prev = np.array(q0, dtype=float)
            self.temp_lpf_prev.fill(0.0)
            self.dtheta_hat.fill(0.0)
            self.tau_ext_hat.fill(0.0)
    
        def update(self, q_measured: np.ndarray, dq_measured: np.ndarray, tau_cmd: np.ndarray, dq_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Run one observer step.
    
            Args:
                q_measured: Current joint positions (rad).
                dq_measured: Current joint velocities (rad/s).
                tau_cmd: Commanded joint torques that were sent to the servos
                    (these typically already include gravity/Coriolis/feedforward).
                dq_ref: Reference joint velocities used by your controller.
    
            Returns:
                (dtheta_hat, tau_ext_hat): estimated joint velocities and external torques.
            """
    
            # Discrete filter coefficients (Tustin)
            wc = self.omega_c
            T = self.dt
            alpha_num = 2.0 - wc * T
            alpha_den = 2.0 + wc * T
            beta = (wc * T) / alpha_den
            
            # Model terms
            pin.computeCoriolis(self.model, self.data, q_measured, dq_measured)
            coriolis = self.data.C @ dq_measured
            g = pin.computeGeneralizedGravity(self.model, self.data, q_measured)
            
            # Form input torque without non-linear terms: tau_u = tau_cmd - (C*dq + g)
            tau_u = tau_cmd - coriolis - g
            
            # Mass matrix and its inverse applied to tau_u
            pin.crba(self.model, self.data, q_measured)
            M = self.data.M
            M_inv_tau_u = np.linalg.solve(M, tau_u)
            
            # --- Velocity Observer (VOB) ---
            term1_vel = (alpha_num / alpha_den) * self.dtheta_int_hpf_prev
            term2_vel = (T / alpha_den) * (dq_ref + self.dtheta_ref_prev)
            dtheta_int_hpf = term1_vel + term2_vel
            theta_lpf_vob = (alpha_num / alpha_den) * self.theta_lpf_vob_prev + (
                    2.0 * beta
            ) * (q_measured + self.q_prev)
            dtheta_pdiff_vob = 2.0 * wc * (q_measured - theta_lpf_vob)
            self.dtheta_hat = dtheta_int_hpf + dtheta_pdiff_vob
            
            # --- Force/Disturbance Observer (FOB) ---
            tau_u_lpf = (alpha_num / alpha_den) * self.tau_u_lpf_prev + beta * (
                    M_inv_tau_u + self.tau_u_lpf_prev
            )
            theta_lpf_fob = (alpha_num / alpha_den) * self.theta_lpf_fob_prev + beta * (
                    q_measured + self.q_prev
            )
            dtheta_pdiff_fob = wc * (q_measured - theta_lpf_fob)
            temp = tau_u_lpf + wc * dtheta_pdiff_fob
            temp_lpf = (alpha_num / alpha_den) * self.temp_lpf_prev + beta * (
                    temp + self.temp_lpf_prev
            )
            term_bracket = -temp_lpf + wc * dtheta_pdiff_fob
            self.tau_ext_hat = M @ term_bracket
            
            # --- Update stored states ---
            self.q_prev = np.array(q_measured, dtype=float)
            self.dtheta_ref_prev = np.array(dq_ref, dtype=float)
            self.dtheta_int_hpf_prev = dtheta_int_hpf
            self.theta_lpf_vob_prev = theta_lpf_vob
            self.tau_u_lpf_prev = tau_u_lpf
            self.theta_lpf_fob_prev = theta_lpf_fob
            self.temp_lpf_prev = temp_lpf
            return self.dtheta_hat.copy(), self.tau_ext_hat.copy()

if __name__ == "__main__":
	# Smoke test: run one observer step
	urdf_file = "gello/factr/urdf/GELLO_Assembly_URDF_V3_1/GELLO_Assembly_URDF_V3.urdf"
	q0 = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
	obs = LeaderObserver(urdf_path=urdf_file, omega_c=50.0, dt=0.001, q0=q0)
	q_meas = q0.copy()
	dq_meas = np.zeros_like(q0)
	tau_cmd = np.zeros_like(q0)
	dq_ref = np.zeros_like(q0)
	dq_hat, tau_ext_hat = obs.update(q_meas, dq_meas, tau_cmd, dq_ref)
	print("dtheta_hat:", dq_hat)
	print("tau_ext_hat:", tau_ext_hat)
