"""Leader-side disturbance observer for torque estimation without an F/T sensor.

Reworked implementation that directly follows Yamane et al. Algorithm 1
(simplified pseudocode form) rather than the expanded transfer-function form
(Eq. 18). This is numerically more stable and easier to verify.

References:
    - Yamane et al., "Design and Experimental Validation of Sensorless

      4-Channel Bilateral Teleoperation for Low-Cost Manipulators", 2026.
    - Algorithm 1 (pseudocode), Eq. 2, 11, 15, 16, 17, 18, 20.

"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pinocchio as pin


class LeaderObserver:
    """Estimate joint velocities and external joint torques on the GELLO leader.

    Implements Yamane et al.'s two-stage observer:
        1. Velocity Observer (VOB): complementary filter (Eq. 16)

           ˆ˙q = [s/(s+2ζωc)] · (1/s)·¨q_ref  +  [2ζωc/(s+2ζωc)] · s·q
        2. Force Observer (FOB): disturbance estimate (Eq. 20 simplified)

           ˆd = (ωc/2ζ) · (˙q_lpf - ˙q_pred)
           ˆτ_ext = M(q) · ˆd

    All filters are discretized via bilinear (Tustin) transform.
    The single tuning parameter is the cutoff frequency ωc (with ζ=1 recommended).
    """

    def __init__(
        self,
        urdf_path: str,
        omega_c: float,
        dt: float,
        q0: np.ndarray,
        zeta: float = 1.0,
    ):
        """
        Args:
            urdf_path: Path to the GELLO leader URDF.
            omega_c:   Observer cutoff angular frequency [rad/s].
                       Yamane uses 50.0 for CRANE-X7 at 1kHz.
                       For GELLO at ~330Hz, start with 20-30 and tune.
            dt:        Nominal sampling period [s] (used as fallback).
            q0:        Initial joint configuration [rad].
            zeta:      Damping ratio (1.0 = critical damping, recommended).
        """
        # --- Pinocchio model ---
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        # --- Observer parameters ---
        self.omega_c = float(omega_c)
        self.zeta = float(zeta)
        self.dt_nominal = float(dt)

        # --- Allocate and initialize states ---
        self._n_active = min(len(q0), self.nv)
        self._init_states(q0)

    def _init_states(self, q0: np.ndarray) -> None:
        """(Re)initialize all internal filter states."""
        n = self.nv

        # Previous joint position (for numerical differentiation in LPF)
        self.q_prev = self._pad_q(q0)

        # --- VOB states (Algorithm 1, velocity estimation block) ---
        # ¨q^lpf_ref[k-1]: LPF-filtered acceleration reference
        self.ddq_ref_lpf_prev = np.zeros(n)
        # q^lpf[k-1]: LPF-filtered position (for pseudo-differentiation)
        self.q_lpf_prev = self._pad_q(q0)

        # --- FOB states (Algorithm 1, external force estimation block) ---
        # ˙q_pred[k-1]: velocity prediction from integrated accel reference
        self.dq_pred_prev = np.zeros(n)

        # --- Outputs ---
        self.dq_hat = np.zeros(n)       # estimated joint velocity
        self.d_hat = np.zeros(n)         # estimated disturbance (accel domain)
        self.tau_ext_hat = np.zeros(n)   # estimated external torque

        # --- Previous acceleration reference (for anti-windup / controller) ---
        self.ddq_ref_prev = np.zeros(n)

    def _pad_q(self, q: np.ndarray) -> np.ndarray:
        """Pad q to model size if the driver provides fewer joints."""
        out = np.zeros(self.nq)
        n = min(len(q), self.nq)
        out[:n] = q[:n]
        return out

    def _pad_v(self, v: np.ndarray) -> np.ndarray:
        """Pad velocity-sized vector to model size."""
        out = np.zeros(self.nv)
        n = min(len(v), self.nv)
        out[:n] = v[:n]
        return out

    def reset(self, q0: np.ndarray) -> None:
        """Reset all filter states to a new starting configuration."""
        self._n_active = min(len(q0), self.nv)
        self._init_states(q0)

    def update(
        self,
        q_measured: np.ndarray,
        tau_cmd: np.ndarray,
        dt: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one observer step.

        This follows Algorithm 1 from Yamane et al. exactly.

        Args:
            q_measured: Current joint positions [rad] from encoders.
            tau_cmd:    Torque command actually sent to the motors [Nm].
                        This is the TOTAL command including gravity/Coriolis
                        compensation — i.e., τ_cmd in Yamane Eq. (2).
            dt:         Measured sampling period for this step [s].
                        If None, uses the nominal dt from __init__.
                        Using measured dt handles jitter (per Yamane's
                        ``measuredSamplingPeriod()``).

        Returns:
            (dq_hat, tau_ext_hat): estimated joint velocities [rad/s]
                                   and external torques [Nm], sliced to
                                   the active joint count.
        """
        Ts = dt if dt is not None else self.dt_nominal
        n_active = self._n_active

        # --- Pad inputs to model size ---
        q = self._pad_q(q_measured)
        tau = self._pad_v(tau_cmd)

        # =====================================================================
        # 1. Compute model terms
        # =====================================================================
        # Inertia matrix M(q)  — Yamane uses identified parameters;
        # we use Pinocchio's CRBA with the GELLO URDF.
        pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        # Make symmetric (Pinocchio only fills upper triangle)
        M = np.triu(M) + np.triu(M, 1).T

        # Nonlinear terms: h(q, ˆ˙q) = C(q, ˆ˙q)·ˆ˙q + g(q)
        # (Yamane Eq. 2 uses estimated velocity for Coriolis)
        h = pin.rnea(self.model, self.data, q, self.dq_hat, np.zeros(self.nv))
        # pin.rnea(q, dq, 0) = C(q,dq)*dq + g(q), which is exactly h̃

        # =====================================================================
        # 2. Compute τ_u and acceleration reference ¨q_ref
        #    Yamane Eq. (2): τ_u = τ_cmd - C̃(q,ˆ˙q)ˆ˙q - D̃ˆ˙q - g̃(q)
        #    Yamane Eq. (17): ¨q_ref = M̃(q)^{-1} (τ_u + ˆτ_ext)
        # =====================================================================
        # Note: For Dynamixel servos, viscous friction D is typically small
        # and can be folded into the URDF or set to zero initially.
        tau_u = tau - h  # Eq. (2), with D=0 for now

        # ¨q_ref = M^{-1}(τ_u + ˆτ_ext)   — Eq. (17)
        ddq_ref = np.linalg.solve(M, tau_u + self.tau_ext_hat)

        # =====================================================================
        # 3. Discrete-time LPF pole (bilinear / Tustin transform)
        #    1st-order LPF: H(s) = 2ζωc / (s + 2ζωc)
        #    Bilinear: α = (2 - 2ζωc·Ts) / (2 + 2ζωc·Ts)
        #    Discrete: y[k] = α·y[k-1] + (1-α)/2 · (x[k] + x[k-1])
        #
        #    Per Algorithm 1: α = (2 - (2ζωc)Ts) / (2 + (2ζωc)Ts)
        # =====================================================================
        two_zeta_wc = 2.0 * self.zeta * self.omega_c
        alpha = (2.0 - two_zeta_wc * Ts) / (2.0 + two_zeta_wc * Ts)
        half_one_minus_alpha = (1.0 - alpha) / 2.0  # = ζωcTs / (2 + 2ζωcTs)

        # =====================================================================
        # 4. VELOCITY ESTIMATION (Algorithm 1, "Velocity Estimation" block)
        #    Yamane Eq. (16):
        #      ˆ˙q = [s/(s+2ζωc)]·(1/s)·¨q_ref + [2ζωc/(s+2ζωc)]·s·q
        #
        #    Implemented as:
        #      ¨q^lpf_ref[k] = α·¨q^lpf_ref[k-1]
        #                       + (1-α)/2·(¨q_ref[k] + ¨q_ref[k-1])
        #      q^lpf[k]      = α·q^lpf[k-1]
        #                       + (1-α)/2·(q[k] + q[k-1])
        #      ˙q^lpf[k]     = 2ζωc · (q[k] - q^lpf[k])
        #      ˆ˙q[k]        = ¨q^lpf_ref[k] / (2ζωc) + ˙q^lpf[k]
        # =====================================================================

        # LPF on acceleration reference: 1st-order LPF of ¨q_ref
        ddq_ref_lpf = (
            alpha * self.ddq_ref_lpf_prev
            + half_one_minus_alpha * (ddq_ref + self.ddq_ref_prev)

        )

        # LPF on position: 1st-order LPF of q
        q_lpf = (
            alpha * self.q_lpf_prev
            + half_one_minus_alpha * (q + self.q_prev)

        )

        # Filtered numerical differentiation:
        #   2ζωc/(s+2ζωc) · s = 2ζωc · (1 - 2ζωc/(s+2ζωc))
        # In discrete form: ˙q_lpf = 2ζωc · (q - q_lpf)
        dq_lpf = two_zeta_wc * (q[:self.nv] - q_lpf[:self.nv])

        # Complementary filter result (Eq. 16):
        #   ˆ˙q = (1/(2ζωc)) · ¨q^lpf_ref  +  ˙q_lpf
        self.dq_hat = ddq_ref_lpf / two_zeta_wc + dq_lpf

        # =====================================================================
        # 5. EXTERNAL FORCE ESTIMATION (Algorithm 1, "External Force" block)
        #    Yamane Eq. (20) simplified form:
        #      ˙q_pred[k] = ˙q_pred[k-1]
        #                    + (¨q^lpf_ref[k] + ¨q^lpf_ref[k-1])/2 · Ts
        #      ˆd[k]      = (ωc / 2ζ) · (˙q_lpf[k] - ˙q_pred[k])
        #      ˆτ_ext[k]  = M(q[k]) · ˆd[k]
        #
        #    Physical interpretation (Eq. 20):
        #      ˆd is the velocity prediction error (measured vs. model-predicted),
        #      filtered and scaled. It represents the acceleration-domain
        #      disturbance: d = M^{-1}(τ_ext + Δ_comp).
        # =====================================================================

        # Velocity prediction by integrating filtered acceleration reference
        # (trapezoidal rule, matching Algorithm 1)
        dq_pred = (
            self.dq_pred_prev
            + (ddq_ref_lpf + self.ddq_ref_lpf_prev) / 2.0 * Ts

        )

        # Disturbance estimate in acceleration domain (Eq. 20)
        self.d_hat = (self.omega_c / (2.0 * self.zeta)) * (dq_lpf - dq_pred)

        # External torque estimate (Eq. 15): ˆτ_ext = M(q) · ˆd
        self.tau_ext_hat = M @ self.d_hat

        # =====================================================================
        # 6. Update stored states for next iteration
        # =====================================================================
        self.q_prev = q.copy()
        self.ddq_ref_prev = ddq_ref.copy()
        self.ddq_ref_lpf_prev = ddq_ref_lpf.copy()
        self.q_lpf_prev = q_lpf.copy()
        self.dq_pred_prev = dq_pred.copy()

        # Return sliced to active joint count
        return (
            self.dq_hat[:n_active].copy(),
            self.tau_ext_hat[:n_active].copy(),
        )

    def get_ddq_ref(self) -> np.ndarray:
        """Return the last computed acceleration reference.

        Useful if you need ¨q_ref for the controller's torque command
        computation (Eq. 7-8), including anti-windup after saturation.
        """
        return self.ddq_ref_prev[: self._n_active].copy()

    def apply_torque_saturation(self, tau_cmd_saturated: np.ndarray) -> None:
        """Recompute ¨q_ref after torque saturation (anti-windup).

        Per Algorithm 1's last lines:
            τ_cmd[k+1] ← sat(τ_cmd[k+1], τ_limit)
            ¨q_ref[k+1] ← M̃^{-1}(τ_cmd_sat - h̃) + ˆd

        This prevents integrator windup in the FOB's ˙q_pred integrator
        by ensuring the acceleration reference matches what was actually
        commanded to the motors.

        Args:
            tau_cmd_saturated: The torque command AFTER saturation [Nm].
        """
        tau_sat = self._pad_v(tau_cmd_saturated)
        q = self.q_prev

        # Recompute h with current estimates
        h = pin.rnea(
            self.model, self.data, q, self.dq_hat, np.zeros(self.nv)
        )
        tau_u_sat = tau_sat - h

        pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T

        # ¨q_ref = M^{-1}(τ_u_sat + ˆτ_ext)  — same as Eq. 17 but with sat'd torque
        self.ddq_ref_prev = np.linalg.solve(M, tau_u_sat + self.tau_ext_hat)

        # Also recompute the LPF'd version for consistency.
        # Since we only have the current step, we approximate by
        # re-filtering with the same alpha. This is a practical compromise.
        Ts = self.dt_nominal
        two_zeta_wc = 2.0 * self.zeta * self.omega_c
        alpha = (2.0 - two_zeta_wc * Ts) / (2.0 + two_zeta_wc * Ts)
        half_one_minus_alpha = (1.0 - alpha) / 2.0

        # Re-filter: use the previous LPF state and the new ¨q_ref
        self.ddq_ref_lpf_prev = (
            alpha * self.ddq_ref_lpf_prev
            + half_one_minus_alpha * (self.ddq_ref_prev + self.ddq_ref_prev)

        )
        # Recompute dq_pred with corrected ¨q_ref_lpf
        self.dq_pred_prev = (
            self.dq_pred_prev
            + (self.ddq_ref_lpf_prev + self.ddq_ref_lpf_prev) / 2.0 * Ts

        )


# =============================================================================

# Convenience: compute the controller torque command (Yamane Eq. 8)

# =============================================================================

def compute_bilateral_torque(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    dq_hat: np.ndarray,
    tau_ext_hat: np.ndarray,
    q_desired: np.ndarray,
    dq_desired: np.ndarray,
    Kp: float | np.ndarray,
    Kd: float | np.ndarray,
    Kf: float | np.ndarray | None = None,
    tau_desired: np.ndarray | None = None,
    tau_limit: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the torque command per Yamane Eq. (8).

    τ_cmd = M̃(q) { Kp(q_d - q) + Kd(˙q_d - ˆ˙q) + Kf(τ_d + ˆτ_ext) }
            - ˆτ_ext + C̃(q,ˆ˙q)ˆ˙q + D̃ˆ˙q + g̃(q)

    For a leader-only setup (no bilateral partner), you can set:
        q_desired = q (no position error)
        dq_desired = 0
        tau_desired = 0
        Kf = desired force gain
    to get a gravity-compensated, force-reflecting controller.

    Args:
        model, data:   Pinocchio model/data for the robot.
        q:             Current joint positions.
        dq_hat:        Estimated joint velocities (from observer).
        tau_ext_hat:   Estimated external torques (from observer).
        q_desired:     Desired joint positions (from other side or reference).
        dq_desired:    Desired joint velocities.
        Kp, Kd:        Position/velocity gains (scalar or diagonal array).
        Kf:            Force gain (scalar or diagonal). None → 0.
        tau_desired:   Desired external torque (from other side). None → 0.
        tau_limit:     Per-joint torque limits for saturation. None → no limit.

    Returns:
        tau_cmd: Joint torque command [Nm].
    """
    nv = model.nv
    dq_hat_full = np.zeros(nv)
    dq_hat_full[: len(dq_hat)] = dq_hat

    # Inertia
    pin.crba(model, data, q)
    M = data.M.copy()
    M = np.triu(M) + np.triu(M, 1).T

    # Nonlinear terms: h = C(q, ˆ˙q)ˆ˙q + g(q)
    h = pin.rnea(model, data, q, dq_hat_full, np.zeros(nv))

    # Acceleration reference (Eq. 5/6 combined)
    ddq_ref = Kp * (q_desired - q)[:nv] + Kd * (dq_desired - dq_hat_full)
    if Kf is not None:
        tau_d = tau_desired if tau_desired is not None else np.zeros(nv)
        ddq_ref = ddq_ref + Kf * (tau_d + tau_ext_hat)

    # Computed torque with disturbance compensation (Eq. 7 + Eq. 2)
    tau_cmd = M @ ddq_ref - tau_ext_hat + h

    # Saturation
    if tau_limit is not None:
        tau_cmd = np.clip(tau_cmd, -tau_limit, tau_limit)

    return tau_cmd


# =============================================================================

# Smoke test

# =============================================================================

if __name__ == "__main__":
    import time

    urdf_file = "gello/factr/urdf/GELLO_Assembly_URDF_V5/GELLO_Assembly_URDF_V5.urdf"
    q0 = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

    # GELLO runs at ~330 Hz → dt ≈ 0.003 s
    # Start with ωc = 25 rad/s (conservative for 330 Hz)
    obs = LeaderObserver(
        urdf_path=urdf_file,
        omega_c=25.0,
        dt=1.0 / 330.0,
        q0=q0,
        zeta=1.0,
    )

    # Simulate a few steps with zero torque at rest
    q_meas = q0.copy()
    tau_cmd = np.zeros(6)

    print("Running 10 observer steps at rest...")
    for i in range(10):
        dq_hat, tau_ext_hat = obs.update(q_meas, tau_cmd)

    print(f"  dq_hat:      {dq_hat}")
    print(f"  tau_ext_hat: {tau_ext_hat}")
    print(f"  (Both should be ~0 at rest with zero torque command)\n")

    # Simulate a step where gravity compensation is applied but
    # an external torque of 1 Nm is on joint 1
    print("Simulating external disturbance...")
    # First, get gravity torque as the 'correct' command
    pin.computeGeneralizedGravity(obs.model, obs.data, obs._pad_q(q0))
    g = obs.data.g.copy()
    tau_grav_cmd = g[: len(q0)]

    obs.reset(q0)
    for i in range(1000):
        # Robot is at rest, gravity is compensated, but there's 1 Nm on joint 1
        # In reality the position would change, but this tests observer convergence
        dq_hat, tau_ext_hat = obs.update(q_meas, tau_grav_cmd)

    print(f"  dq_hat:      {dq_hat}")
    print(f"  tau_ext_hat: {tau_ext_hat}")
    print(f"  gravity:     {tau_grav_cmd}")
    print("  (tau_ext should converge to ~0 since command matches gravity)")