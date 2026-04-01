"""Leader-side disturbance observer for torque estimation without an F/T sensor.

Reworked to follow Yamane et al. Algorithm 1 internally while preserving
the original LeaderObserver API.  The previous transfer-function-form
implementation had several filter-coefficient bugs (incorrect ζ scaling,
previous-output/input confusion in bilinear LPFs).  This version
eliminates them by using the pseudocode from Algorithm 1 directly.

References
----------
Yamane et al., "Design and Experimental Validation of Sensorless
4-Channel Bilateral Teleoperation for Low-Cost Manipulators", 2026.
  - Algorithm 1, Eq. 2, 15, 16, 17, 20.

"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pinocchio as pin


class LeaderObserver:
    """Estimate joint velocities and external joint torques on the leader.

    Internally follows Yamane et al. Algorithm 1:
        1. Velocity Observer (VOB) — complementary filter (Eq. 16)
        2. Force Observer (FOB) — disturbance estimate (Eq. 20)

    All filters use bilinear (Tustin) discretisation with a single
    tuning parameter ωc (cutoff frequency) and ζ (damping ratio, default 1).

    API note
    --------
    The ``update()`` signature accepts ``dq_measured`` and ``dq_ref`` for
    backward compatibility, but **neither is used**.  Velocity is estimated
    internally, and the acceleration reference is derived from ``tau_cmd``
    via Eq. 17.
    """

    def __init__(
        self,
        urdf_path: str,
        omega_c: float,
        dt: float,
        q0: np.ndarray,
        zeta: float = 1.0,
    ):
        # --- Pinocchio model ---
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq: int = self.model.nq
        self.nv: int = self.model.nv

        # --- Observer parameters ---
        self.omega_c = float(omega_c)
        self.dt = float(dt)
        self.zeta = float(zeta)

        # --- State allocation ---
        self._n_active = min(len(q0), self.nv)
        self._init_states(q0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_q(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros(self.nq)
        n = min(len(q), self.nq)
        out[:n] = q[:n]
        return out

    def _pad_v(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.nv)
        n = min(len(v), self.nv)
        out[:n] = v[:n]
        return out

    def _init_states(self, q0: np.ndarray) -> None:
        n = self.nv

        # Previous joint position measurement
        self.q_prev = self._pad_q(q0)

        # --- VOB states (Algorithm 1, velocity-estimation block) ---
        self.ddq_ref_prev = np.zeros(n)       # previous raw ¨q_ref
        self.ddq_ref_lpf_prev = np.zeros(n)   # previous LPF-filtered ¨q_ref
        self.q_lpf_prev = self._pad_q(q0)     # previous LPF-filtered position

        # --- FOB states (Algorithm 1, force-estimation block) ---
        self.dq_pred_prev = np.zeros(n)        # previous velocity prediction

        # --- Outputs ---
        self.dtheta_hat = np.zeros(n)          # estimated joint velocity
        self.tau_ext_hat = np.zeros(n)         # estimated external torque

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, q0: np.ndarray) -> None:
        """Reset all filter states to a new starting configuration."""
        self._n_active = min(len(q0), self.nv)
        self._init_states(q0)

    def update(
        self,
        q_measured: np.ndarray,
        dq_measured: np.ndarray,
        tau_cmd: np.ndarray,
        dq_ref: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one observer step (Algorithm 1, Yamane et al.).

        Parameters
        ----------
        q_measured : array
            Current joint positions [rad] from encoders.
        dq_measured : array
            **Unused** — kept for API compatibility.  Velocity is
            estimated internally by the observer.
        tau_cmd : array
            Total torque command sent to the servos [Nm], including
            gravity / Coriolis / feedforward terms (Yamane Eq. 2).
        dq_ref : array
            **Unused** — kept for API compatibility.  The acceleration
            reference is computed internally from *tau_cmd* (Eq. 17).

        Returns
        -------
        dtheta_hat : array
            Estimated joint velocities [rad/s].
        tau_ext_hat : array
            Estimated external joint torques [Nm].
        """
        # Remember original size so we can slice the output
        input_size_v = min(len(q_measured), self.nv)

        # --- Pad inputs to model size ---
        q = self._pad_q(q_measured)
        tau = self._pad_v(tau_cmd)

        Ts = self.dt
        wc = self.omega_c
        zeta = self.zeta

        # =============================================================
        # Bilinear LPF coefficient  (1st-order, cutoff = 2ζωc)
        #   α = (2 − 2ζωc·Ts) / (2 + 2ζωc·Ts)
        #   gain = (1−α)/2 = 2ζωc·Ts / (2 + 2ζωc·Ts)
        # =============================================================
        two_zeta_wc = 2.0 * zeta * wc
        alpha = (2.0 - two_zeta_wc * Ts) / (2.0 + two_zeta_wc * Ts)
        half_1ma = (1.0 - alpha) / 2.0          # correct for all ζ

        # =============================================================
        # 1. Model terms
        # =============================================================
        pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T        # symmetrise

        # h(q, ˆ˙q) = C(q, ˆ˙q)·ˆ˙q + g(q)
        h = pin.rnea(
            self.model, self.data, q, self.dtheta_hat, np.zeros(self.nv)
        )

        # =============================================================
        # 2. τ_u  and  acceleration reference  ¨q_ref   (Eq. 2 & 17)
        # =============================================================
        tau_u = tau - h                                     # Eq. 2, D=0
        ddq_ref = np.linalg.solve(M, tau_u + self.tau_ext_hat)  # Eq. 17

        # =============================================================
        # 3. VELOCITY ESTIMATION  (Eq. 16 / Algorithm 1)
        #    ˆ˙q = ¨q^lpf_ref/(2ζωc) + ˙q_lpf
        # =============================================================
        # LPF on acceleration reference
        ddq_ref_lpf = (
            alpha * self.ddq_ref_lpf_prev
            + half_1ma * (ddq_ref + self.ddq_ref_prev)

        )

        # LPF on position
        q_lpf = (
            alpha * self.q_lpf_prev
            + half_1ma * (q + self.q_prev)

        )

        # Filtered numerical differentiation:  2ζωc·(q − q^lpf)
        dq_lpf = two_zeta_wc * (q[: self.nv] - q_lpf[: self.nv])

        # Complementary filter  (Eq. 16)
        self.dtheta_hat = ddq_ref_lpf / two_zeta_wc + dq_lpf

        # =============================================================
        # 4. EXTERNAL FORCE ESTIMATION  (Eq. 20 / Algorithm 1)
        #    ˆd = (ωc/2ζ)·(˙q_lpf − ˙q_pred)
        #    ˆτ_ext = M(q)·ˆd
        # =============================================================
        # Velocity prediction  (trapezoidal integration of ¨q^lpf_ref)
        dq_pred = (
            self.dq_pred_prev
            + (ddq_ref_lpf + self.ddq_ref_lpf_prev) / 2.0 * Ts

        )

        # Disturbance in acceleration domain
        d_hat = (wc / (2.0 * zeta)) * (dq_lpf - dq_pred)

        # External torque  (Eq. 15)
        self.tau_ext_hat = M @ d_hat

        # =============================================================
        # 5. Store states for next step
        # =============================================================
        self.q_prev = q.copy()
        self.ddq_ref_prev = ddq_ref.copy()
        self.ddq_ref_lpf_prev = ddq_ref_lpf.copy()
        self.q_lpf_prev = q_lpf.copy()
        self.dq_pred_prev = dq_pred.copy()

        return (
            self.dtheta_hat[:input_size_v].copy(),
            self.tau_ext_hat[:input_size_v].copy(),
        )


# =====================================================================

# Smoke test

# =====================================================================

if __name__ == "__main__":
    urdf_file = (
        "gello/factr/urdf/GELLO_Assembly_URDF_V5/GELLO_Assembly_URDF_V5.urdf"
    )
    q0 = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    obs = LeaderObserver(urdf_path=urdf_file, omega_c=50.0, dt=0.001, q0=q0)

    q_meas = q0.copy()
    dq_meas = np.zeros_like(q0)
    tau_cmd = np.zeros_like(q0)
    dq_ref = np.zeros_like(q0)

    dq_hat, tau_ext_hat = obs.update(q_meas, dq_meas, tau_cmd, dq_ref)
    print("dtheta_hat:", dq_hat)
    print("tau_ext_hat:", tau_ext_hat)
