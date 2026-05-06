"""Generalized momentum observer for external torque estimation."""
from __future__ import annotations

import numpy as np
import pinocchio as pin


class GeneralizedMomentumObserver:
    """Momentum-based external torque observer for the GELLO leader.

    Uses the commanded torque as input. Human-induced torques are not in
    `tau_cmd` and therefore appear as estimated external torque.
    """

    def __init__(
        self,
        pin_model: pin.Model,
        num_arm_joints: int,
        K_obs: float = 50.0,
        dt: float = 0.002,
        filter_fc: float = 5.0,
    ) -> None:
        self.pin_model = pin_model
        self.pin_data = pin_model.createData()
        self.n = int(num_arm_joints)
        self.K_obs = float(K_obs)
        self.dt = float(dt)

        self._pin_nq = int(pin_model.nq)
        self._pin_nv = int(pin_model.nv)

        self._p_prev = np.zeros(self.n, dtype=float)
        self._tau_ext = np.zeros(self.n, dtype=float)
        self._tau_ext_filtered = np.zeros(self.n, dtype=float)
        self._initialized = False

        if filter_fc > 0.0:
            rc = 1.0 / (2.0 * np.pi * float(filter_fc))
            self._alpha_lp = self.dt / (rc + self.dt)
        else:
            self._alpha_lp = 1.0

    def _pad_q(self, q: np.ndarray) -> np.ndarray:
        q_full = np.zeros(self._pin_nq, dtype=float)
        n = min(len(q), self._pin_nq)
        q_full[:n] = q[:n]
        return q_full

    def _pad_v(self, v: np.ndarray) -> np.ndarray:
        v_full = np.zeros(self._pin_nv, dtype=float)
        n = min(len(v), self._pin_nv)
        v_full[:n] = v[:n]
        return v_full

    def reset(self, q: np.ndarray, dq: np.ndarray) -> None:
        """Reset observer state at the current configuration."""
        q_full = self._pad_q(q)
        v_full = self._pad_v(dq)

        pin.crba(self.pin_model, self.pin_data, q_full)
        M = self.pin_data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T
        self._p_prev = (M[: self.n, : self.n] @ dq[: self.n]).astype(float)

        self._tau_ext = np.zeros(self.n, dtype=float)
        self._tau_ext_filtered = np.zeros(self.n, dtype=float)
        self._initialized = True

    def update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_cmd: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """Update observer and return filtered external torque estimate."""
        if not self._initialized:
            self.reset(q, dq)
            return self._tau_ext_filtered.copy()

        Ts = float(dt if dt is not None else self.dt)
        if Ts <= 0.0:
            Ts = self.dt

        q_full = self._pad_q(q)
        v_full = self._pad_v(dq)
        tau_cmd = np.asarray(tau_cmd, dtype=float)
        if tau_cmd.shape[0] < self.n:
            raise ValueError(f"tau_cmd must have at least {self.n} elements")

        pin.crba(self.pin_model, self.pin_data, q_full)
        M = self.pin_data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T
        p_current = M[: self.n, : self.n] @ dq[: self.n]

        a_zero = np.zeros(self._pin_nv, dtype=float)
        tau_bias_full = pin.rnea(self.pin_model, self.pin_data, q_full, v_full, a_zero)
        bias = tau_bias_full[: self.n]

        dp = p_current - self._p_prev
        residual = dp - (tau_cmd[: self.n] - bias) * Ts
        self._tau_ext = self._tau_ext + self.K_obs * (residual - self._tau_ext * Ts)

        self._p_prev = p_current.copy()

        self._tau_ext_filtered = (
            self._alpha_lp * self._tau_ext
            + (1.0 - self._alpha_lp) * self._tau_ext_filtered
        )
        return self._tau_ext_filtered.copy()

    @property
    def raw_estimate(self) -> np.ndarray:
        """Return the unfiltered tau_ext estimate."""
        return self._tau_ext.copy()
