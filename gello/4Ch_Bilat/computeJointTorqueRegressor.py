"""Offline identification of leader dynamics (rigid body + friction).

This keeps the workflow local to recorded logs: provide q, dq, tau, dt and a
URDF, get back identified parameters that can be plugged into a regressor in
the observer/FACTR loop.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter


def _compute_ddq(dq: np.ndarray, dt: float, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Differentiate dq with Savitzky-Golay to reduce noise."""

    ddq = np.zeros_like(dq)
    # Clamp window to be odd and not exceed available samples
    wl = min(window_length, dq.shape[0] - (1 - dq.shape[0] % 2))
    if wl < 5:
        wl = 5 if dq.shape[0] >= 5 else dq.shape[0] | 1  # ensure odd
    for j in range(dq.shape[1]):
        ddq[:, j] = savgol_filter(dq[:, j], window_length=wl, polyorder=polyorder, deriv=1, delta=dt)
    return ddq


def _build_friction_regressor(v: np.ndarray) -> np.ndarray:
    """Build per-sample viscous + Coulomb friction regressor."""

    nq = v.shape[0]
    Y_f = np.zeros((nq, 2 * nq))
    for j in range(nq):
        Y_f[j, 2 * j] = v[j]  # viscous term
        Y_f[j, 2 * j + 1] = np.sign(v[j])  # Coulomb term
    return Y_f


def identify_friction_only(
    urdf_path: str,
    recorded_data: Dict[str, np.ndarray],
    include_bias: bool = True,
    ridge_lambda: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Identify viscous/Coulomb friction (and optional torque bias), using URDF gravity.

    This is the simplest identification that matches the typical FACTR usage in this
    codebase: Pinocchio provides gravity (and optionally Coriolis) from the URDF, and
    you learn friction terms that help make the model more realistic at low speeds.

    Model assumed (per joint):
        tau_meas(q, dq) ≈ g_urdf(q) + Fv * dq + Fc * sign(dq) + b

    Args:
        urdf_path: Path to leader URDF.
        recorded_data: dict with keys 'q', 'dq', 'tau', 'dt'. Shapes: (N, nq).
        include_bias: Adds a per-joint constant torque bias parameter.
        ridge_lambda: Optional L2 regularization weight.

    Returns:
        (params, info)
        params contains arrays: fv (nq,), fc (nq,), bias (nq,) if include_bias.
    """

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    qs = np.asarray(recorded_data["q"], dtype=float)
    dqs = np.asarray(recorded_data["dq"], dtype=float)
    taus_meas = np.asarray(recorded_data["tau"], dtype=float)
    _ = recorded_data.get("dt", None)  # not required for friction-only

    assert qs.shape == dqs.shape == taus_meas.shape, "q, dq, tau must have same shape"
    assert qs.shape[1] == model.nq, "State dimension must match model.nq"

    nq = model.nq
    n_bias = nq if include_bias else 0
    n_params = 2 * nq + n_bias

    Y_stack = []
    tau_stack = []

    for k in range(qs.shape[0]):
        q = qs[k]
        v = dqs[k]
        tau_m = taus_meas[k]

        # Use URDF gravity as known term and identify remaining friction/bias
        g = pin.computeGeneralizedGravity(model, data, q)
        tau_res = tau_m - g

        Y_friction = _build_friction_regressor(v)
        if include_bias:
            Y_bias = np.eye(nq)
            Y = np.hstack([Y_friction, Y_bias])
        else:
            Y = Y_friction

        Y_stack.append(Y)
        tau_stack.append(tau_res)

    Y_matrix = np.vstack(Y_stack)  # (N*nq, n_params)
    tau_vector = np.hstack(tau_stack)

    if ridge_lambda > 0.0:
        A = Y_matrix.T @ Y_matrix + ridge_lambda * np.eye(n_params)
        b = Y_matrix.T @ tau_vector
        theta = np.linalg.solve(A, b)
        residuals = np.linalg.norm(Y_matrix @ theta - tau_vector) ** 2
        rank = np.linalg.matrix_rank(Y_matrix)
    else:
        theta, residuals, rank, _ = np.linalg.lstsq(Y_matrix, tau_vector, rcond=None)

    fv = theta[0 : 2 * nq : 2]
    fc = theta[1 : 2 * nq : 2]
    params: Dict[str, np.ndarray] = {"fv": fv, "fc": fc}
    if include_bias:
        params["bias"] = theta[2 * nq : 2 * nq + nq]

    info = {"residuals": float(residuals if np.ndim(residuals) else residuals), "rank": int(rank)}
    return params, info


def identify_dynamics(
    urdf_path: str,
    recorded_data: Dict[str, np.ndarray],
    sg_window: int = 11,
    sg_poly: int = 3,
    ridge_lambda: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Identify rigid-body and friction parameters via linear regression.

    Args:
        urdf_path: Path to leader URDF.
        recorded_data: dict with keys 'q', 'dq', 'tau', 'dt'. Shapes: (N, nq).
        sg_window: Savitzky-Golay window for ddq estimation (odd, will be clamped).
        sg_poly: Savitzky-Golay polynomial order.
        ridge_lambda: Optional L2 regularization weight (0.0 -> plain LS).

    Returns:
        (phi_identified, info) where phi contains [rigid params..., friction params...]
        and info provides residual norm and rank.
    """

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    qs = np.asarray(recorded_data["q"], dtype=float)
    dqs = np.asarray(recorded_data["dq"], dtype=float)
    taus_meas = np.asarray(recorded_data["tau"], dtype=float)
    dt = float(recorded_data["dt"])

    assert qs.shape == dqs.shape == taus_meas.shape, "q, dq, tau must have same shape"
    assert qs.shape[1] == model.nq, "State dimension must match model.nq"

    # 1) Estimate accelerations
    ddqs = _compute_ddq(dqs, dt, window_length=sg_window, polyorder=sg_poly)

    Y_stack = []
    tau_stack = []

    # 2) Build regressor per sample
    for k in range(qs.shape[0]):
        q = qs[k]
        v = dqs[k]
        a = ddqs[k]
        tau_m = taus_meas[k]

        # Rigid-body regressor from Pinocchio
        Y_rigid = pin.computeJointTorqueRegressor(model, data, q, v, a)

        # Friction regressor (viscous + Coulomb)
        Y_friction = _build_friction_regressor(v)

        # Stack
        Y_total = np.hstack([Y_rigid, Y_friction])
        Y_stack.append(Y_total)
        tau_stack.append(tau_m)

    Y_matrix = np.vstack(Y_stack)  # (N*nq, n_params)
    tau_vector = np.hstack(tau_stack)

    # Optional ridge to tame noise/collinearity
    if ridge_lambda > 0.0:
        n_params = Y_matrix.shape[1]
        A = Y_matrix.T @ Y_matrix + ridge_lambda * np.eye(n_params)
        b = Y_matrix.T @ tau_vector
        phi_identified = np.linalg.solve(A, b)
        residuals = np.linalg.norm(Y_matrix @ phi_identified - tau_vector) ** 2
        rank = np.linalg.matrix_rank(Y_matrix)
    else:
        phi_identified, residuals, rank, _ = np.linalg.lstsq(
            Y_matrix, tau_vector, rcond=None
        )

    info = {"residuals": float(residuals if np.ndim(residuals) else residuals), "rank": int(rank)}
    return phi_identified, info


# --- Usage hint ---
# After identification, in the observer/FACTR loop you can form
#   tau_model = Y_total_current @ phi_identified
# using the same regressor construction (rigid + friction) with live q, dq, ddq.