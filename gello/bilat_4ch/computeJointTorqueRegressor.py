"""Offline identification of leader dynamics (rigid body + friction).

This keeps the workflow local to recorded logs: provide q, dq, tau, dt and a
URDF, get back identified parameters that can be plugged into a regressor in
the observer/FACTR loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _resolve_urdf_path(urdf_path: str) -> str:
    """Resolve URDF path: accept absolute or workspace-relative."""
    p = Path(urdf_path)
    if p.is_absolute() and p.exists():
        return str(p)

    # Resolve relative to repo root (two parents up from this file: gello/..)
    repo_root = Path(__file__).resolve().parents[2]
    cand = (repo_root / p).resolve()
    if cand.exists():
        return str(cand)
    return str(p)


def _load_pinocchio_model(urdf_path: str, package_dirs: Optional[str] = None) -> Tuple[Any, Any]:
    """Load a Pinocchio model with optional package dirs.

    In many URDFs, meshes use package:// paths. Passing package_dirs helps.
    """
    urdf_resolved = _resolve_urdf_path(urdf_path)
    if not os.path.exists(urdf_resolved):
        raise FileNotFoundError(f"URDF not found: {urdf_path} (resolved: {urdf_resolved})")

    if package_dirs is None:
        # Default: directory containing URDF, plus repo root.
        pkg_dirs = [str(Path(urdf_resolved).parent), str(Path(__file__).resolve().parents[2])]
    else:
        pkg_dirs = [p for p in package_dirs.split(os.pathsep) if p]

    model, _, _ = pin.buildModelsFromUrdf(filename=urdf_resolved, package_dirs=pkg_dirs)  # type: ignore[attr-defined]
    data = model.createData()
    return model, data


def _smooth_sign(v: np.ndarray, tanh_eps: float = 0.02) -> np.ndarray:
    """Smooth approximation to sign(v) to avoid numerical chattering near zero."""
    eps = float(max(tanh_eps, 1e-6))
    return np.tanh(v / eps)


def _build_friction_regressor(
    v: np.ndarray,
    *,
    coulomb_mode: str = "sign",
    tanh_eps: float = 0.02,
    min_abs_vel_for_coulomb: float = 0.0,
    include_bias: bool = False,
) -> np.ndarray:
    """Build per-sample viscous + Coulomb friction regressor.

    This produces a per-joint regressor Y_f so that:
      tau_friction ≈ Y_f @ theta

    theta layout (per joint):
      [fv_0, fc_0, fv_1, fc_1, ...] (+ optional biases at end)
    """
    nq = int(v.shape[0])
    n_bias = nq if include_bias else 0
    Y_f = np.zeros((nq, 2 * nq + n_bias), dtype=float)

    if coulomb_mode not in ("sign", "tanh"):
        raise ValueError(f"Unsupported coulomb_mode={coulomb_mode!r} (use 'sign' or 'tanh')")

    for j in range(nq):
        vj = float(v[j])
        Y_f[j, 2 * j] = vj  # viscous

        if abs(vj) < float(min_abs_vel_for_coulomb):
            coul = 0.0
        else:
            coul = float(np.sign(vj)) if coulomb_mode == "sign" else float(_smooth_sign(np.array([vj]), tanh_eps=tanh_eps)[0])

        Y_f[j, 2 * j + 1] = coul

        if include_bias:
            Y_f[j, 2 * nq + j] = 1.0

    return Y_f


def _stack_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    return np.vstack(blocks) if len(blocks) else np.zeros((0, 0), dtype=float)


def _stack_targets(targets: list[np.ndarray]) -> np.ndarray:
    return np.hstack(targets) if len(targets) else np.zeros((0,), dtype=float)


def load_npz_recording(path: str) -> Dict[str, np.ndarray]:
    """Load a .npz recording (e.g. from scripts/record_physical_behavior.py)."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def identify_friction_only(
    urdf_path: str,
    recorded_data: Dict[str, np.ndarray],
    include_bias: bool = True,
    ridge_lambda: float = 0.0,
    coulomb_mode: str = "sign",
    tanh_eps: float = 0.02,
    min_abs_vel_for_coulomb: float = 0.0,
    package_dirs: Optional[str] = None,
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

    model, data = _load_pinocchio_model(urdf_path, package_dirs=package_dirs)

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
        g = pin.computeGeneralizedGravity(model, data, q)  # type: ignore[attr-defined]
        tau_res = tau_m - g

        Y_friction = _build_friction_regressor(
            v,
            coulomb_mode=coulomb_mode,
            tanh_eps=tanh_eps,
            min_abs_vel_for_coulomb=min_abs_vel_for_coulomb,
            include_bias=include_bias,
        )
        if include_bias:
            Y = Y_friction
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
    coulomb_mode: str = "sign",
    tanh_eps: float = 0.02,
    min_abs_vel_for_coulomb: float = 0.0,
    package_dirs: Optional[str] = None,
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

    model, data = _load_pinocchio_model(urdf_path, package_dirs=package_dirs)

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
        Y_rigid = pin.computeJointTorqueRegressor(model, data, q, v, a)  # type: ignore[attr-defined]

        # Friction regressor (viscous + Coulomb)
        Y_friction = _build_friction_regressor(
            v,
            coulomb_mode=coulomb_mode,
            tanh_eps=tanh_eps,
            min_abs_vel_for_coulomb=min_abs_vel_for_coulomb,
            include_bias=False,
        )

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


def identify_friction_from_free_motion(
    recorded_data: Dict[str, np.ndarray],
    *,
    use_rnea: bool = False,
    include_coriolis: bool = True,
    include_bias: bool = True,
    coulomb_mode: str = "tanh",
    tanh_eps: float = 0.02,
    min_abs_vel_for_coulomb: float = 0.02,
    max_abs_acc: float = 0.8,
    min_abs_vel: float = 0.05,
    ridge_lambda: float = 1e-4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Estimate friction params from a passive/free-motion recording.

    This is a pragmatic, "make it usable" mode for leader tuning when you do NOT
    have measured joint torques. It assumes that during gentle motion (low accel)
    the human-applied torques are small, so:

      tau_friction(q, dq) ≈ -tau_model(q, dq, ddq)

    where tau_model is either:
      - gravity (+ optional coriolis) from URDF logging, OR
      - full RNEA from URDF logging (if available)

    The fit is only meaningful if you record slow, smooth movements with pauses.
    """
    qs = np.asarray(recorded_data.get("joint_positions"), dtype=float)
    dqs = np.asarray(recorded_data.get("joint_velocities"), dtype=float)
    ddqs = np.asarray(recorded_data.get("joint_accelerations"), dtype=float)

    if qs.ndim != 2 or dqs.shape != qs.shape or ddqs.shape != qs.shape:
        raise ValueError("Expected joint_positions/joint_velocities/joint_accelerations with shape (N, nq)")

    gravity = recorded_data.get("urdf_gravity_torques")
    coriolis = recorded_data.get("urdf_coriolis_torques")
    rnea = recorded_data.get("urdf_rnea_torques")

    if use_rnea:
        if rnea is None:
            raise ValueError("use_rnea=True requires urdf_rnea_torques in the recording")
        tau_model = np.asarray(rnea, dtype=float)
    else:
        if gravity is None:
            raise ValueError("Recording missing urdf_gravity_torques; record with scripts/record_physical_behavior.py --urdf ...")
        tau_model = np.asarray(gravity, dtype=float)
        if include_coriolis and coriolis is not None:
            tau_model = tau_model + np.asarray(coriolis, dtype=float)

    nq = qs.shape[1]

    # Pick samples that are informative: low accel, and enough velocity.
    acc_ok = np.all(np.abs(ddqs) < float(max_abs_acc), axis=1)
    vel_ok = np.any(np.abs(dqs) > float(min_abs_vel), axis=1)
    mask = acc_ok & vel_ok

    n_used = int(np.sum(mask))
    if n_used < max(50, 5 * nq):
        raise RuntimeError(
            f"Not enough usable samples for friction fit (used {n_used}). "
            f"Try recording longer and moving smoothly (max_abs_acc={max_abs_acc}, min_abs_vel={min_abs_vel})."
        )

    Y_stack: list[np.ndarray] = []
    tau_stack: list[np.ndarray] = []
    for k in np.where(mask)[0]:
        v = dqs[k]
        # Target: friction should cancel model torque under gentle motion assumption.
        y = -tau_model[k]
        Yk = _build_friction_regressor(
            v,
            coulomb_mode=coulomb_mode,
            tanh_eps=tanh_eps,
            min_abs_vel_for_coulomb=min_abs_vel_for_coulomb,
            include_bias=include_bias,
        )
        Y_stack.append(Yk)
        tau_stack.append(y)

    Y_matrix = _stack_blocks(Y_stack)  # (N*nq, n_params)
    tau_vector = _stack_targets(tau_stack)

    n_params = Y_matrix.shape[1]
    if ridge_lambda > 0.0:
        A = Y_matrix.T @ Y_matrix + float(ridge_lambda) * np.eye(n_params)
        b = Y_matrix.T @ tau_vector
        theta = np.linalg.solve(A, b)
        residuals = float(np.linalg.norm(Y_matrix @ theta - tau_vector) ** 2)
        rank = int(np.linalg.matrix_rank(Y_matrix))
    else:
        theta, residuals, rank, _ = np.linalg.lstsq(Y_matrix, tau_vector, rcond=None)
        residuals = float(residuals if np.ndim(residuals) else residuals)
        rank = int(rank)

    fv = theta[0 : 2 * nq : 2]
    fc = theta[1 : 2 * nq : 2]
    params: Dict[str, np.ndarray] = {"fv": fv, "fc": fc}
    if include_bias:
        params["bias"] = theta[2 * nq : 2 * nq + nq]

    info = {
        "residuals": float(residuals),
        "rank": int(rank),
        "samples_total": int(qs.shape[0]),
        "samples_used": int(n_used),
    }
    return params, info


def _print_yaml_suggestion(params: Dict[str, np.ndarray], *, n_joints: int) -> None:
    fv = params.get("fv", np.zeros(n_joints))
    fc = params.get("fc", np.zeros(n_joints))
    bias = params.get("bias", np.zeros(n_joints))

    def fmt(arr: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(x):.6f}" for x in arr[:n_joints]) + "]"

    print("\nYAML suggestion (copy into controller.static_friction_comp):")
    print(f"  friction_feedforward: {fmt(fc)}")
    print(f"  viscous_friction: {fmt(fv)}")
    print("  # optional (not currently used in gravity_compensation.py):")
    print(f"  # torque_bias: {fmt(bias)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute friction parameters from a recorded leader motion (.npz)",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to .npz recording (from scripts/record_physical_behavior.py)",
    )
    parser.add_argument(
        "--mode",
        choices=["free-motion"],
        default="free-motion",
        help="Identification mode (default: free-motion)",
    )
    parser.add_argument("--use-rnea", action="store_true", help="Use urdf_rnea_torques if present")
    parser.add_argument(
        "--no-coriolis",
        action="store_true",
        help="Don’t add urdf_coriolis_torques (if available) in free-motion mode",
    )
    parser.add_argument("--no-bias", action="store_true", help="Don’t fit a constant torque bias")

    # Friction model options
    parser.add_argument("--coulomb-mode", choices=["sign", "tanh"], default="tanh")
    parser.add_argument("--tanh-eps", type=float, default=0.02)
    parser.add_argument("--min-abs-vel-for-coulomb", type=float, default=0.02)

    # Sample selection
    parser.add_argument("--max-abs-acc", type=float, default=0.8)
    parser.add_argument("--min-abs-vel", type=float, default=0.05)

    # Solver
    parser.add_argument("--ridge", type=float, default=1e-4)

    args = parser.parse_args()

    rec_path = args.input
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Recording not found: {rec_path}")

    rec = load_npz_recording(rec_path)
    qs = np.asarray(rec.get("joint_positions"))
    if qs.ndim != 2:
        raise ValueError("Recording missing joint_positions")
    n_joints = int(qs.shape[1])

    if args.mode == "free-motion":
        params, info = identify_friction_from_free_motion(
            rec,
            use_rnea=bool(args.use_rnea),
            include_coriolis=not bool(args.no_coriolis),
            include_bias=not bool(args.no_bias),
            coulomb_mode=str(args.coulomb_mode),
            tanh_eps=float(args.tanh_eps),
            min_abs_vel_for_coulomb=float(args.min_abs_vel_for_coulomb),
            max_abs_acc=float(args.max_abs_acc),
            min_abs_vel=float(args.min_abs_vel),
            ridge_lambda=float(args.ridge),
        )

        print("\n=== Friction fit (free-motion) ===")
        print(f"Recording: {rec_path}")
        print(f"Joints: {n_joints}")
        print(f"Samples used: {info.get('samples_used')} / {info.get('samples_total')}")
        print(f"Residuals: {info.get('residuals'):.6e}  rank: {info.get('rank')}")
        print("\nEstimated parameters:")
        print("  fv (viscous):", np.array2string(params["fv"], precision=6, floatmode="fixed"))
        print("  fc (coulomb):", np.array2string(params["fc"], precision=6, floatmode="fixed"))
        if "bias" in params:
            print("  bias:", np.array2string(params["bias"], precision=6, floatmode="fixed"))

        _print_yaml_suggestion(params, n_joints=n_joints)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- Usage hint ---
# After identification, in the observer/FACTR loop you can form
#   tau_model = Y_total_current @ phi_identified
# using the same regressor construction (rigid + friction) with live q, dq, ddq.