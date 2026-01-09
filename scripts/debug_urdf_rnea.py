#!/usr/bin/env python3
"""URDF/RNEA Debug Tool (Pinocchio)

Ziel
----
Dieses Skript hilft dabei, URDF-Probleme (Joint-Achsen / Joint-Origins / Frames)
zu identifizieren, wenn RNEA-Gravitationsmomente nicht zu Messdaten passen.

Es macht zwei Dinge:
1) Gibt für eine gewählte Pose die Joint-Achsen im Weltframe aus (aus dem
   Jacobian) und zeigt deren Orientierung relativ zur Gravitation.
2) Führt einfache Sweeps durch (nur q2 bzw. nur q3 variieren), um zu prüfen,
   welche Joint-Torques im URDF dabei dominieren sollten.

Optional kann es eine Mess-JSON laden (aus scripts/measure_holding_torques.py)
und die gespeicherten RNEA-Torques gegen eine frische Pinocchio-Rechnung
validieren.

Beispiele
---------
- Nur URDF sanity sweeps:
  python scripts/debug_urdf_rnea.py --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf

- Joint-Achsen am "home" aus einer Messdatei (Motor-Konvention) ausgeben:
  python scripts/debug_urdf_rnea.py --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf \
      --measurement scripts/recordings/holding_torques_20260102_174756.json --pose home

Hinweise
--------
- Benötigt: numpy, pinocchio
- Erwartet 6 Arm-DOF + optional 1 Gripper-DOF (wie URDF V4)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] pinocchio ist nicht installiert. Installiere es oder nutze eine Umgebung mit pinocchio."
    ) from exc


def motor_to_urdf_angles(motor_angles: np.ndarray) -> np.ndarray:
    """Motor-Konvention -> URDF-Konvention.

    Konsistent mit scripts/measure_holding_torques.py (inkl. J5 Offset Fix).

    Motor-Null: [180°, 0°, 180°, 0°, 180°, 0°]
    """

    if motor_angles.shape[-1] != 6:
        raise ValueError(f"Expected 6 motor angles, got shape {motor_angles.shape}")

    urdf_angles = np.zeros_like(motor_angles)
    urdf_angles[0] = motor_angles[0] - np.pi
    urdf_angles[1] = -motor_angles[1]
    urdf_angles[2] = -(motor_angles[2] - np.pi)
    urdf_angles[3] = motor_angles[3]
    urdf_angles[4] = motor_angles[4] - np.pi
    urdf_angles[5] = motor_angles[5]
    return urdf_angles


def build_model(urdf_path: Path) -> pin.Model:
    if not urdf_path.exists():
        raise FileNotFoundError(str(urdf_path))
    return pin.buildModelFromUrdf(str(urdf_path))


def q_from_arm(model: pin.Model, q_arm: np.ndarray, num_arm: int = 6) -> np.ndarray:
    q = np.zeros(model.nq)
    q[:num_arm] = q_arm
    if model.nq > num_arm:
        q[num_arm] = 0.0
    return q


def rnea_gravity(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    return pin.rnea(model, data, q, v, a)


def print_joint_axes_world(model: pin.Model, data: pin.Data, q: np.ndarray, num_arm: int = 6):
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)

    g_dir = np.array(model.gravity.linear, dtype=float)
    g_norm = float(np.linalg.norm(g_dir))
    if g_norm > 1e-12:
        g_dir = g_dir / g_norm

    print("\nJoint list (Pinocchio idx -> name):")
    for jid in range(1, model.njoints):
        print(f"  {jid}: {model.names[jid]}")

    print("\nWorld angular axis directions (approx, from WORLD Jacobian):")
    for jid in range(1, min(num_arm, model.njoints - 1) + 1):
        J = pin.getJointJacobian(model, data, jid, pin.ReferenceFrame.WORLD)
        # For a 1-DoF revolute joint chain, column (jid-1) corresponds to that joint's axis.
        axis = J[3:6, jid - 1].copy()
        n = float(np.linalg.norm(axis))
        if n > 1e-12:
            axis /= n
        dot = float(axis.dot(g_dir)) if g_norm > 1e-12 else float("nan")
        print(f"  J{jid}: {model.names[jid]:<18} axis={axis}  dot(gravity_dir)={dot:+.3f}")


def sweep_single_joint(
    model: pin.Model,
    num_arm: int,
    joint_index0: int,
    deg_min: float = -180.0,
    deg_max: float = 180.0,
    deg_step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep q[joint_index0] over range while keeping other q=0.

    Returns:
      max_abs_tau_arm: (num_arm,) max |tau| per joint
      argmax_deg: (num_arm,) angle where that max occurs
    """

    data = model.createData()
    angles_deg = np.arange(deg_min, deg_max + 1e-9, deg_step)

    max_abs = np.zeros(num_arm)
    argmax = np.zeros(num_arm)

    for deg in angles_deg:
        q_arm = np.zeros(num_arm)
        q_arm[joint_index0] = np.deg2rad(deg)
        q = q_from_arm(model, q_arm, num_arm=num_arm)
        tau = rnea_gravity(model, data, q)[:num_arm]

        for j in range(num_arm):
            val = abs(float(tau[j]))
            if val > max_abs[j]:
                max_abs[j] = val
                argmax[j] = deg

    return max_abs, argmax


def load_measurement(measurement_path: Path) -> dict:
    return json.loads(measurement_path.read_text())


def iter_measurements(data: dict) -> Iterable[dict]:
    for m in data.get("measurements", []):
        yield m


def find_pose(data: dict, pose_name: str) -> Optional[dict]:
    for m in iter_measurements(data):
        if m.get("name") == pose_name:
            return m
    return None


def _compute_rnea_for_measurements(
    model: pin.Model, measurement_data: dict, num_arm: int = 6
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Compute RNEA gravity torques for all measurements.

    Returns:
      R_recomputed: (N, num_arm)
      R_stored: (N, num_arm) or None if not present
      T: (N, num_arm) measured torques
    """

    data_pin = model.createData()

    meas = list(iter_measurements(measurement_data))
    if not meas:
        raise ValueError("Keine Messungen in JSON")

    if "torque_nm" not in meas[0]:
        raise ValueError("Mess-JSON enthält keine torque_nm Felder")

    T = np.array([m["torque_nm"] for m in meas], dtype=float)

    R_stored: Optional[np.ndarray]
    if "rnea_torque_nm" in meas[0]:
        R_stored = np.array([m["rnea_torque_nm"] for m in meas], dtype=float)
    else:
        R_stored = None

    M = np.array([m["position_rad"] for m in meas], dtype=float)

    R_recomputed = np.zeros((len(meas), num_arm), dtype=float)
    for i in range(len(meas)):
        q_arm = motor_to_urdf_angles(M[i])
        q = q_from_arm(model, q_arm, num_arm=num_arm)
        tau = rnea_gravity(model, data_pin, q)[:num_arm]
        R_recomputed[i] = tau

    return R_recomputed, R_stored, T


def compare_rnea_against_measurement(model: pin.Model, measurement_data: dict, num_arm: int = 6):
    data_pin = model.createData()

    meas = list(iter_measurements(measurement_data))
    if not meas:
        print("\n[WARN] Keine Messungen in JSON")
        return

    try:
        R_recomputed, R_stored, T = _compute_rnea_for_measurements(model, measurement_data, num_arm=num_arm)
    except Exception as exc:
        print(f"\n[WARN] RNEA/Measurement Vergleich nicht möglich: {exc}")
        return

    if R_stored is None:
        print("\n[INFO] Mess-JSON enthält keine rnea_torque_nm Felder (nur Recompute verwendet).")
    else:
        max_diff = float(np.max(np.abs(R_recomputed - R_stored)))
        print("\nRNEA-Recompute Consistency Check:")
        print(f"  max |recomputed - stored| = {max_diff:.6g} Nm")
        if max_diff > 1e-9:
            print("  Hinweis: Das ist erwartbar, wenn die URDF nach der Aufnahme geändert wurde.")
            print("  Tipp: Nutze --rewrite-rnea um die JSON auf die aktuelle URDF zu aktualisieren.")

    # Additionally show how far measured torques are from URDF gravity model.
    E = T - R_recomputed
    rms = np.sqrt(np.mean(E**2, axis=0))
    mae = np.mean(np.abs(E), axis=0)

    print("\nMeasured vs. URDF(RNEA) error summary:")
    for j in range(num_arm):
        print(f"  J{j+1}: RMS={float(rms[j]):.4f} Nm  MAE={float(mae[j]):.4f} Nm")


def rewrite_measurement_rnea(
    measurement_data: dict,
    R_recomputed: np.ndarray,
    urdf_path: Path,
    num_arm: int = 6,
) -> dict:
    """Return a copy of measurement_data with rnea_* fields overwritten from R_recomputed."""

    out = dict(measurement_data)
    measurements = list(iter_measurements(measurement_data))
    if len(measurements) != R_recomputed.shape[0]:
        raise ValueError("Mismatch between measurement count and recomputed RNEA array")

    new_measurements = []
    for i, m in enumerate(measurements):
        m2 = dict(m)
        tau = R_recomputed[i, :num_arm]
        m2["rnea_torque_nm"] = tau.tolist()
        m2["rnea_torque_mnm"] = (tau * 1000.0).tolist()
        if "torque_nm" in m2:
            torque = np.asarray(m2["torque_nm"], dtype=float)[:num_arm]
            err = np.abs(torque - tau)
            m2["rnea_error_nm"] = err.tolist()
            m2["rnea_error_mnm"] = (err * 1000.0).tolist()
        new_measurements.append(m2)

    out["measurements"] = new_measurements

    meta = dict(out.get("metadata", {}))
    meta["rnea_recomputed_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["rnea_recomputed_urdf"] = str(urdf_path)
    out["metadata"] = meta
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="URDF/RNEA Debug Tool (Pinocchio)")
    parser.add_argument(
        "--urdf",
        type=str,
        default="gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf",
        help="Pfad zum URDF",
    )
    parser.add_argument(
        "--measurement",
        type=str,
        default=None,
        help="Optional: Mess-JSON (aus measure_holding_torques.py) für Pose/Achsen/RNEA-Check",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="home",
        help="Pose-Name in der Mess-JSON (default: home)",
    )
    parser.add_argument(
        "--sweep-step-deg",
        type=float,
        default=1.0,
        help="Step size für Sweeps [deg] (default: 1)",
    )
    parser.add_argument(
        "--rewrite-rnea",
        action="store_true",
        help="Wenn gesetzt: überschreibt rnea_* Felder in der Mess-JSON mit aktuellen RNEA-Werten aus der URDF.",
    )
    parser.add_argument(
        "--rewrite-output",
        type=str,
        default=None,
        help="Optional: Zielpfad für --rewrite-rnea (default: überschreibt die Eingabe-JSON).",
    )

    args = parser.parse_args()

    # Resolve paths robustly even if script is executed from a different CWD.
    # Repo root is the parent of the scripts/ directory.
    repo_root = Path(__file__).resolve().parent.parent

    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = (repo_root / urdf_path).resolve()
    model = build_model(urdf_path)

    print("=" * 70)
    print("URDF/RNEA DEBUG")
    print("=" * 70)
    print(f"URDF: {urdf_path}")
    print(f"nq={model.nq}, nv={model.nv}, njoints={model.njoints}")
    print(f"gravity.linear={model.gravity.linear}")

    num_arm = 6

    # Always print joint axes at q=0 for quick URDF sanity checking.
    data_pin0 = model.createData()
    q0 = np.zeros(model.nq)
    print_joint_axes_world(model, data_pin0, q0, num_arm=num_arm)

    if args.measurement:
        measurement_path = Path(args.measurement)
        if not measurement_path.is_absolute():
            measurement_path = (repo_root / measurement_path).resolve()
        mdata = load_measurement(measurement_path)

        pose = find_pose(mdata, args.pose)
        if not pose:
            available = [m.get("name") for m in iter_measurements(mdata)]
            print(f"\n[ERROR] Pose '{args.pose}' nicht gefunden. Verfügbar (Auszug): {available[:10]}")
            return 2

        motor = np.array(pose["position_rad"], dtype=float)
        q_arm = motor_to_urdf_angles(motor)
        q = q_from_arm(model, q_arm, num_arm=num_arm)

        print(f"\nPose from measurement: {args.pose}")
        print(f"motor position_rad: {motor}")
        print(f"urdf  q_arm(rad):   {q_arm}")

        data_pin = model.createData()
        print_joint_axes_world(model, data_pin, q, num_arm=num_arm)

        # Print comparison summary
        compare_rnea_against_measurement(model, mdata, num_arm=num_arm)

        # Optionally rewrite JSON with recomputed RNEA fields.
        if args.rewrite_rnea:
            try:
                R_recomputed, _, _ = _compute_rnea_for_measurements(model, mdata, num_arm=num_arm)
                updated = rewrite_measurement_rnea(mdata, R_recomputed, urdf_path=urdf_path, num_arm=num_arm)

                out_path = Path(args.rewrite_output) if args.rewrite_output else measurement_path
                if not out_path.is_absolute():
                    out_path = (repo_root / out_path).resolve()

                out_path.write_text(json.dumps(updated, indent=2))
                print(f"\n[✓] Mess-JSON aktualisiert (rnea_* überschrieben): {out_path}")
            except Exception as exc:
                print(f"\n[ERROR] Konnte Mess-JSON nicht aktualisieren: {exc}")

    # Always run sanity sweeps (URDF-only)
    print("\n" + "-" * 70)
    print("Sanity sweeps (URDF-only, other joints = 0)")
    print("-" * 70)

    max_abs, argmax = sweep_single_joint(
        model,
        num_arm=num_arm,
        joint_index0=1,  # q2
        deg_min=-180,
        deg_max=180,
        deg_step=args.sweep_step_deg,
    )
    print("\nSweep q2 only: max |gravity torque| per joint")
    for j in range(num_arm):
        print(f"  J{j+1}: max {max_abs[j]:.4f} Nm at q2={argmax[j]:.1f} deg")

    max_abs, argmax = sweep_single_joint(
        model,
        num_arm=num_arm,
        joint_index0=2,  # q3
        deg_min=-180,
        deg_max=180,
        deg_step=args.sweep_step_deg,
    )
    print("\nSweep q3 only: max |gravity torque| per joint")
    for j in range(num_arm):
        print(f"  J{j+1}: max {max_abs[j]:.4f} Nm at q3={argmax[j]:.1f} deg")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
