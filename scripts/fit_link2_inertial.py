#!/usr/bin/env python3
"""Fit a simple inertial correction for GELLO URDF using holding-torque recordings.

This script is intentionally simple and targeted at the current debugging workflow:

- Load a measurement JSON produced by scripts/measure_holding_torques.py
- Recompute gravity torques via Pinocchio RNEA for the provided URDF
- Fit a lightweight correction model for the dominant mismatch on J2

Model A (default)
-----------------
Add a point mass (fixed to link_2) at a candidate lever arm in the link_2 frame,
then fit the mass (and an optional constant torque bias) to best match J2.

Notes
-----
- This is a gravity-only fit (v=a=0). Rotational inertia terms do not affect
  gravity torques directly; the COM (lever) and mass do.
- The recorded JSON field name `current_ma` may actually be Dynamixel "current
  units" (e.g. XM430 uses 2.69mA/unit). The script works on `torque_nm`, so the
  labeling does not matter here.

Examples
--------
python scripts/fit_link2_inertial.py \
  --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf \
  --measurement scripts/recordings/holding_torques_20260103_112141.json \
  --allow-bias --nonnegative-mass

# Narrow grid (faster):
python scripts/fit_link2_inertial.py --measurement scripts/recordings/holding_torques_20260103_112141.json \
	--dx-min -0.35 --dx-max 0.35 --dx-step 0.02 \
	--dz-min -0.35 --dz-max 0.35 --dz-step 0.02
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
	import pinocchio as pin
except ImportError as exc:  # pragma: no cover
	raise SystemExit(
		"[ERROR] pinocchio ist nicht installiert. Installiere es oder nutze eine Umgebung mit pinocchio."
	) from exc


def motor_to_urdf_angles(motor_angles: np.ndarray) -> np.ndarray:
	"""Motor-Konvention -> URDF-Konvention (6-DOF arm).

	Consistent with scripts/measure_holding_torques.py.
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


def recompute_rnea_all(model: pin.Model, q_arm_all: np.ndarray) -> np.ndarray:
	data = model.createData()
	out = np.zeros((q_arm_all.shape[0], 6), dtype=float)
	for i in range(q_arm_all.shape[0]):
		q = q_from_arm(model, q_arm_all[i])
		out[i] = rnea_gravity(model, data, q)[:6]
	return out


@dataclass(frozen=True)
class FitResult:
	score: float
	rms_j2: float
	rms_j3: float
	m_kg: float
	bias_nm: float
	lever_m: np.ndarray


def fit_point_mass_grid(
	model_base: pin.Model,
	q_arm_all: np.ndarray,
	tau_meas: np.ndarray,
	joint_name: str,
	dx_vals: np.ndarray,
	dz_vals: np.ndarray,
	allow_bias: bool,
	nonnegative_mass: bool,
	w_j3: float,
	progress_every: int = 50,
) -> FitResult:
	"""Grid search lever=(dx,0,dz), fit mass m (and optional bias) to match J2."""

	jid = model_base.getJointId(joint_name)
	if jid <= 0:
		raise ValueError(f"Joint '{joint_name}' not found in model")

	tau_base = recompute_rnea_all(model_base, q_arm_all)

	best: Optional[FitResult] = None
	n_eval = 0
	total = int(dx_vals.size * dz_vals.size)

	for dx in dx_vals:
		for dz in dz_vals:
			lever = np.array([float(dx), 0.0, float(dz)], dtype=float)

			model = model_base.copy()
			# Add 1kg point-mass located at `lever` in the body frame.
			add = pin.Inertia(1.0, lever, np.zeros((3, 3), dtype=float))
			model.inertias[jid] = model.inertias[jid] + add

			tau_1kg = recompute_rnea_all(model, q_arm_all)
			delta = tau_1kg - tau_base

			# Fit on J2 (index 1)
			x = delta[:, 1]
			y = tau_meas[:, 1] - tau_base[:, 1]

			if allow_bias:
				A = np.column_stack([x, np.ones_like(x)])
				m_hat, b_hat = np.linalg.lstsq(A, y, rcond=None)[0]
			else:
				denom = float(np.dot(x, x))
				m_hat = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
				b_hat = 0.0

			if nonnegative_mass and m_hat < 0.0:
				m_hat = 0.0

			# Evaluate residuals (bias only applied for J2)
			pred_j2 = tau_base[:, 1] + m_hat * x + b_hat
			r_j2 = tau_meas[:, 1] - pred_j2
			rms_j2 = float(np.sqrt(np.mean(r_j2**2)))

			# Also see what happens to J3 if we apply the same mass correction
			pred_j3 = tau_base[:, 2] + m_hat * delta[:, 2]
			r_j3 = tau_meas[:, 2] - pred_j3
			rms_j3 = float(np.sqrt(np.mean(r_j3**2)))

			score = rms_j2 + w_j3 * rms_j3

			res = FitResult(
				score=score,
				rms_j2=rms_j2,
				rms_j3=rms_j3,
				m_kg=float(m_hat),
				bias_nm=float(b_hat),
				lever_m=lever,
			)

			if best is None or res.score < best.score:
				best = res

			n_eval += 1
			if progress_every > 0 and (n_eval % progress_every == 0 or n_eval == total):
				assert best is not None
				print(
					f"  grid {n_eval:>5}/{total} best score={best.score:.4f} "
					f"(rmsJ2={best.rms_j2:.4f}, m={best.m_kg:.3f}kg)"
				)

	assert best is not None
	return best


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Fit link_2 inertial correction (gravity) using measurement JSON"
	)
	parser.add_argument(
		"--urdf",
		type=str,
		default="gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf",
		help="Path to URDF",
	)
	parser.add_argument(
		"--measurement",
		type=str,
		required=True,
		help="Path to holding_torques_*.json",
	)
	parser.add_argument(
		"--joint",
		type=str,
		default="link_2_joint_2",
		help="Joint whose attached body inertia is modified (default: link_2_joint_2)",
	)
	parser.add_argument("--dx-min", type=float, default=-0.35)
	parser.add_argument("--dx-max", type=float, default=0.35)
	parser.add_argument("--dx-step", type=float, default=0.02)
	parser.add_argument("--dz-min", type=float, default=-0.35)
	parser.add_argument("--dz-max", type=float, default=0.35)
	parser.add_argument("--dz-step", type=float, default=0.02)
	parser.add_argument(
		"--allow-bias",
		action="store_true",
		help="Also fit a constant torque bias (Nm) for J2.",
	)
	parser.add_argument(
		"--nonnegative-mass",
		action="store_true",
		help="Clamp fitted mass to be >= 0.",
	)
	parser.add_argument(
		"--w-j3",
		type=float,
		default=0.25,
		help="Penalty weight for degrading J3 while fitting J2 (default: 0.25)",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=50,
		help="Progress print interval (grid evaluations)",
	)

	args = parser.parse_args()

	repo_root = Path(__file__).resolve().parent.parent

	urdf_path = Path(args.urdf)
	if not urdf_path.is_absolute():
		urdf_path = (repo_root / urdf_path).resolve()

	measurement_path = Path(args.measurement)
	if not measurement_path.is_absolute():
		measurement_path = (repo_root / measurement_path).resolve()

	data = json.loads(measurement_path.read_text())
	meas = data.get("measurements", [])
	if not meas:
		raise SystemExit("[ERROR] measurement JSON has no measurements")

	tau_meas = np.array([m["torque_nm"] for m in meas], dtype=float)

	if "position_urdf_rad" in meas[0]:
		q_arm_all = np.array([m["position_urdf_rad"] for m in meas], dtype=float)
	else:
		motor = np.array([m["position_rad"] for m in meas], dtype=float)
		q_arm_all = np.array([motor_to_urdf_angles(m) for m in motor], dtype=float)

	model_base = pin.buildModelFromUrdf(str(urdf_path))

	dx_vals = np.arange(args.dx_min, args.dx_max + 1e-12, args.dx_step)
	dz_vals = np.arange(args.dz_min, args.dz_max + 1e-12, args.dz_step)

	print("=" * 70)
	print("FIT link_2 INERTIAL (gravity-only)")
	print("=" * 70)
	print("URDF:", urdf_path)
	print("Measurement:", measurement_path)
	print(
		f"Grid: dx in [{dx_vals[0]:+.3f},{dx_vals[-1]:+.3f}] step {args.dx_step:.3f} ({dx_vals.size})"
	)
	print(
		f"      dz in [{dz_vals[0]:+.3f},{dz_vals[-1]:+.3f}] step {args.dz_step:.3f} ({dz_vals.size})"
	)
	print(
		f"Fitting: point-mass on '{args.joint}' body, target J2 with w_j3={args.w_j3}"
	)

	# baseline errors
	tau_base = recompute_rnea_all(model_base, q_arm_all)
	rms0_j2 = float(np.sqrt(np.mean((tau_meas[:, 1] - tau_base[:, 1]) ** 2)))
	rms0_j3 = float(np.sqrt(np.mean((tau_meas[:, 2] - tau_base[:, 2]) ** 2)))
	print(f"Baseline RMS: J2={rms0_j2:.4f} Nm, J3={rms0_j3:.4f} Nm")

	best = fit_point_mass_grid(
		model_base=model_base,
		q_arm_all=q_arm_all,
		tau_meas=tau_meas,
		joint_name=args.joint,
		dx_vals=dx_vals,
		dz_vals=dz_vals,
		allow_bias=bool(args.allow_bias),
		nonnegative_mass=bool(args.nonnegative_mass),
		w_j3=float(args.w_j3),
		progress_every=int(args.progress_every),
	)

	print("\n" + "-" * 70)
	print("Best point-mass fit")
	print("-" * 70)
	print(
		f"lever (link_2 frame) [m]: dx={best.lever_m[0]:+.4f}, dy={best.lever_m[1]:+.4f}, dz={best.lever_m[2]:+.4f}"
	)
	print(f"mass m [kg]: {best.m_kg:.4f}")
	if args.allow_bias:
		print(f"J2 torque bias b [Nm]: {best.bias_nm:+.4f}")
	print(
		f"RMS after fit: J2={best.rms_j2:.4f} Nm (baseline {rms0_j2:.4f}), "
		f"J3={best.rms_j3:.4f} Nm (baseline {rms0_j3:.4f})"
	)

	print("\nSuggested URDF snippet (fixed extra mass on link_2):")
	print("  <link name=\"link_2_extra_mass\">")
	print("    <inertial>")
	print(
		f"      <origin xyz=\"{best.lever_m[0]:.6g} {best.lever_m[1]:.6g} {best.lever_m[2]:.6g}\" rpy=\"0 0 0\"/>"
	)
	print(f"      <mass value=\"{best.m_kg:.6g}\"/>")
	print("      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>")
	print("    </inertial>")
	print("  </link>")
	print("  <joint name=\"link_2_extra_mass_joint\" type=\"fixed\">")
	print("    <parent link=\"link_2\"/>")
	print("    <child link=\"link_2_extra_mass\"/>")
	print("    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>")
	print("  </joint>")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

