#!/usr/bin/env python3
"""Fit a lightweight MiniOne wrench compensation model from GELLO pose sweeps.

The intended input is a no-contact pose sweep recorded with scripts/bota_minione.py
using --use-gello.  The fitted model predicts the residual gravity/leakage wrench
from the runtime sensor-frame orientation and can be subtracted from the measured
base-frame wrench after software biasing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _default_output_yaml() -> Path:
    return Path(__file__).resolve().parents[1] / "gello" / "cr_dagger" / "config" / "bota_minione_wrench_calibration.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a linear BOTA MiniOne wrench compensation matrix from no-contact pose sweeps."
    )
    parser.add_argument("--calib", type=Path, required=True, help="Calibration NPZ from bota_minione.py.")
    parser.add_argument("--val", type=Path, default=None, help="Optional validation NPZ.")
    parser.add_argument(
        "--target-field",
        type=str,
        default="wrench_base_static_ema",
        help="NPZ wrench field to model. Usually wrench_base_static_ema.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["relative_rotation", "absolute_rotation"],
        default="relative_rotation",
        help=(
            "relative_rotation uses [1, vec(R - R_ref)] with R_ref from the first kept sample. "
            "This matches software-bias/tare operation."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization for matrix fit.")
    parser.add_argument("--trim-start", type=float, default=1.0, help="Seconds to drop at start of each NPZ.")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Seconds to drop at end of each NPZ.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth sample for fitting/evaluation.")
    parser.add_argument("--output-yaml", type=Path, default=_default_output_yaml())
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional NPZ containing coefficients and evaluation arrays.",
    )
    return parser.parse_args()


def _metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz:
        return {}
    obj = npz["metadata"][0]
    if isinstance(obj, dict):
        return obj
    return obj.item()


def _load_npz(path: Path, target_field: str, trim_start: float, trim_end: float, stride: int) -> dict[str, Any]:
    data = np.load(path.expanduser().resolve(), allow_pickle=True)
    for key in ("t_rel", "r_base_sensor", target_field):
        if key not in data:
            raise KeyError(f"{path} does not contain required field '{key}'")
    t = np.asarray(data["t_rel"], dtype=float)
    r = np.asarray(data["r_base_sensor"], dtype=float).reshape(-1, 3, 3)
    y = np.asarray(data[target_field], dtype=float).reshape(-1, 6)
    keep = np.isfinite(t)
    keep &= np.all(np.isfinite(r.reshape(r.shape[0], -1)), axis=1)
    keep &= np.all(np.isfinite(y), axis=1)
    if trim_start > 0.0:
        keep &= t >= (float(t[0]) + float(trim_start))
    if trim_end > 0.0:
        keep &= t <= (float(t[-1]) - float(trim_end))
    idx = np.flatnonzero(keep)
    if stride > 1:
        idx = idx[:: int(stride)]
    if idx.size < 20:
        raise ValueError(f"Too few usable samples in {path}: {idx.size}")
    return {
        "path": str(path.expanduser().resolve()),
        "metadata": _metadata(data),
        "t": t[idx],
        "r": r[idx],
        "y": y[idx],
        "idx": idx,
    }


def _features(r: np.ndarray, feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float).reshape(-1, 3, 3)
    r_ref = r[0].copy()
    if feature_mode == "relative_rotation":
        body = (r - r_ref).reshape(r.shape[0], 9)
    elif feature_mode == "absolute_rotation":
        body = r.reshape(r.shape[0], 9)
    else:  # pragma: no cover
        raise ValueError(feature_mode)
    phi = np.concatenate([np.ones((r.shape[0], 1), dtype=float), body], axis=1)
    return phi, r_ref


def _fit_ridge(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    y = np.asarray(y, dtype=float)
    reg = float(max(ridge, 0.0)) * np.eye(phi.shape[1], dtype=float)
    reg[0, 0] = 0.0  # do not regularize intercept
    return np.linalg.solve(phi.T @ phi + reg, phi.T @ y)


def _predict(phi: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return np.asarray(phi, dtype=float) @ np.asarray(coeff, dtype=float)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1, 6)
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    residual = y - pred

    def block(a: np.ndarray) -> dict[str, Any]:
        force_mag = np.linalg.norm(a[:, :3], axis=1)
        torque_mag = np.linalg.norm(a[:, 3:], axis=1)
        return {
            "mean": np.mean(a, axis=0).tolist(),
            "rms_axis": np.sqrt(np.mean(a * a, axis=0)).tolist(),
            "p95_abs_axis": np.percentile(np.abs(a), 95, axis=0).tolist(),
            "max_abs_axis": np.max(np.abs(a), axis=0).tolist(),
            "force_mag_mean": float(np.mean(force_mag)),
            "force_mag_rms": float(np.sqrt(np.mean(force_mag * force_mag))),
            "force_mag_p95": float(np.percentile(force_mag, 95)),
            "force_mag_max": float(np.max(force_mag)),
            "torque_mag_mean": float(np.mean(torque_mag)),
            "torque_mag_rms": float(np.sqrt(np.mean(torque_mag * torque_mag))),
            "torque_mag_p95": float(np.percentile(torque_mag, 95)),
            "torque_mag_max": float(np.max(torque_mag)),
        }

    return {
        "raw": block(y),
        "corrected": block(residual),
        "improvement": {
            "force_rms_ratio": float(block(residual)["force_mag_rms"] / max(block(y)["force_mag_rms"], 1e-12)),
            "torque_rms_ratio": float(block(residual)["torque_mag_rms"] / max(block(y)["torque_mag_rms"], 1e-12)),
        },
    }


def _print_metrics(name: str, metrics: dict[str, Any]) -> None:
    raw = metrics["raw"]
    cor = metrics["corrected"]
    imp = metrics["improvement"]
    print(f"\n{name}")
    print(
        "  |F| RMS raw -> corrected: "
        f"{raw['force_mag_rms']:.5f} N -> {cor['force_mag_rms']:.5f} N "
        f"(ratio {imp['force_rms_ratio']:.3f})"
    )
    print(
        "  |F| p95 raw -> corrected: "
        f"{raw['force_mag_p95']:.5f} N -> {cor['force_mag_p95']:.5f} N"
    )
    print(
        "  |T| RMS raw -> corrected: "
        f"{raw['torque_mag_rms']:.6f} Nm -> {cor['torque_mag_rms']:.6f} Nm "
        f"(ratio {imp['torque_rms_ratio']:.3f})"
    )
    print(
        "  |T| p95 raw -> corrected: "
        f"{raw['torque_mag_p95']:.6f} Nm -> {cor['torque_mag_p95']:.6f} Nm"
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = _parse_args()
    stride = max(int(args.stride), 1)
    calib = _load_npz(args.calib, args.target_field, args.trim_start, args.trim_end, stride)
    phi_calib, r_ref_calib = _features(calib["r"], args.feature_mode)
    coeff = _fit_ridge(phi_calib, calib["y"], args.ridge)
    pred_calib = _predict(phi_calib, coeff)
    calib_metrics = _metrics(calib["y"], pred_calib)

    val_payload = None
    pred_val = None
    val_metrics = None
    if args.val is not None:
        val_payload = _load_npz(args.val, args.target_field, args.trim_start, args.trim_end, stride)
        phi_val, r_ref_val = _features(val_payload["r"], args.feature_mode)
        pred_val = _predict(phi_val, coeff)
        val_metrics = _metrics(val_payload["y"], pred_val)
    else:
        r_ref_val = None

    _print_metrics("Calibration", calib_metrics)
    if val_metrics is not None:
        _print_metrics("Validation", val_metrics)

    out = {
        "kind": "bota_minione_linear_wrench_compensation",
        "version": 1,
        "target_field": str(args.target_field),
        "feature_mode": str(args.feature_mode),
        "feature_order": [
            "intercept",
            "R00_delta", "R01_delta", "R02_delta",
            "R10_delta", "R11_delta", "R12_delta",
            "R20_delta", "R21_delta", "R22_delta",
        ] if args.feature_mode == "relative_rotation" else [
            "intercept", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22",
        ],
        "usage": "predicted_wrench = phi(R_base_sensor, R_ref_runtime) @ coefficients; corrected = wrench_base_static_ema - predicted_wrench",
        "ridge": float(args.ridge),
        "trim_start_s": float(args.trim_start),
        "trim_end_s": float(args.trim_end),
        "stride": int(stride),
        "calibration_npz": calib["path"],
        "validation_npz": val_payload["path"] if val_payload is not None else "",
        "samples_calibration": int(calib["y"].shape[0]),
        "samples_validation": int(val_payload["y"].shape[0]) if val_payload is not None else 0,
        "r_ref_calibration": r_ref_calib.tolist(),
        "coefficients": coeff.tolist(),
        "metrics_calibration": calib_metrics,
        "metrics_validation": val_metrics if val_metrics is not None else {},
        "metadata_calibration": calib["metadata"],
        "metadata_validation": val_payload["metadata"] if val_payload is not None else {},
    }
    _write_yaml(args.output_yaml, out)
    print(f"\n[YAML] wrote {args.output_yaml.expanduser().resolve()}")

    if args.output_npz is not None:
        args.output_npz.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coefficients": coeff.astype(np.float64),
            "r_ref_calibration": r_ref_calib.astype(np.float64),
            "calib_y": calib["y"].astype(np.float32),
            "calib_pred": pred_calib.astype(np.float32),
            "calib_residual": (calib["y"] - pred_calib).astype(np.float32),
            "calib_t": calib["t"].astype(np.float64),
        }
        if val_payload is not None and pred_val is not None and r_ref_val is not None:
            payload.update(
                {
                    "val_y": val_payload["y"].astype(np.float32),
                    "val_pred": pred_val.astype(np.float32),
                    "val_residual": (val_payload["y"] - pred_val).astype(np.float32),
                    "val_t": val_payload["t"].astype(np.float64),
                    "r_ref_validation": r_ref_val.astype(np.float64),
                }
            )
        np.savez_compressed(args.output_npz.expanduser().resolve(), **payload)
        print(f"[NPZ] wrote {args.output_npz.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
