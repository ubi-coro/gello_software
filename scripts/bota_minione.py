#!/usr/bin/env python3
"""Manual BotaSys MiniOne readout smoke run.

This script intentionally avoids pytest naming conventions. It is meant for
bench bring-up of a USB-connected MiniOne without IMU usage before the sensor is
mounted on GELLO.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - production env includes PyYAML
    yaml = None

_RUNNING = True


@dataclass(frozen=True)
class WrenchSample:
    host_t: float
    sensor_timestamp_us: int
    force: tuple[float, float, float]
    torque: tuple[float, float, float]
    temperature_c: float
    status_val: int
    throttled: bool
    overrange: bool
    invalid: bool
    raw: bool


def _handle_signal(signum: int, frame: object) -> None:
    del signum, frame
    global _RUNNING
    _RUNNING = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "bota_binary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual MiniOne wrench readout and live plotting utility."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to a bota-driver JSON config file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Run duration in seconds. Use <=0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=10.0,
        help="Time span shown in the live plot.",
    )
    parser.add_argument(
        "--nominal-rate",
        type=float,
        default=500.0,
        help="Only used to size local plotting/rate buffers.",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=2.0,
        help="Console status print frequency in Hz.",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=20.0,
        help="Maximum live plot refresh rate in Hz.",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=500,
        help="Number of recent inter-arrival samples used for rate statistics.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Use non-blocking read_frame() instead of read_frame_blocking().",
    )
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=500.0,
        help="Loop rate for --poll mode in Hz.",
    )
    parser.add_argument(
        "--no-tare",
        action="store_true",
        help="Skip tare in INACTIVE state before activation.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib live plot. Best for clean frequency readings.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help=(
            "EMA fixed alpha in (0, 1]. Set 0 to use --ema-cutoff. "
            "Phase-B default is 0.0."
        ),
    )
    parser.add_argument(
        "--ema-cutoff",
        type=float,
        default=25.0,
        help=(
            "EMA cutoff frequency in Hz. Phase-B default is 25.0. "
            "Set 0 together with --ema-alpha 0 to disable filtering."
        ),
    )
    parser.add_argument(
        "--ema-print-only",
        action="store_true",
        help="Print EMA values but keep the live plot on raw wrench values.",
    )
    parser.add_argument(
        "--use-gello",
        action="store_true",
        help="Read GELLO Dynamixel encoders and compute the MiniOne frame pose from the GELLO URDF.",
    )
    parser.add_argument(
        "--gello-config",
        type=Path,
        default=_repo_root() / "configs" / "ur5e_gello_factr_hw_V3_PhaseA.yaml",
        help="GELLO config used for encoder calibration and URDF resolution.",
    )
    parser.add_argument(
        "--sensor-urdf",
        type=Path,
        default=None,
        help=(
            "Optional URDF used only for GELLO-based MiniOne sensor-frame compensation. "
            "Encoder calibration still comes from --gello-config."
        ),
    )
    parser.add_argument(
        "--bota-frame",
        type=str,
        default="link_6_handle_minione",
        help="Pinocchio frame name for the MiniOne sensor. Empty uses the last URDF frame.",
    )
    parser.add_argument(
        "--bias-seconds",
        type=float,
        default=2.0,
        help="Software-bias duration after activation. Hold GELLO still in the start pose.",
    )
    parser.add_argument(
        "--compensation-mode",
        choices=["raw", "static", "dynamic", "dynamic_abs", "calibrated"],
        default="static",
        help="Signal shown in console/plot. NPZ logs all available variants.",
    )
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=0.070,
        help="Distal payload mass seen by the MiniOne [kg]. Used for dynamic gravity compensation.",
    )
    parser.add_argument(
        "--payload-com",
        type=str,
        default="0,0,0.0146",
        help="Payload COM in MiniOne sensor frame [m], comma separated.",
    )
    parser.add_argument(
        "--gravity-sign",
        type=float,
        default=1.0,
        help="Gravity compensation sign. Use -1 if modeled compensation has the wrong sign.",
    )
    parser.add_argument(
        "--wrench-calibration",
        type=Path,
        default=_repo_root() / "gello" / "cr_dagger" / "config" / "bota_minione_wrench_calibration.yaml",
        help="YAML calibration matrix used by --compensation-mode calibrated.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If set, save all samples and compensation variants as an .npz in this directory.",
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="bota_minione",
        help="Filename prefix for --log-dir NPZ output.",
    )
    parser.add_argument(
        "--deadband",
        type=str,
        default="0.25,0.25,0.25,0.01,0.01,0.01",
        help="Phase-B-style wrench deadband [Fx,Fy,Fz,Tx,Ty,Tz].",
    )
    parser.add_argument(
        "--saturation",
        type=str,
        default="25,25,25,1.5,1.5,1.5",
        help="Phase-B-style wrench saturation [Fx,Fy,Fz,Tx,Ty,Tz].",
    )
    parser.add_argument(
        "--base-axis-map",
        type=str,
        default="2,1,0,5,4,3",
        help="Phase-B BOTA base-axis map, comma separated permutation of 0..5.",
    )
    parser.add_argument(
        "--base-axis-signs",
        type=str,
        default="-1,1,1,-1,1,1",
        help="Phase-B BOTA base-axis signs, comma separated.",
    )
    return parser.parse_args()


def _frame_to_sample(frame: object, host_t: float) -> WrenchSample:
    status = frame.status
    return WrenchSample(
        host_t=host_t,
        sensor_timestamp_us=int(frame.timestamp),
        force=tuple(float(v) for v in frame.force[:3]),
        torque=tuple(float(v) for v in frame.torque[:3]),
        temperature_c=float(frame.temperature),
        status_val=int(getattr(status, "val", 0)),
        throttled=bool(status.throttled),
        overrange=bool(status.overrange),
        invalid=bool(status.invalid),
        raw=bool(status.raw),
    )


def _frequency_hz(intervals_s: Iterable[float]) -> float:
    values = [v for v in intervals_s if v > 0.0]
    if not values:
        return 0.0
    return len(values) / sum(values)


def _parse_vec(text: str, n: int, default: float = 0.0) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    values = [float(p) for p in parts]
    if not values:
        values = [float(default)] * n
    if len(values) < n:
        values.extend([values[-1]] * (n - len(values)))
    return np.asarray(values[:n], dtype=float)


def _wrench_array(sample: WrenchSample) -> np.ndarray:
    return np.asarray(tuple(sample.force) + tuple(sample.torque), dtype=float)


def _sample_from_wrench(template: WrenchSample, wrench: np.ndarray) -> WrenchSample:
    arr = np.asarray(wrench, dtype=float).reshape(6)
    return WrenchSample(
        host_t=template.host_t,
        sensor_timestamp_us=template.sensor_timestamp_us,
        force=tuple(float(v) for v in arr[:3]),
        torque=tuple(float(v) for v in arr[3:]),
        temperature_c=template.temperature_c,
        status_val=template.status_val,
        throttled=template.throttled,
        overrange=template.overrange,
        invalid=template.invalid,
        raw=template.raw,
    )


def _payload_gravity_wrench_base(
    r_base_sensor: np.ndarray,
    payload_mass_kg: float,
    payload_com_sensor: np.ndarray,
) -> np.ndarray:
    if payload_mass_kg <= 0.0:
        return np.zeros(6, dtype=float)
    g_base = np.array([0.0, 0.0, -9.80665], dtype=float)
    force_g_base = float(payload_mass_kg) * g_base
    r_com_base = np.asarray(r_base_sensor, dtype=float).reshape(3, 3) @ payload_com_sensor.reshape(3)
    torque_g_base = np.cross(r_com_base, force_g_base)
    return np.concatenate([force_g_base, torque_g_base])


def _transform_sensor_wrench_to_base(wrench_sensor: np.ndarray, r_base_sensor: np.ndarray) -> np.ndarray:
    wrench = np.asarray(wrench_sensor, dtype=float).reshape(6)
    rot = np.asarray(r_base_sensor, dtype=float).reshape(3, 3)
    return np.concatenate([rot @ wrench[:3], rot @ wrench[3:]])


def _parse_axis_map(text: str) -> np.ndarray:
    arr = np.asarray([int(float(v)) for v in str(text).split(",") if v.strip()], dtype=int)
    if arr.size != 6 or sorted(arr.tolist()) != list(range(6)):
        raise ValueError("--base-axis-map must be a permutation of 0,1,2,3,4,5")
    return arr


def _parse_signs(text: str) -> np.ndarray:
    arr = _parse_vec(text, 6, 1.0)
    signs = np.sign(arr)
    signs[signs == 0.0] = 1.0
    return signs.astype(float)


def _apply_axis_correction(wrench_base: np.ndarray, axis_map: np.ndarray, axis_signs: np.ndarray) -> np.ndarray:
    wrench = np.asarray(wrench_base, dtype=float).reshape(6)
    return np.asarray(axis_signs, dtype=float).reshape(6) * wrench[np.asarray(axis_map, dtype=int).reshape(6)]


def _axis_corrected_labels(axis_map: np.ndarray, axis_signs: np.ndarray) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    source_labels = ("Fx_base", "Fy_base", "Fz_base", "Tx_base", "Ty_base", "Tz_base")
    mapped = np.asarray(axis_map, dtype=int).reshape(6)
    signs = np.asarray(axis_signs, dtype=float).reshape(6)
    labels = []
    for src_idx, sign in zip(mapped, signs):
        prefix = "-" if sign < 0 else ""
        labels.append(f"{prefix}{source_labels[int(src_idx)]}")
    return tuple(labels[:3]), tuple(labels[3:])


def _condition_wrench(wrench: np.ndarray, deadband: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    arr = np.asarray(wrench, dtype=float).reshape(6)
    db = np.asarray(deadband, dtype=float).reshape(6)
    sat = np.asarray(saturation, dtype=float).reshape(6)
    conditioned = np.where(np.abs(arr) > db, arr - np.sign(arr) * db, 0.0)
    return np.clip(conditioned, -sat, sat)


class WrenchEmaFilter:
    def __init__(self, cutoff_hz: float, alpha: float) -> None:
        self.cutoff_hz = max(float(cutoff_hz), 0.0)
        self.alpha = float(np.clip(float(alpha), 0.0, 1.0))
        self.value: np.ndarray | None = None
        self.last_sensor_ts: int | None = None

    def _alpha(self, dt_s: float) -> float:
        if self.cutoff_hz > 0.0:
            return _ema_alpha_from_cutoff(self.cutoff_hz, dt_s)
        if self.alpha > 0.0:
            return self.alpha
        return 1.0

    def update(self, wrench: np.ndarray, sensor_timestamp_us: int) -> np.ndarray:
        arr = np.asarray(wrench, dtype=float).reshape(6)
        ts = int(sensor_timestamp_us)
        if self.value is None:
            self.value = arr.copy()
            self.last_sensor_ts = ts
            return self.value.copy()
        if self.last_sensor_ts == ts:
            return self.value.copy()
        dt_s = 0.0 if self.last_sensor_ts is None else max(0.0, (ts - self.last_sensor_ts) * 1e-6)
        a = self._alpha(dt_s)
        self.value = self.value + a * (arr - self.value)
        self.last_sensor_ts = ts
        return self.value.copy()


class CompensationFilters:
    def __init__(self, cutoff_hz: float, alpha: float) -> None:
        self.static = WrenchEmaFilter(cutoff_hz, alpha)
        self.dynamic = WrenchEmaFilter(cutoff_hz, alpha)
        self.dynamic_abs = WrenchEmaFilter(cutoff_hz, alpha)


class WrenchCalibration:
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.feature_mode = str(payload.get("feature_mode", "relative_rotation"))
        self.target_field = str(payload.get("target_field", "wrench_base_static_ema"))
        self.coefficients = np.asarray(payload["coefficients"], dtype=float)
        if self.coefficients.shape != (10, 6):
            raise ValueError(
                f"Expected calibration coefficients shape (10, 6), got {self.coefficients.shape}"
            )

    def features(self, r_base_sensor: np.ndarray, r_ref_runtime: np.ndarray) -> np.ndarray:
        rot = np.asarray(r_base_sensor, dtype=float).reshape(3, 3)
        if self.feature_mode == "relative_rotation":
            body = (rot - np.asarray(r_ref_runtime, dtype=float).reshape(3, 3)).reshape(9)
        elif self.feature_mode == "absolute_rotation":
            body = rot.reshape(9)
        else:
            raise ValueError(f"Unsupported calibration feature_mode: {self.feature_mode}")
        return np.concatenate([[1.0], body])

    def predict(self, r_base_sensor: np.ndarray, r_ref_runtime: np.ndarray) -> np.ndarray:
        return self.features(r_base_sensor, r_ref_runtime) @ self.coefficients


def _load_wrench_calibration(path: Path) -> WrenchCalibration:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --wrench-calibration")
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"--wrench-calibration does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid wrench calibration YAML: {resolved}")
    if payload.get("kind") != "bota_minione_linear_wrench_compensation":
        raise ValueError(f"Unsupported wrench calibration kind: {payload.get('kind')!r}")
    calib = WrenchCalibration(resolved, payload)
    print(
        f"[CALIB] loaded wrench calibration: {resolved} "
        f"feature_mode={calib.feature_mode} target={calib.target_field}"
    )
    return calib


class GelloEncoderContext:
    def __init__(self, system: Any, kin: Any, temp_config_path: Path | None):
        self.system = system
        self.kin = kin
        self.temp_config_path = temp_config_path

    def close(self) -> None:
        try:
            self.system.shutdown()
        except Exception as exc:
            print(f"WARNING: GELLO shutdown failed: {exc}", file=sys.stderr)
        if self.temp_config_path is not None:
            try:
                self.temp_config_path.unlink(missing_ok=True)
            except Exception:
                pass


def _make_encoder_only_config(config_path: Path, sensor_urdf: Path | None = None) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --use-gello")
    cfg = yaml.safe_load(config_path.read_text())
    cfg.setdefault("teleop", {})["enable"] = False
    cfg.setdefault("controller", {}).setdefault("gravity_comp", {})["enable"] = False
    cfg.setdefault("logging", {})["enable"] = False
    # The test uses GELLO as an encoder; fake fallback would make the logged pose meaningless.
    cfg.setdefault("dynamixel", {})["use_fake_fallback"] = False
    if sensor_urdf is not None:
        sensor_urdf_path = sensor_urdf.expanduser().resolve()
        if not sensor_urdf_path.exists():
            raise FileNotFoundError(f"--sensor-urdf does not exist: {sensor_urdf_path}")
        cfg.setdefault("arm_teleop", {})["leader_urdf"] = str(sensor_urdf_path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_bota_minione_gello_encoder.yaml",
        prefix="gello_",
        delete=False,
    )
    try:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


def _open_gello_encoder(
    config_path: Path,
    frame_name: str,
    sensor_urdf: Path | None = None,
) -> GelloEncoderContext:
    from gello.factr.gravity_compensation import FACTRGravityCompensation
    from gello.cr_dagger.core.bota_se3_residual import GelloTaskspaceKinematics

    temp_cfg = _make_encoder_only_config(config_path.expanduser().resolve(), sensor_urdf)
    system = FACTRGravityCompensation(str(temp_cfg), enable_visualization=False)
    if bool(getattr(system.driver, "_is_fake", False)):
        system.shutdown()
        temp_cfg.unlink(missing_ok=True)
        raise RuntimeError("Refusing --use-gello with FakeDynamixelDriver")
    kin = GelloTaskspaceKinematics(
        system,
        int(system.num_arm_joints),
        str(frame_name or "") or None,
    )
    if sensor_urdf is not None:
        print(f"[GELLO] sensor compensation URDF: {sensor_urdf.expanduser().resolve()}")
    print(f"[GELLO] encoder mode active, BOTA frame='{kin.frame_name}'")
    return GelloEncoderContext(system=system, kin=kin, temp_config_path=temp_cfg)


def _read_gello_pose(ctx: GelloEncoderContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    system = ctx.system
    n = int(system.num_arm_joints)
    pos, vel, _cur = system.driver.get_positions_velocities_and_currents()
    q_raw = np.asarray(pos[:n], dtype=float)
    dq_raw = np.asarray(vel[:n], dtype=float)
    q = (q_raw - system.joint_offsets[:n]) * system.joint_signs[:n]
    dq = dq_raw * system.joint_signs[:n]
    pos_base_sensor, r_base_sensor, _ = ctx.kin.pose_and_jacobian(q)
    return q, dq, pos_base_sensor, r_base_sensor


class NpzLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        if not self.rows:
            print("[NPZ] no rows recorded; skipping save")
            return
        keys = sorted(self.rows[0].keys())
        arrays: dict[str, Any] = {}
        for key in keys:
            arrays[key] = np.asarray([row[key] for row in self.rows])
        arrays["metadata"] = np.asarray([metadata], dtype=object)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        print(f"[NPZ] saved {len(self.rows)} samples: {path}")


def _select_display_wrench(
    mode: str,
    raw: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
    dynamic_abs: np.ndarray,
    calibrated: np.ndarray,
) -> np.ndarray:
    if mode == "raw":
        return raw
    if mode == "dynamic":
        return dynamic
    if mode == "dynamic_abs":
        return dynamic_abs
    if mode == "calibrated":
        return calibrated
    return static


def _format_log_path(log_dir: Path, prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(prefix)).strip("_")
    if not safe_prefix:
        safe_prefix = "bota_minione"
    return log_dir.expanduser().resolve() / f"{safe_prefix}_{stamp}.npz"


def _format_vec(values: tuple[float, float, float], unit: str) -> str:
    return f"[{values[0]: .4f}, {values[1]: .4f}, {values[2]: .4f}] {unit}"


def _print_sample(sample: WrenchSample, avg_hz: float, expected_hz: float | None) -> None:
    expected = f", expected={expected_hz:.2f} Hz" if expected_hz else ""
    print(
        " | ".join(
            [
                f"rate={avg_hz:.2f} Hz{expected}",
                f"F={_format_vec(sample.force, 'N')}",
                f"T={_format_vec(sample.torque, 'Nm')}",
                f"temp={sample.temperature_c:.2f} C",
                (
                    "status="
                    f"val={sample.status_val} "
                    f"thr={sample.throttled} over={sample.overrange} "
                    f"invalid={sample.invalid} raw={sample.raw}"
                ),
                f"sensor_ts={sample.sensor_timestamp_us} us",
            ]
        ),
        flush=True,
    )


def _print_filtered_sample(
    raw_sample: WrenchSample,
    filtered_sample: WrenchSample,
    avg_hz: float,
    expected_hz: float | None,
) -> None:
    expected = f", expected={expected_hz:.2f} Hz" if expected_hz else ""
    print(
        " | ".join(
            [
                f"rate={avg_hz:.2f} Hz{expected}",
                f"F={_format_vec(raw_sample.force, 'N')}",
                f"T={_format_vec(raw_sample.torque, 'Nm')}",
                f"F_ema={_format_vec(filtered_sample.force, 'N')}",
                f"T_ema={_format_vec(filtered_sample.torque, 'Nm')}",
                f"temp={raw_sample.temperature_c:.2f} C",
                (
                    "status="
                    f"val={raw_sample.status_val} "
                    f"thr={raw_sample.throttled} over={raw_sample.overrange} "
                    f"invalid={raw_sample.invalid} raw={raw_sample.raw}"
                ),
                f"sensor_ts={raw_sample.sensor_timestamp_us} us",
            ]
        ),
        flush=True,
    )


def _ema_alpha_from_cutoff(cutoff_hz: float, dt_s: float) -> float:
    if cutoff_hz <= 0.0 or dt_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)


def _filtered_sample(
    sample: WrenchSample,
    previous: WrenchSample | None,
    alpha: float,
) -> WrenchSample:
    if previous is None:
        return sample

    a = min(max(float(alpha), 0.0), 1.0)
    force = tuple(
        previous.force[idx] + a * (sample.force[idx] - previous.force[idx])
        for idx in range(3)
    )
    torque = tuple(
        previous.torque[idx] + a * (sample.torque[idx] - previous.torque[idx])
        for idx in range(3)
    )
    return WrenchSample(
        host_t=sample.host_t,
        sensor_timestamp_us=sample.sensor_timestamp_us,
        force=force,
        torque=torque,
        temperature_c=sample.temperature_c,
        status_val=sample.status_val,
        throttled=sample.throttled,
        overrange=sample.overrange,
        invalid=sample.invalid,
        raw=sample.raw,
    )


class LiveWrenchPlot:
    def __init__(
        self,
        buffer_seconds: float,
        force_labels: tuple[str, str, str] = ("Fx", "Fy", "Fz"),
        torque_labels: tuple[str, str, str] = ("Tx", "Ty", "Tz"),
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._buffer_seconds = float(buffer_seconds)
        self._fig, (self._force_ax, self._torque_ax) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7)
        )
        self._fig.canvas.manager.set_window_title("BotaSys MiniOne wrench")
        self._force_lines = tuple(
            self._force_ax.plot([], [], label=label)[0] for label in force_labels
        )
        self._torque_lines = tuple(
            self._torque_ax.plot([], [], label=label)[0] for label in torque_labels
        )
        self._force_ax.set_ylabel("Force [N]")
        self._torque_ax.set_ylabel("Torque [Nm]")
        self._torque_ax.set_xlabel("Time [s]")
        self._force_ax.grid(True)
        self._torque_ax.grid(True)
        self._force_ax.legend(loc="upper right")
        self._torque_ax.legend(loc="upper right")
        self._plt.ion()
        self._plt.show(block=False)

    def update(self, samples: Deque[WrenchSample], start_t: float) -> None:
        if not samples:
            return

        t = [sample.host_t - start_t for sample in samples]
        force = list(zip(*(sample.force for sample in samples)))
        torque = list(zip(*(sample.torque for sample in samples)))

        for idx, line in enumerate(self._force_lines):
            line.set_data(t, force[idx])
        for idx, line in enumerate(self._torque_lines):
            line.set_data(t, torque[idx])

        x_max = max(t[-1], self._buffer_seconds)
        x_min = max(0.0, x_max - self._buffer_seconds)
        self._force_ax.set_xlim(x_min, x_max)
        self._torque_ax.set_xlim(x_min, x_max)
        self._force_ax.relim()
        self._force_ax.autoscale_view(scalex=False, scaley=True)
        self._torque_ax.relim()
        self._torque_ax.autoscale_view(scalex=False, scaley=True)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)


def _expected_rate(driver: object) -> float | None:
    try:
        timestep = driver.get_expected_timestep()
    except Exception:
        return None
    seconds = float(timestep.total_seconds())
    if seconds <= 0.0:
        return None
    return 1.0 / seconds


def _load_bota_driver() -> object:
    try:
        import bota_driver
    except ImportError as exc:  # pragma: no cover - manual hardware utility
        raise RuntimeError(
            "Could not import bota_driver. Install/activate the environment that "
            "contains the Bota Systems Python driver."
        ) from exc
    return bota_driver


def _open_driver(config_path: Path, tare: bool) -> object:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass --config /path/to/bota.json."
        )

    bota_driver = _load_bota_driver()
    driver = bota_driver.BotaDriver(str(config_path))
    print(f"BotaDriver version: {driver.get_driver_version_string()}")
    print(f"Initial state: {driver.get_driver_state()}")

    if not driver.configure():
        raise RuntimeError("Failed to configure Bota driver")
    print(f"Configured state: {driver.get_driver_state()}")

    if tare:
        print("Taring sensor in INACTIVE state...")
        if not driver.tare():
            raise RuntimeError("Failed to tare MiniOne")

    if not driver.activate():
        raise RuntimeError("Failed to activate Bota driver")
    print(f"Active state: {driver.get_driver_state()}")
    return driver


def _state_name(state: object) -> str:
    return str(getattr(state, "name", state))


def _close_driver(driver: object | None) -> None:
    if driver is None:
        return

    try:
        state = _state_name(driver.get_driver_state())
        if state.endswith("ACTIVE") and not state.endswith("INACTIVE"):
            if not driver.deactivate():
                print("WARNING: Bota driver deactivate() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver deactivate failed: {exc}", file=sys.stderr)

    try:
        state = _state_name(driver.get_driver_state())
        if state.endswith("INACTIVE"):
            if not driver.cleanup():
                print("WARNING: Bota driver cleanup() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver cleanup failed: {exc}", file=sys.stderr)

    try:
        if not driver.shutdown():
            print("WARNING: Bota driver shutdown() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver shutdown failed: {exc}", file=sys.stderr)


def _read_once(driver: object, poll: bool) -> object:
    if poll:
        return driver.read_frame()
    return driver.read_frame_blocking()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    max_samples = max(10, int(args.buffer_seconds * args.nominal_rate))
    samples: Deque[WrenchSample] = deque(maxlen=max_samples)
    plot_samples: Deque[WrenchSample] = deque(maxlen=max_samples)
    intervals_s: Deque[float] = deque(maxlen=max(2, int(args.rate_window)))
    ema_enabled = args.ema_cutoff > 0.0 or args.ema_alpha > 0.0
    filtered_sample: WrenchSample | None = None
    last_filter_sensor_ts: int | None = None

    payload_com = _parse_vec(args.payload_com, 3, 0.0)
    payload_mass = max(float(args.payload_mass), 0.0)
    gravity_sign = float(args.gravity_sign)
    deadband = _parse_vec(args.deadband, 6, 0.0)
    saturation = _parse_vec(args.saturation, 6, 25.0)
    base_axis_map = _parse_axis_map(args.base_axis_map)
    base_axis_signs = _parse_signs(args.base_axis_signs)
    comp_filters = CompensationFilters(args.ema_cutoff, args.ema_alpha)
    wrench_calibration = None
    if str(args.compensation_mode) == "calibrated":
        wrench_calibration = _load_wrench_calibration(args.wrench_calibration)

    if str(args.compensation_mode) == "raw":
        force_labels = ("Fx_sensor", "Fy_sensor", "Fz_sensor")
        torque_labels = ("Tx_sensor", "Ty_sensor", "Tz_sensor")
    else:
        force_labels, torque_labels = _axis_corrected_labels(base_axis_map, base_axis_signs)

    plot = None
    if not args.no_plot:
        try:
            plot = LiveWrenchPlot(args.buffer_seconds, force_labels, torque_labels)
            print(f"[PLOT] force labels={list(force_labels)} torque labels={list(torque_labels)}")
        except Exception as exc:
            print(f"Live plot disabled: {exc}", file=sys.stderr)

    driver = None
    gello_ctx: GelloEncoderContext | None = None
    npz_logger = NpzLogger() if args.log_dir is not None else None
    npz_path = _format_log_path(args.log_dir, args.log_prefix) if args.log_dir is not None else None

    start_t = time.perf_counter()
    last_read_t: float | None = None
    last_print_t = start_t
    last_plot_t = start_t
    next_poll_t = start_t
    expected_hz: float | None = None
    bias_sensor = np.zeros(6, dtype=float)
    model_gravity_initial_base = np.zeros(6, dtype=float)
    calibration_r_ref_runtime = np.eye(3, dtype=float)
    bias_samples = 0

    exit_code = 0
    try:
        if args.use_gello:
            gello_ctx = _open_gello_encoder(args.gello_config, args.bota_frame, args.sensor_urdf)

        driver = _open_driver(args.config.expanduser().resolve(), tare=not args.no_tare)
        expected_hz = _expected_rate(driver)
        if expected_hz:
            print(f"Driver expected timestep: {1000.0 / expected_hz:.3f} ms ({expected_hz:.2f} Hz)")

        bias_s = max(float(args.bias_seconds), 0.0)
        if bias_s > 0.0:
            print(f"[BIAS] collecting software bias for {bias_s:.2f}s; keep GELLO still")
            raw_bias: list[np.ndarray] = []
            model_bias: list[np.ndarray] = []
            t_bias = time.perf_counter()
            while _RUNNING and time.perf_counter() - t_bias < bias_s:
                frame = _read_once(driver, args.poll)
                sample = _frame_to_sample(frame, time.perf_counter())
                raw_bias.append(_wrench_array(sample))
                if gello_ctx is not None:
                    _q, _dq, _pos_sensor, rot_sensor = _read_gello_pose(gello_ctx)
                    model_bias.append(
                        _payload_gravity_wrench_base(rot_sensor, payload_mass, payload_com)
                    )
                if args.poll:
                    time.sleep(1.0 / max(args.poll_rate, 1e-6))
            if raw_bias:
                bias_sensor = np.mean(np.asarray(raw_bias, dtype=float), axis=0)
                bias_samples = len(raw_bias)
            if model_bias:
                model_gravity_initial_base = np.mean(np.asarray(model_bias, dtype=float), axis=0)
            print(f"[BIAS] sensor bias: {np.array2string(bias_sensor, precision=5)}")
            if gello_ctx is not None:
                print(
                    "[BIAS] initial modeled payload gravity(base): "
                    f"{np.array2string(model_gravity_initial_base, precision=5)}"
                )

        if wrench_calibration is not None:
            if gello_ctx is None:
                raise RuntimeError("--compensation-mode calibrated requires --use-gello")
            _q_ref, _dq_ref, _pos_ref, calibration_r_ref_runtime = _read_gello_pose(gello_ctx)
            print(
                "[CALIB] runtime reference R_base_sensor set after bias; "
                f"target_field={wrench_calibration.target_field}"
            )

        # Reset timing after bias so logs/plots start at the actual experiment.
        start_t = time.perf_counter()
        last_read_t = None
        last_print_t = start_t
        last_plot_t = start_t
        next_poll_t = start_t

        while _RUNNING:
            now = time.perf_counter()
            if args.duration > 0.0 and now - start_t >= args.duration:
                break

            if args.poll:
                next_poll_t += 1.0 / max(args.poll_rate, 1e-6)

            frame = _read_once(driver, args.poll)
            read_t = time.perf_counter()
            sample_raw = _frame_to_sample(frame, read_t)
            samples.append(sample_raw)

            if last_read_t is not None:
                intervals_s.append(read_t - last_read_t)
            dt_s = 0.0 if last_read_t is None else read_t - last_read_t
            last_read_t = read_t

            wrench_sensor_raw = _wrench_array(sample_raw)
            wrench_sensor_bias = wrench_sensor_raw - bias_sensor
            q_leader = np.full(6, np.nan, dtype=float)
            dq_leader = np.full(6, np.nan, dtype=float)
            sensor_pos_base = np.full(3, np.nan, dtype=float)
            r_base_sensor = np.eye(3, dtype=float)
            model_gravity_base = np.zeros(6, dtype=float)
            model_gravity_delta_base = np.zeros(6, dtype=float)

            if gello_ctx is not None:
                q, dq, sensor_pos_base, r_base_sensor = _read_gello_pose(gello_ctx)
                q_leader[: min(6, q.size)] = q[: min(6, q.size)]
                dq_leader[: min(6, dq.size)] = dq[: min(6, dq.size)]
                model_gravity_base = _payload_gravity_wrench_base(
                    r_base_sensor,
                    payload_mass,
                    payload_com,
                )
                model_gravity_delta_base = model_gravity_base - model_gravity_initial_base

            wrench_base_raw_uncorrected = _transform_sensor_wrench_to_base(wrench_sensor_raw, r_base_sensor)
            wrench_base_static_uncorrected = _transform_sensor_wrench_to_base(wrench_sensor_bias, r_base_sensor)
            wrench_base_raw = _apply_axis_correction(
                wrench_base_raw_uncorrected, base_axis_map, base_axis_signs
            )
            wrench_base_static = _apply_axis_correction(
                wrench_base_static_uncorrected, base_axis_map, base_axis_signs
            )
            model_gravity_base_corrected = _apply_axis_correction(
                model_gravity_base, base_axis_map, base_axis_signs
            )
            model_gravity_delta_base_corrected = _apply_axis_correction(
                model_gravity_delta_base, base_axis_map, base_axis_signs
            )
            # dynamic is the fair comparison after initial software tare: compensate only
            # the orientation-dependent model change relative to the start pose.
            wrench_base_dynamic = wrench_base_static - gravity_sign * model_gravity_delta_base_corrected
            # dynamic_abs mirrors the Phase-B conditioner option: subtract absolute modeled gravity.
            wrench_base_dynamic_abs = wrench_base_static - gravity_sign * model_gravity_base_corrected
            wrench_base_static_ema = comp_filters.static.update(
                wrench_base_static, sample_raw.sensor_timestamp_us
            )
            wrench_base_dynamic_ema = comp_filters.dynamic.update(
                wrench_base_dynamic, sample_raw.sensor_timestamp_us
            )
            wrench_base_dynamic_abs_ema = comp_filters.dynamic_abs.update(
                wrench_base_dynamic_abs, sample_raw.sensor_timestamp_us
            )
            wrench_base_static_conditioned = _condition_wrench(
                wrench_base_static_ema, deadband, saturation
            )
            wrench_base_dynamic_conditioned = _condition_wrench(
                wrench_base_dynamic_ema, deadband, saturation
            )
            wrench_calibration_prediction = np.zeros(6, dtype=float)
            if wrench_calibration is not None:
                wrench_calibration_prediction = wrench_calibration.predict(
                    r_base_sensor, calibration_r_ref_runtime
                )
            wrench_base_calibrated = wrench_base_static_ema - wrench_calibration_prediction
            wrench_base_calibrated_conditioned = _condition_wrench(
                wrench_base_calibrated, deadband, saturation
            )
            wrench_base_dynamic_abs_conditioned = _condition_wrench(
                wrench_base_dynamic_abs_ema, deadband, saturation
            )
            display_wrench = _select_display_wrench(
                str(args.compensation_mode),
                wrench_sensor_raw,
                wrench_base_static_conditioned,
                wrench_base_dynamic_conditioned,
                wrench_base_dynamic_abs_conditioned,
                wrench_base_calibrated_conditioned,
            )
            display_sample = _sample_from_wrench(sample_raw, display_wrench)

            if ema_enabled:
                is_new_sensor_frame = sample_raw.sensor_timestamp_us != last_filter_sensor_ts
                if is_new_sensor_frame:
                    if last_filter_sensor_ts is None:
                        filter_dt_s = 0.0
                    else:
                        filter_dt_s = max(
                            0.0,
                            (sample_raw.sensor_timestamp_us - last_filter_sensor_ts) * 1e-6,
                        )
                    if args.ema_cutoff > 0.0:
                        alpha = _ema_alpha_from_cutoff(args.ema_cutoff, filter_dt_s)
                    else:
                        alpha = args.ema_alpha
                    filtered_sample = _filtered_sample(display_sample, filtered_sample, alpha)
                    last_filter_sensor_ts = sample_raw.sensor_timestamp_us
                plot_sample = display_sample if args.ema_print_only else filtered_sample or display_sample
            else:
                plot_sample = display_sample
            plot_samples.append(plot_sample)

            if npz_logger is not None:
                status = sample_raw
                npz_logger.append(
                    host_t=float(read_t),
                    t_rel=float(read_t - start_t),
                    dt_s=float(dt_s),
                    sensor_timestamp_us=int(sample_raw.sensor_timestamp_us),
                    wrench_sensor_raw=wrench_sensor_raw.astype(np.float32),
                    wrench_sensor_bias=wrench_sensor_bias.astype(np.float32),
                    wrench_base_raw=wrench_base_raw.astype(np.float32),
                    wrench_base_raw_uncorrected=wrench_base_raw_uncorrected.astype(np.float32),
                    wrench_base_static=wrench_base_static.astype(np.float32),
                    wrench_base_dynamic=wrench_base_dynamic.astype(np.float32),
                    wrench_base_dynamic_abs=wrench_base_dynamic_abs.astype(np.float32),
                    wrench_base_static_ema=wrench_base_static_ema.astype(np.float32),
                    wrench_base_dynamic_ema=wrench_base_dynamic_ema.astype(np.float32),
                    wrench_base_dynamic_abs_ema=wrench_base_dynamic_abs_ema.astype(np.float32),
                    wrench_base_calibrated=wrench_base_calibrated.astype(np.float32),
                    wrench_calibration_prediction=wrench_calibration_prediction.astype(np.float32),
                    wrench_base_static_conditioned=wrench_base_static_conditioned.astype(np.float32),
                    wrench_base_dynamic_conditioned=wrench_base_dynamic_conditioned.astype(np.float32),
                    wrench_base_dynamic_abs_conditioned=wrench_base_dynamic_abs_conditioned.astype(np.float32),
                    wrench_base_calibrated_conditioned=wrench_base_calibrated_conditioned.astype(np.float32),
                    model_gravity_base=model_gravity_base.astype(np.float32),
                    model_gravity_delta_base=model_gravity_delta_base.astype(np.float32),
                    model_gravity_base_corrected=model_gravity_base_corrected.astype(np.float32),
                    model_gravity_delta_base_corrected=model_gravity_delta_base_corrected.astype(np.float32),
                    q_leader=q_leader.astype(np.float32),
                    dq_leader=dq_leader.astype(np.float32),
                    sensor_pos_base=sensor_pos_base.astype(np.float32),
                    r_base_sensor=r_base_sensor.astype(np.float32),
                    temperature_c=float(status.temperature_c),
                    status_val=int(status.status_val),
                    status_flags=np.asarray(
                        [status.throttled, status.overrange, status.invalid, status.raw],
                        dtype=bool,
                    ),
                )

            if read_t - last_print_t >= 1.0 / max(args.print_rate, 1e-6):
                label = f"mode={args.compensation_mode}"
                if ema_enabled and filtered_sample is not None:
                    print(label + " | ", end="")
                    _print_filtered_sample(
                        display_sample,
                        filtered_sample,
                        _frequency_hz(intervals_s),
                        expected_hz,
                    )
                else:
                    print(label + " | ", end="")
                    _print_sample(display_sample, _frequency_hz(intervals_s), expected_hz)
                last_print_t = read_t

            if plot and read_t - last_plot_t >= 1.0 / max(args.plot_rate, 1e-6):
                plot.update(plot_samples, start_t)
                last_plot_t = read_t

            if args.poll:
                sleep_s = max(0.0, next_poll_t - time.perf_counter())
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    except Exception as exc:
        exit_code = 1
        print(f"FATAL: {exc}", file=sys.stderr)
    finally:
        total_s = max(time.perf_counter() - start_t, 1e-9)
        if npz_logger is not None and npz_path is not None:
            metadata = {
                "script": str(Path(__file__).resolve()),
                "config": str(args.config.expanduser().resolve()),
                "use_gello": bool(args.use_gello),
                "gello_config": str(args.gello_config.expanduser().resolve()) if args.use_gello else "",
                "sensor_urdf": str(args.sensor_urdf.expanduser().resolve()) if args.sensor_urdf is not None else "",
                "bota_frame": str(args.bota_frame),
                "compensation_mode_display": str(args.compensation_mode),
                "hardware_tare": not bool(args.no_tare),
                "bias_seconds": float(args.bias_seconds),
                "bias_samples": int(bias_samples),
                "bias_sensor": bias_sensor.tolist(),
                "payload_mass_kg": float(payload_mass),
                "payload_com_sensor_m": payload_com.tolist(),
                "gravity_sign": float(gravity_sign),
                "model_gravity_initial_base": model_gravity_initial_base.tolist(),
                "wrench_calibration": str(args.wrench_calibration.expanduser().resolve()) if wrench_calibration is not None else "",
                "wrench_calibration_feature_mode": str(wrench_calibration.feature_mode) if wrench_calibration is not None else "",
                "wrench_calibration_target_field": str(wrench_calibration.target_field) if wrench_calibration is not None else "",
                "calibration_r_ref_runtime": calibration_r_ref_runtime.tolist(),
                "ema_alpha": float(args.ema_alpha),
                "ema_cutoff": float(args.ema_cutoff),
                "deadband": deadband.tolist(),
                "saturation": saturation.tolist(),
                "base_axis_map": base_axis_map.tolist(),
                "base_axis_signs": base_axis_signs.tolist(),
                "conditioned_fields_match_phase_b_order": True,
                "poll": bool(args.poll),
                "poll_rate": float(args.poll_rate),
                "expected_hz": float(expected_hz) if expected_hz else 0.0,
            }
            try:
                npz_logger.save(npz_path, metadata)
            except Exception as exc:
                exit_code = 1
                print(f"WARNING: failed to save NPZ: {exc}", file=sys.stderr)
        _close_driver(driver)
        if gello_ctx is not None:
            gello_ctx.close()
        print(
            f"Finished: samples={len(samples)}, elapsed={total_s:.3f} s, "
            f"recent_rate={_frequency_hz(intervals_s):.2f} Hz"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
