#!/usr/bin/env python3
"""Startup-safe UR5e <-> GELLO alignment helper.

This script provides two modes:
1) calibrate: compute leader offsets so GELLO decoded joints match current UR5e joints
2) verify:   check that current GELLO decoded joints match current UR5e joints
3) assist_match: slowly drive GELLO leader to configured startup pose

The script never commands UR5e motion. In assist_match mode, it commands GELLO only.
It is intended for safe bring-up checks before starting teleoperation.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro
import yaml  # type: ignore[import-not-found]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gello.dynamixel.driver import DynamixelDriver
from gello.robots.ur5e import URRobot


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ur5e_gello_factr_hw_V2.yaml"


@dataclass
class Args:
    mode: str = "calibrate"
    """Mode: 'calibrate', 'verify', or 'assist_match'."""

    verify_reference: str = "target"
    """Verify reference: 'target' (startup target pose) or 'mapped_ur' (current UR pose via teleop mapping)."""

    config_path: str = str(DEFAULT_CONFIG_PATH)
    """Path to the YAML config used to populate defaults."""


    gello_port: str | None = None
    gello_baudrate: int | None = None
    num_arm_joints: int | None = None

    ur_robot_ip: str | None = None

    # Leader decoding (raw -> decoded): q_decoded = sign * (q_raw - offset)
    joint_signs: Tuple[float, ...] | None = None
    joint_offsets: Tuple[float, ...] | None = None

    # Follower mapping: q_ur ~= map_sign * q_decoded + map_offset
    map_signs: Tuple[float, ...] | None = None
    map_offsets: Tuple[float, ...] | None = None

    # Driver torque-current mapping config (len >= used joints)
    servo_types: Tuple[str, ...] | None = None

    # Configured startup poses from YAML
    calibration_joint_pos: Tuple[float, ...] | None = None
    initial_match_joint_pos: Tuple[float, ...] | None = None

    # Sampling and search
    samples: int = 25
    sample_dt: float = 0.01
    search_min_pi: float = -10.0
    search_max_pi: float = 10.0
    search_steps: int = 721
    snap_to_pi_over_2: bool = True

    # Safety thresholds
    warn_error_deg: float = 8.0
    ok_error_deg: float = 3.0

    # Verify mode behavior
    continuous: bool = False
    print_hz: float = 5.0

    # Assist mode behavior
    assist_target: str = "initial"
    """Assist target pose: 'initial' uses initial_match_joint_pos, 'calibration' uses calibration_joint_pos."""
    assist_rate_hz: float = 50.0
    """Command update rate for assist mode."""
    assist_speed_deg_s: float = 20.0
    """Approximate joint speed limit for assist mode."""
    assist_min_duration_s: float = 2.0
    """Minimum move duration in assist mode."""
    assist_timeout_s: float = 20.0
    """Maximum allowed assist motion time."""
    assist_tolerance_deg: float = 3.0
    """Success tolerance after assist motion."""
    assist_max_displacement_deg: float = 45.0
    """Abort assist if required move exceeds this unless overridden."""
    assist_allow_large_motion: bool = False
    """If true, allow assist motion above assist_max_displacement_deg."""

    def __post_init__(self) -> None:
        if self.mode not in ("calibrate", "verify", "assist_match"):
            raise ValueError("mode must be 'calibrate', 'verify' or 'assist_match'")

        if self.verify_reference not in ("target", "mapped_ur"):
            raise ValueError("verify_reference must be 'target' or 'mapped_ur'")

        if self.assist_target not in ("initial", "calibration"):
            raise ValueError("assist_target must be 'initial' or 'calibration'")

        if self.config_path is not None and not self.config_path:
            raise ValueError("config_path cannot be empty")

        if self.num_arm_joints is not None and self.num_arm_joints <= 0:
            raise ValueError("num_arm_joints must be > 0")

        if self.gello_baudrate is not None and self.gello_baudrate <= 0:
            raise ValueError("gello_baudrate must be > 0")

        if self.joint_signs is not None and self.num_arm_joints is not None:
            if len(self.joint_signs) < self.num_arm_joints:
                raise ValueError("joint_signs must have at least num_arm_joints elements")

        if self.joint_offsets is not None and self.num_arm_joints is not None:
            if len(self.joint_offsets) < self.num_arm_joints:
                raise ValueError("joint_offsets must have at least num_arm_joints elements")

        if self.map_signs is not None and self.num_arm_joints is not None:
            if len(self.map_signs) < self.num_arm_joints:
                raise ValueError("map_signs must have at least num_arm_joints elements")

        if self.map_offsets is not None and self.num_arm_joints is not None:
            if len(self.map_offsets) < self.num_arm_joints:
                raise ValueError("map_offsets must have at least num_arm_joints elements")

        if self.servo_types is not None and self.num_arm_joints is not None:
            if len(self.servo_types) < self.num_arm_joints:
                raise ValueError("servo_types must have at least num_arm_joints elements")

        if self.joint_signs is not None and self.num_arm_joints is not None:
            for idx, sign in enumerate(self.joint_signs[: self.num_arm_joints]):
                if sign not in (-1, 1):
                    raise ValueError(f"joint_signs[{idx}] must be -1 or 1, got {sign}")

        if self.map_signs is not None and self.num_arm_joints is not None:
            for idx, sign in enumerate(self.map_signs[: self.num_arm_joints]):
                if sign not in (-1, 1):
                    raise ValueError(f"map_signs[{idx}] must be -1 or 1, got {sign}")

        if self.samples <= 0:
            raise ValueError("samples must be > 0")

        if self.search_steps < 3:
            raise ValueError("search_steps must be >= 3")

        if self.print_hz <= 0:
            raise ValueError("print_hz must be > 0")

        if self.assist_rate_hz <= 0:
            raise ValueError("assist_rate_hz must be > 0")

        if self.assist_speed_deg_s <= 0:
            raise ValueError("assist_speed_deg_s must be > 0")

        if self.assist_min_duration_s <= 0:
            raise ValueError("assist_min_duration_s must be > 0")

        if self.assist_timeout_s <= 0:
            raise ValueError("assist_timeout_s must be > 0")

        if self.assist_tolerance_deg <= 0:
            raise ValueError("assist_tolerance_deg must be > 0")

        if self.assist_max_displacement_deg <= 0:
            raise ValueError("assist_max_displacement_deg must be > 0")


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_tuple(value, *, cast=None, default=None) -> tuple:
    if value is None:
        value = default if default is not None else ()
    cast_fn = cast if cast is not None else (lambda item: item)
    return tuple(cast_fn(item) for item in value)


def _resolve_runtime_values(args: Args) -> tuple[Args, dict]:
    cfg = _load_yaml_config(args.config_path)
    dynamixel_cfg = cfg.get("dynamixel", {}) if isinstance(cfg.get("dynamixel", {}), dict) else {}
    arm_cfg = cfg.get("arm_teleop", {}) if isinstance(cfg.get("arm_teleop", {}), dict) else {}
    init_cfg = arm_cfg.get("initialization", {}) if isinstance(arm_cfg.get("initialization", {}), dict) else {}
    teleop_cfg = cfg.get("teleop", {}) if isinstance(cfg.get("teleop", {}), dict) else {}
    mapping_cfg = teleop_cfg.get("mapping", {}) if isinstance(teleop_cfg.get("mapping", {}), dict) else {}
    robot_cfg = teleop_cfg.get("robot", {}) if isinstance(teleop_cfg.get("robot", {}), dict) else {}

    num_arm_joints = int(args.num_arm_joints or arm_cfg.get("num_arm_joints", 6))

    resolved = Args(
        mode=args.mode,
        verify_reference=args.verify_reference,
        config_path=args.config_path,
        gello_port=args.gello_port or dynamixel_cfg.get("dynamixel_port", "/dev/ttyDXL_gello"),
        gello_baudrate=int(args.gello_baudrate or dynamixel_cfg.get("baudrate", 1000000)),
        num_arm_joints=num_arm_joints,
        ur_robot_ip=args.ur_robot_ip or robot_cfg.get("robot_ip", "192.168.1.11"),
        joint_signs=args.joint_signs or _as_tuple(dynamixel_cfg.get("joint_signs"), default=[1] * num_arm_joints),
        joint_offsets=args.joint_offsets or _as_tuple(init_cfg.get("joint_offsets"), default=[0.0] * num_arm_joints),
        map_signs=args.map_signs or _as_tuple(mapping_cfg.get("signs"), default=[1] * num_arm_joints),
        map_offsets=args.map_offsets or _as_tuple(mapping_cfg.get("offsets"), default=[0.0] * num_arm_joints),
        servo_types=args.servo_types or _as_tuple(dynamixel_cfg.get("servo_types"), cast=str, default=[]),
        calibration_joint_pos=args.calibration_joint_pos or _as_tuple(init_cfg.get("calibration_joint_pos"), default=[]),
        initial_match_joint_pos=args.initial_match_joint_pos or _as_tuple(init_cfg.get("initial_match_joint_pos"), default=[]),
        samples=args.samples,
        sample_dt=args.sample_dt,
        search_min_pi=args.search_min_pi,
        search_max_pi=args.search_max_pi,
        search_steps=args.search_steps,
        snap_to_pi_over_2=args.snap_to_pi_over_2,
        warn_error_deg=args.warn_error_deg,
        ok_error_deg=args.ok_error_deg,
        continuous=args.continuous,
        print_hz=args.print_hz,
    )
    return resolved, cfg


def _mean_samples(driver: DynamixelDriver, ur: URRobot, args: Args) -> tuple[np.ndarray, np.ndarray]:
    n_joints = int(args.num_arm_joints or 0)
    gello_buf = []
    ur_buf = []
    for _ in range(args.samples):
        gello_q = driver.get_joints()[:n_joints]
        ur_q = np.array(ur.get_joint_state()[:n_joints], dtype=float)
        gello_buf.append(gello_q)
        ur_buf.append(ur_q)
        time.sleep(args.sample_dt)

    return np.mean(np.array(gello_buf), axis=0), np.mean(np.array(ur_buf), axis=0)


def _decode_gello(raw_q: np.ndarray, signs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return signs * (raw_q - offsets)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference a-b in [-pi, pi)."""
    return _wrap_to_pi(a - b)


def _raw_from_decoded(decoded_q: np.ndarray, signs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Invert q_decoded = sign * (q_raw - offset) to get q_raw."""
    return signs * decoded_q + offsets


def _calibrate_offsets(raw_q: np.ndarray, target_q: np.ndarray, signs: np.ndarray, args: Args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_joints = int(args.num_arm_joints or 0)
    search_grid = np.linspace(args.search_min_pi * np.pi, args.search_max_pi * np.pi, args.search_steps)
    best_offsets = np.zeros(n_joints, dtype=float)
    best_errors = np.zeros(n_joints, dtype=float)

    for i in range(n_joints):
        best_offset = search_grid[0]
        best_error = float("inf")
        for offset in search_grid:
            decoded = signs[i] * (raw_q[i] - offset)
            err = abs(decoded - target_q[i])
            if err < best_error:
                best_error = err
                best_offset = offset
        best_offsets[i] = best_offset
        best_errors[i] = best_error

    snapped_offsets = np.round(best_offsets / (np.pi / 2.0)) * (np.pi / 2.0)
    return best_offsets, snapped_offsets, best_errors


def _print_alignment_table(raw_q: np.ndarray, decoded_q: np.ndarray, ur_q: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    err = _angle_diff(decoded_q, ur_q)
    print("\nJoint alignment summary:")
    print("-" * 92)
    print(f"{'J':>2} | {'GELLO raw [deg]':>15} | {'offset [deg]':>13} | {'decoded [deg]':>14} | {'ref [deg]':>10} | {'err [deg]':>10}")
    print("-" * 92)
    for i in range(len(decoded_q)):
        print(
            f"{i + 1:>2} | "
            f"{np.rad2deg(raw_q[i]):>+15.2f} | "
            f"{np.rad2deg(offsets[i]):>+13.2f} | "
            f"{np.rad2deg(decoded_q[i]):>+14.2f} | "
            f"{np.rad2deg(ur_q[i]):>+10.2f} | "
            f"{np.rad2deg(err[i]):>+10.2f}"
        )
    print("-" * 92)
    return err


def _status(max_abs_err_deg: float, args: Args) -> str:
    if max_abs_err_deg <= args.ok_error_deg:
        return "OK"
    if max_abs_err_deg <= args.warn_error_deg:
        return "WARN"
    return "UNSAFE"


def run(args: Args) -> None:
    args, cfg = _resolve_runtime_values(args)
    teleop_cfg = cfg.get("teleop", {}) if isinstance(cfg.get("teleop", {}), dict) else {}
    mapping_cfg = teleop_cfg.get("mapping", {}) if isinstance(teleop_cfg.get("mapping", {}), dict) else {}
    n_joints = int(args.num_arm_joints or 0)
    gello_port = str(args.gello_port or "/dev/ttyDXL_gello")
    gello_baudrate = int(args.gello_baudrate or 1000000)
    ur_robot_ip = str(args.ur_robot_ip or "192.168.1.11")
    joint_signs = tuple(args.joint_signs or ())
    joint_offsets = tuple(args.joint_offsets or ())
    map_signs = tuple(args.map_signs or ())
    map_offsets = tuple(args.map_offsets or ())
    servo_types = list((args.servo_types or ())[:n_joints])
    target_pose = np.array(
        (args.initial_match_joint_pos or args.calibration_joint_pos or (0.0,) * n_joints)[:n_joints],
        dtype=float,
    )

    print("=" * 70)
    print("UR5e <-> GELLO STARTUP ALIGNMENT")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    if args.mode == "verify":
        print(f"Verify reference: {args.verify_reference}")
    print(f"Config: {args.config_path}")
    print(f"GELLO port: {gello_port}")
    print(f"UR robot ip: {ur_robot_ip}")
    if args.mode == "assist_match":
        print("This script will command GELLO only (slow assist). UR5e remains read-only.")
    else:
        print("This script does not command motion. It only reads states and computes checks.")
    if args.initial_match_joint_pos:
        print(f"Startup target from config: {[f'{np.rad2deg(x):.1f}°' for x in target_pose]}")
    elif args.calibration_joint_pos:
        print(f"Calibration target from config: {[f'{np.rad2deg(x):.1f}°' for x in target_pose]}")

    joint_ids = list(range(1, n_joints + 1))

    driver = None
    ur = None
    try:
        driver = DynamixelDriver(
            ids=joint_ids,
            port=gello_port,
            baudrate=gello_baudrate,
            servo_types=servo_types,
        )
        ur = URRobot(robot_ip=ur_robot_ip, no_gripper=True)

        # Warmup reads for stable averages
        for _ in range(8):
            driver.get_joints()
            ur.get_joint_state()

        signs = np.array(joint_signs[:n_joints], dtype=float)
        map_signs = np.array(map_signs[:n_joints], dtype=float)
        leader_offsets = np.array(joint_offsets[:n_joints], dtype=float)
        follower_offsets = np.array(map_offsets[:n_joints], dtype=float)

        if args.mode == "calibrate":
            raw_q, ur_q = _mean_samples(driver, ur, args)
            best_offsets, snapped_offsets, search_fit_error = _calibrate_offsets(raw_q, target_pose, signs, args)

            chosen_offsets = snapped_offsets if args.snap_to_pi_over_2 else best_offsets
            decoded_q = _decode_gello(raw_q, signs, chosen_offsets)
            leader_err = decoded_q - target_pose
            follower_err = ur_q - target_pose
            _print_alignment_table(raw_q, decoded_q, target_pose, chosen_offsets)

            map_offsets = target_pose - map_signs * decoded_q

            max_abs_err_deg = float(
                max(
                    np.max(np.abs(np.rad2deg(leader_err))),
                    np.max(np.abs(np.rad2deg(follower_err))),
                )
            )
            print("\nComputed leader offsets:")
            print(f"  config_leader_offsets: {[float(f'{x:.6f}') for x in leader_offsets]}")
            print(f"  joint_offsets_rad: {[float(f'{x:.6f}') for x in chosen_offsets]}")
            print(
                "  joint_offsets_pi_over_2: ["
                + ", ".join([f"{int(np.round(x / (np.pi / 2.0)))}*np.pi/2" for x in chosen_offsets])
                + "]"
            )
            print(f"  calibration_search_fit_error_deg: {[float(f'{x:.3f}') for x in np.rad2deg(search_fit_error)]}")

            print("\nTeleop mapping snippet (UR follower side):")
            print(f"  signs: {[int(x) for x in map_signs.tolist()]}")
            print(f"  offsets: {[float(f'{x:.6f}') for x in map_offsets]}")
            print(f"  config_offsets: {[float(f'{x:.6f}') for x in follower_offsets]}")
            print(f"  follower_pose_error_deg: {[float(f'{x:.3f}') for x in np.rad2deg(follower_err)]}")
            if bool(mapping_cfg.get("auto_align", True)):
                print("  note: teleop.mapping.auto_align is enabled in the YAML")

            print("\nStartup safety result:")
            print(f"  max_abs_error_deg: {max_abs_err_deg:.3f}")
            print(f"  status: {_status(max_abs_err_deg, args)}")
            if max_abs_err_deg > args.warn_error_deg:
                print("  action: Re-check the configured startup pose and retry calibration.")

        elif args.mode == "verify":
            offsets = leader_offsets
            map_signs_arr = np.array(map_signs[:n_joints], dtype=float)
            map_offsets_arr = np.array(map_offsets[:n_joints], dtype=float)

            def verify_once() -> tuple[
                float,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                str,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
                raw_q, ur_q = _mean_samples(driver, ur, args)
                decoded_q = _decode_gello(raw_q, signs, offsets)

                if args.verify_reference == "mapped_ur":
                    # Compare decoded leader joints against current UR joints through mapping,
                    # i.e., does map_sign * q_leader + map_offset match q_ur now?
                    follower_pred = map_signs_arr * decoded_q + map_offsets_arr
                    follower_err = _angle_diff(follower_pred, ur_q)
                    # Express error back in leader space for easier interpretation.
                    leader_ref_from_ur = (ur_q - map_offsets_arr) / np.where(
                        np.abs(map_signs_arr) < 1e-9, 1.0, map_signs_arr
                    )
                    leader_err = _angle_diff(decoded_q, leader_ref_from_ur)
                    max_abs_err_deg = float(np.max(np.abs(np.rad2deg(follower_err))))
                    ref_label = "current UR pose via teleop mapping"

                    # Offset consistency diagnostic:
                    # implied_offset = raw - sign * q_ref
                    implied_offsets = raw_q - signs * leader_ref_from_ur
                    step = np.pi
                    implied_k = np.round(implied_offsets / step)
                    implied_permanent = implied_k * step
                    implied_residual = implied_offsets - implied_permanent
                    return (
                        max_abs_err_deg,
                        raw_q,
                        decoded_q,
                        leader_ref_from_ur,
                        leader_err,
                        follower_err,
                        ref_label,
                        implied_permanent,
                        implied_residual,
                        implied_k,
                    )

                leader_err = _angle_diff(decoded_q, target_pose)
                follower_err = _angle_diff(ur_q, target_pose)
                max_abs_err_deg = float(
                    max(
                        np.max(np.abs(np.rad2deg(leader_err))),
                        np.max(np.abs(np.rad2deg(follower_err))),
                    )
                )
                ref_label = "configured startup target pose"
                return (
                    max_abs_err_deg,
                    raw_q,
                    decoded_q,
                    target_pose,
                    leader_err,
                    follower_err,
                    ref_label,
                    np.zeros_like(decoded_q),
                    np.zeros_like(decoded_q),
                    np.zeros_like(decoded_q),
                )

            if args.continuous:
                period = 1.0 / args.print_hz
                print("\nContinuous verify mode. Ctrl+C to stop.")
                while True:
                    t0 = time.monotonic()
                    max_err, _, decoded_q, ref_q, leader_err, _, ref_label, _, _, _ = verify_once()
                    st = _status(max_err, args)
                    print(
                        f"status={st:>6}  max_err_deg={max_err:6.2f}  "
                        f"leader0={np.rad2deg(decoded_q[0]):7.2f}  "
                        f"ref0={np.rad2deg(ref_q[0]):7.2f}  "
                        f"err0={np.rad2deg(leader_err[0]):+7.2f}  "
                        f"ref='{ref_label}'"
                    )
                    dt = time.monotonic() - t0
                    if dt < period:
                        time.sleep(period - dt)
            else:
                max_err, raw_q, decoded_q, ref_q, leader_err, follower_err, ref_label, implied_permanent, implied_residual, implied_k = verify_once()
                print(f"\nVerification reference: {ref_label}")
                _print_alignment_table(raw_q, decoded_q, ref_q, offsets)
                print("\nStartup safety result:")
                print(f"  max_abs_error_deg: {max_err:.3f}")
                print(f"  status: {_status(max_err, args)}")
                print(
                    "  follower_mapping_error_deg: "
                    f"{[float(f'{x:.3f}') for x in np.rad2deg(follower_err)]}"
                )
                if args.verify_reference == "mapped_ur":
                    print(
                        "  implied_permanent_offset_pi: "
                        f"{[float(f'{x:.3f}') for x in (implied_permanent / np.pi)]}"
                    )
                    print(
                        "  implied_turn_index_k: "
                        f"{[int(x) for x in implied_k.astype(int)]}"
                    )
                    print(
                        "  implied_residual_deg (hold/model error): "
                        f"{[float(f'{x:.3f}') for x in np.rad2deg(implied_residual)]}"
                    )
                if max_err > args.warn_error_deg:
                    if args.verify_reference == "mapped_ur":
                        print("  action: Leader/Follower are not aligned in current pose. Re-check permanent offsets and mapping.")
                    else:
                        print("  action: Do not start teleop. Re-align to the configured startup pose or re-run calibrate mode.")

        else:  # assist_match
            if driver is None:
                raise RuntimeError("Dynamixel driver not available")

            target_decoded = (
                np.array(args.initial_match_joint_pos[:n_joints], dtype=float)
                if args.assist_target == "initial"
                else np.array(args.calibration_joint_pos[:n_joints], dtype=float)
            )

            raw_now, _ = driver.get_positions_and_velocities()
            raw_now = np.asarray(raw_now[:n_joints], dtype=float)

            # Compute desired raw target and wrap to nearest turn relative to current raw,
            # so the assist movement takes the shortest path and avoids large rotations.
            raw_target_nominal = _raw_from_decoded(target_decoded, signs, leader_offsets)
            raw_target = raw_now + _angle_diff(raw_target_nominal, raw_now)

            diff = _angle_diff(raw_target, raw_now)
            max_disp_deg = float(np.max(np.abs(np.rad2deg(diff))))
            duration_s = max(args.assist_min_duration_s, max_disp_deg / args.assist_speed_deg_s)
            duration_s = min(duration_s, args.assist_timeout_s)
            steps = max(2, int(duration_s * args.assist_rate_hz))

            print("\nAssist motion setup:")
            print(f"  target: {args.assist_target}")
            print(f"  max_displacement_deg: {max_disp_deg:.2f}")
            print(f"  duration_s: {duration_s:.2f}")
            print(f"  steps: {steps}")

            if (not args.assist_allow_large_motion) and max_disp_deg > args.assist_max_displacement_deg:
                print("\nAssist aborted for safety:")
                print(
                    f"  required move ({max_disp_deg:.2f} deg) exceeds "
                    f"assist_max_displacement_deg ({args.assist_max_displacement_deg:.2f} deg)."
                )
                print("  hint: initial_match_joint_pos should be a decoded leader pose, not raw offsets.")
                print("  hint: if intentional, rerun with --assist-allow-large-motion.")
                return

            driver.set_torque_mode(False)
            time.sleep(0.02)
            driver.set_operating_mode(3)  # position mode
            time.sleep(0.02)
            driver.set_torque_mode(True)
            time.sleep(0.02)

            print("Starting slow assist motion... (Ctrl+C to stop)")
            interrupted = False
            try:
                for jnt in np.linspace(raw_now, raw_target, steps):
                    driver.set_joints(jnt.tolist())
                    time.sleep(1.0 / args.assist_rate_hz)
            except KeyboardInterrupt:
                interrupted = True
                print("\nAssist interrupted by user.")

            # Hold briefly and verify result in decoded space.
            time.sleep(0.2)
            raw_end, _ = driver.get_positions_and_velocities()
            raw_end = np.asarray(raw_end[:n_joints], dtype=float)
            decoded_end = _decode_gello(raw_end, signs, leader_offsets)
            err_end = _angle_diff(decoded_end, target_decoded)
            max_err_deg = float(np.max(np.abs(np.rad2deg(err_end))))

            print("\nAssist result:")
            print(f"  decoded_target_deg: {[float(f'{x:.2f}') for x in np.rad2deg(target_decoded)]}")
            print(f"  decoded_final_deg:  {[float(f'{x:.2f}') for x in np.rad2deg(decoded_end)]}")
            print(f"  error_deg:          {[float(f'{x:.2f}') for x in np.rad2deg(err_end)]}")
            print(f"  max_abs_error_deg:  {max_err_deg:.2f}")
            if interrupted:
                print("  status: INTERRUPTED")
                return
            if max_err_deg <= args.assist_tolerance_deg:
                print("  status: OK")
            else:
                print("  status: WARN (reposition leader manually and re-run assist_match)")

    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


if __name__ == "__main__":
    run(tyro.cli(Args))
