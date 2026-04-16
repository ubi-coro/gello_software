#!/usr/bin/env python3
"""Startup-safe UR5e <-> GELLO alignment helper.

This script provides two modes:
1) calibrate: compute leader offsets so GELLO decoded joints match current UR5e joints
2) verify:   check that current GELLO decoded joints match current UR5e joints

The script is read-only with respect to robot motion: it never commands the UR5e or GELLO.
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
    """Mode: 'calibrate' (compute offsets) or 'verify' (check existing offsets)."""

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

    def __post_init__(self) -> None:
        if self.mode not in ("calibrate", "verify"):
            raise ValueError("mode must be 'calibrate' or 'verify'")

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
    err = decoded_q - ur_q
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
    print(f"Config: {args.config_path}")
    print(f"GELLO port: {gello_port}")
    print(f"UR robot ip: {ur_robot_ip}")
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

        else:
            offsets = leader_offsets

            def verify_once() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
                raw_q, ur_q = _mean_samples(driver, ur, args)
                decoded_q = _decode_gello(raw_q, signs, offsets)
                leader_err = decoded_q - target_pose
                follower_err = ur_q - target_pose
                max_abs_err_deg = float(
                    max(
                        np.max(np.abs(np.rad2deg(leader_err))),
                        np.max(np.abs(np.rad2deg(follower_err))),
                    )
                )
                return max_abs_err_deg, raw_q, decoded_q, ur_q

            if args.continuous:
                period = 1.0 / args.print_hz
                print("\nContinuous verify mode. Ctrl+C to stop.")
                while True:
                    t0 = time.monotonic()
                    max_err, raw_q, decoded_q, ur_q = verify_once()
                    st = _status(max_err, args)
                    print(
                        f"status={st:>6}  max_err_deg={max_err:6.2f}  "
                        f"leader0={np.rad2deg(decoded_q[0]):7.2f}  ur0={np.rad2deg(ur_q[0]):7.2f}"
                    )
                    dt = time.monotonic() - t0
                    if dt < period:
                        time.sleep(period - dt)
            else:
                max_err, raw_q, decoded_q, ur_q = verify_once()
                _print_alignment_table(raw_q, decoded_q, target_pose, offsets)
                print("\nStartup safety result:")
                print(f"  max_abs_error_deg: {max_err:.3f}")
                print(f"  status: {_status(max_err, args)}")
                if max_err > args.warn_error_deg:
                    print("  action: Do not start teleop. Re-align to the configured startup pose or re-run calibrate mode.")

    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


if __name__ == "__main__":
    run(tyro.cli(Args))
