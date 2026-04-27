#!/usr/bin/env python3
"""
Turn-Disambiguation Verification Script
======================================

Three test levels (run in order):
  math  - Pure math unit tests, no hardware needed
  read  - Read-only hardware check (GELLO + UR5e), no motion
  sim   - Simulated multi-boot reproducibility (reads once, simulates reboots)

Usage:
    python scripts/turn_disambiguation.py math
    python scripts/turn_disambiguation.py read --config configs/ur5e_gello_factr_hw_V2.yaml
    python scripts/turn_disambiguation.py sim  --config configs/ur5e_gello_factr_hw_V2.yaml

The script NEVER commands any motion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def wrap_to_pi(x: Any) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def disambiguate_leader_offset(
    raw: float,
    permanent_offset: float,
    sign: float,
    expected_decoded: float,
) -> tuple[float, int, float]:
    """Return (session_offset, k, residual_rad)."""
    base_decoded = sign * (raw - permanent_offset)
    k = int(np.round((base_decoded - expected_decoded) / (2.0 * np.pi)))
    session_offset = permanent_offset + sign * k * 2.0 * np.pi
    final_decoded = sign * (raw - session_offset)
    residual = float(wrap_to_pi(final_decoded - expected_decoded))
    return float(session_offset), k, residual


def turn_disambiguate_mapping(
    leader_mapped: np.ndarray,
    follower_actual: np.ndarray,
    map_offsets: np.ndarray,
) -> np.ndarray:
    """Return corrected map_offsets (only integer-2pi adjustment)."""
    expected_follower = leader_mapped + map_offsets
    diff = wrap_to_pi(expected_follower - follower_actual)
    turn_error = expected_follower - follower_actual - diff
    return map_offsets - turn_error


def nearest_turn_target(target: Any, reference: Any) -> np.ndarray:
    """Return equivalent target nearest to reference."""
    target_arr = np.asarray(target, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    return ref_arr + wrap_to_pi(target_arr - ref_arr)


def decode_leader(raw: np.ndarray, offsets: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """q_decoded = signs * (raw - offsets)."""
    return signs * (raw - offsets)


def leader_to_follower(
    decoded: np.ndarray,
    map_signs: np.ndarray,
    map_offsets: np.ndarray,
    map_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mapping: q_follower = map_signs * decoded[map_index] + map_offsets."""
    if map_index is not None:
        decoded = decoded[map_index]
    return map_signs * decoded + map_offsets


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_pass_count = 0
_fail_count = 0


def _reset_counters() -> None:
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  {_GREEN}PASS{_RESET}  {name}" + (f"  ({detail})" if detail else ""))
    else:
        _fail_count += 1
        print(f"  {_RED}FAIL{_RESET}  {name}" + (f"  ({detail})" if detail else ""))


def _section(title: str) -> None:
    print(f"\n{_BOLD}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{_RESET}")


def _summary() -> int:
    total = _pass_count + _fail_count
    if _fail_count == 0:
        print(f"\n{_GREEN}{_BOLD}ALL {total} CHECKS PASSED{_RESET}\n")
    else:
        print(f"\n{_RED}{_BOLD}{_fail_count}/{total} CHECKS FAILED{_RESET}\n")
    return 0 if _fail_count == 0 else 1


def _checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "turn_disambig_checkpoint.json"


def _resolve_port(raw_port: str) -> str:
    port = str(raw_port)
    if os.name == "nt":
        return port
    if port.startswith("/"):
        return port
    return "/dev/serial/by-id/" + port


def _load_runtime_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_mapping(
    mapping_cfg: dict[str, Any],
    num_arm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_map_cfg = np.array(mapping_cfg.get("index_map", list(range(num_arm))), dtype=int)
    map_signs_cfg = np.array(mapping_cfg.get("signs", [1.0] * num_arm), dtype=float)
    map_offsets_cfg = np.array(mapping_cfg.get("offsets", [0.0] * num_arm), dtype=float)

    map_dims = int(min(num_arm, len(index_map_cfg), len(map_signs_cfg), len(map_offsets_cfg)))
    if map_dims <= 0:
        raise ValueError("teleop.mapping must define at least one dimension")

    return (
        index_map_cfg[:map_dims],
        map_signs_cfg[:map_dims],
        map_offsets_cfg[:map_dims],
    )


def _extract_servo_types(dyn_cfg: dict[str, Any], num_arm: int) -> list[str]:
    """Return servo types aligned to the arm joint dimension."""
    raw_servo_types = dyn_cfg.get("servo_types", [])
    servo_types = [str(x) for x in raw_servo_types]
    if len(servo_types) < num_arm:
        raise ValueError(
            f"dynamixel.servo_types has {len(servo_types)} entries but num_arm_joints is {num_arm}"
        )
    if len(servo_types) > num_arm:
        print(
            f"  {_YELLOW}WARNING: dynamixel.servo_types has {len(servo_types)} entries; "
            f"using first {num_arm} arm joints.{_RESET}"
        )
    return servo_types[:num_arm]


def run_math_checks() -> int:
    _reset_counters()

    _section("1a  wrap_to_pi edge cases")

    _check("wrap(0) == 0", abs(float(wrap_to_pi(0.0))) < 1e-12)
    _check("wrap(pi) == -pi or pi", abs(abs(float(wrap_to_pi(np.pi))) - np.pi) < 1e-12)
    _check("wrap(2pi) ~= 0", abs(float(wrap_to_pi(2 * np.pi))) < 1e-12)
    _check("wrap(-pi) == -pi", abs(float(wrap_to_pi(-np.pi)) - (-np.pi)) < 1e-12)
    _check("wrap(3pi) ~= -pi or pi", abs(abs(float(wrap_to_pi(3 * np.pi))) - np.pi) < 1e-12)
    _check("wrap(370deg) ~= 10deg", abs(float(wrap_to_pi(np.deg2rad(370))) - np.deg2rad(10)) < 1e-10)
    _check("wrap(-370deg) ~= -10deg", abs(float(wrap_to_pi(np.deg2rad(-370))) - np.deg2rad(-10)) < 1e-10)

    arr = wrap_to_pi(np.array([0, 2 * np.pi, -2 * np.pi, 4 * np.pi]))
    _check("wrap(array) all ~= 0", float(np.max(np.abs(arr))) < 1e-12)

    _section("1b  disambiguate_leader_offset")

    _, k, res = disambiguate_leader_offset(
        raw=1.57, permanent_offset=np.pi / 2, sign=1.0, expected_decoded=0.0
    )
    _check("sign=+1, k=0", k == 0, f"k={k}, res={np.rad2deg(res):.3f}deg")
    _check("residual < 1deg", abs(np.rad2deg(res)) < 1.0)

    _, k, res = disambiguate_leader_offset(
        raw=1.57 + 2 * np.pi,
        permanent_offset=np.pi / 2,
        sign=1.0,
        expected_decoded=0.0,
    )
    _check("sign=+1, raw+2pi -> k=1", k == 1, f"k={k}, res={np.rad2deg(res):.3f}deg")
    _check("residual < 1deg", abs(np.rad2deg(res)) < 1.0)

    _, k, _ = disambiguate_leader_offset(
        raw=1.57 - 2 * np.pi,
        permanent_offset=np.pi / 2,
        sign=1.0,
        expected_decoded=0.0,
    )
    _check("sign=+1, raw-2pi -> k=-1", k == -1, f"k={k}")

    _, k, _ = disambiguate_leader_offset(
        raw=1.57, permanent_offset=np.pi / 2, sign=-1.0, expected_decoded=0.0
    )
    _check("sign=-1, k=0", k == 0, f"k={k}")

    _, k, _ = disambiguate_leader_offset(
        raw=1.57 + 2 * np.pi,
        permanent_offset=np.pi / 2,
        sign=-1.0,
        expected_decoded=0.0,
    )
    _check("sign=-1, raw+2pi -> k=-1", k == -1, f"k={k}")

    _, k, _ = disambiguate_leader_offset(
        raw=1.57 - 4 * np.pi,
        permanent_offset=np.pi / 2,
        sign=-1.0,
        expected_decoded=0.0,
    )
    _check("sign=-1, raw-4pi -> k=2", k == 2, f"k={k}")

    _, k, res = disambiguate_leader_offset(
        raw=3.14 + 2 * np.pi,
        permanent_offset=np.pi / 2,
        sign=1.0,
        expected_decoded=np.pi / 2,
    )
    _check(
        "expected=pi/2, raw+2pi",
        k == 1 and abs(np.rad2deg(res)) < 1.0,
        f"k={k}, res={np.rad2deg(res):.3f}deg",
    )

    _section("1c  turn_disambiguate_mapping_offsets")

    leader_mapped = np.array([0.5, -1.0, 2.0], dtype=float)
    follower_actual = np.array([0.5, -1.0, 2.0], dtype=float)
    map_off = np.array([0.0, 0.0, 0.0], dtype=float)

    new_off = turn_disambiguate_mapping(leader_mapped, follower_actual, map_off)
    _check("perfect match -> offsets unchanged", float(np.max(np.abs(new_off - map_off))) < 1e-12)

    follower_shifted = follower_actual.copy()
    follower_shifted[0] += 2 * np.pi
    new_off = turn_disambiguate_mapping(leader_mapped, follower_shifted, map_off)
    _check(
        "follower+2pi -> offset adjusted by +2pi",
        abs(float(new_off[0]) - (2 * np.pi)) < 1e-10,
        f"new_off[0]={new_off[0]:.4f}",
    )

    expected_after = leader_mapped + new_off
    err_after = wrap_to_pi(expected_after - follower_shifted)
    _check("corrected error < 1e-10", float(np.max(np.abs(err_after))) < 1e-10)

    map_off_pi = np.array([0.0, np.pi, 0.0], dtype=float)
    follower_pi = np.array([0.5, -1.0 + np.pi, 2.0], dtype=float)
    follower_pi_shifted = follower_pi.copy()
    follower_pi_shifted[1] -= 2 * np.pi

    new_off = turn_disambiguate_mapping(leader_mapped, follower_pi_shifted, map_off_pi)
    expected_after = leader_mapped + new_off
    err_after = wrap_to_pi(expected_after - follower_pi_shifted)
    _check("permanent pi offset + follower-2pi -> error < 1e-10", float(np.max(np.abs(err_after))) < 1e-10)

    offset_change = new_off - map_off_pi
    k_change = np.round(offset_change / (2 * np.pi))
    residual_change = offset_change - k_change * 2 * np.pi
    _check("offset change is integer 2pi", float(np.max(np.abs(residual_change))) < 1e-10)

    _section("1d  nearest_turn_target")

    _check("target=0, ref=0 -> 0", abs(float(nearest_turn_target(0.0, 0.0))) < 1e-12)
    _check(
        "target=0, ref=2pi -> 2pi",
        abs(float(nearest_turn_target(0.0, 2 * np.pi)) - 2 * np.pi) < 1e-12,
    )
    _check(
        "target=2pi, ref=0.01 -> ~=0.01",
        abs(float(nearest_turn_target(2 * np.pi, 0.01)) - 0.01) < 0.02,
    )

    targets = np.array([0.0, 2 * np.pi, -2 * np.pi], dtype=float)
    refs = np.array([0.1, 6.3, -6.2], dtype=float)
    results = nearest_turn_target(targets, refs)
    errors = np.abs(results - refs)
    _check("array results near refs", float(np.max(errors)) < 0.15)

    _section("1e  Full pipeline: Phase1 -> reboot -> Phase2 consistency")

    permanent_offsets = np.array([np.pi / 2, 3 * np.pi / 2, np.pi, np.pi / 2, np.pi, 0.0], dtype=float)
    joint_signs = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0], dtype=float)
    calibration_pose = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi], dtype=float)
    map_signs = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0], dtype=float)
    map_offsets = np.array([0.0, 0.0, np.pi, 0.0, 0.0, 0.0], dtype=float)

    raw_boot1 = permanent_offsets + joint_signs * calibration_pose
    rng = np.random.default_rng(42)
    hold_error = rng.normal(0.0, np.deg2rad(1.5), size=6)
    raw_boot1 += hold_error

    session1 = np.zeros(6, dtype=float)
    for i in range(6):
        so, _, _ = disambiguate_leader_offset(
            raw_boot1[i], permanent_offsets[i], joint_signs[i], calibration_pose[i]
        )
        session1[i] = so

    decoded_boot1 = decode_leader(raw_boot1, session1, joint_signs)
    follower_boot1 = leader_to_follower(decoded_boot1, map_signs, map_offsets)

    turn_shifts = rng.integers(-3, 4, size=6) * 2 * np.pi
    raw_boot2 = raw_boot1 + turn_shifts
    hold_error2 = rng.normal(0.0, np.deg2rad(1.5), size=6)
    raw_boot2 += hold_error2 - hold_error

    session2 = np.zeros(6, dtype=float)
    for i in range(6):
        so, _, _ = disambiguate_leader_offset(
            raw_boot2[i], permanent_offsets[i], joint_signs[i], calibration_pose[i]
        )
        session2[i] = so

    decoded_boot2 = decode_leader(raw_boot2, session2, joint_signs)
    follower_boot2 = leader_to_follower(decoded_boot2, map_signs, map_offsets)

    decoded_diff = wrap_to_pi(decoded_boot1 - decoded_boot2)
    max_decoded_diff_deg = float(np.max(np.abs(np.rad2deg(decoded_diff))))
    _check(
        "decoded diff across boots < 5deg (hold-error only)",
        max_decoded_diff_deg < 5.0,
        f"max_diff={max_decoded_diff_deg:.2f}deg",
    )

    follower_diff = wrap_to_pi(follower_boot1 - follower_boot2)
    max_follower_diff_deg = float(np.max(np.abs(np.rad2deg(follower_diff))))
    _check(
        "follower target diff across boots < 5deg",
        max_follower_diff_deg < 5.0,
        f"max_diff={max_follower_diff_deg:.2f}deg",
    )

    raw_boot3 = raw_boot1 + np.array([10, -8, 6, -12, 4, -6], dtype=float) * np.pi
    raw_boot3 += rng.normal(0.0, np.deg2rad(1.0), size=6) - hold_error

    session3 = np.zeros(6, dtype=float)
    for i in range(6):
        so, _, _ = disambiguate_leader_offset(
            raw_boot3[i], permanent_offsets[i], joint_signs[i], calibration_pose[i]
        )
        session3[i] = so

    decoded_boot3 = decode_leader(raw_boot3, session3, joint_signs)
    decoded_diff3 = wrap_to_pi(decoded_boot1 - decoded_boot3)
    max_decoded_diff3_deg = float(np.max(np.abs(np.rad2deg(decoded_diff3))))
    _check("extreme turns: decoded diff < 5deg", max_decoded_diff3_deg < 5.0, f"max_diff={max_decoded_diff3_deg:.2f}deg")

    _section("1f  DAgger scenario: recorded actions replayed after reboot")

    traj_decoded = np.zeros((5, 6), dtype=float)
    for t in range(5):
        traj_decoded[t] = calibration_pose + np.deg2rad(t * 5) * np.array([1, -0.5, 0.3, 0.8, -0.2, 0.4], dtype=float)

    traj_follower_boot1 = np.array([leader_to_follower(td, map_signs, map_offsets) for td in traj_decoded])
    traj_follower_boot2 = np.array([leader_to_follower(td, map_signs, map_offsets) for td in traj_decoded])

    traj_diff = float(np.max(np.abs(traj_follower_boot1 - traj_follower_boot2)))
    _check("DAgger replay: follower targets identical", traj_diff < 1e-12, f"max_diff={traj_diff:.2e}")

    bad_map_offsets = map_offsets + np.array([0, 0, 2 * np.pi, 0, 0, 0], dtype=float)
    traj_follower_bad = np.array([leader_to_follower(td, map_signs, bad_map_offsets) for td in traj_decoded])

    bad_diff = float(np.max(np.abs(wrap_to_pi(traj_follower_boot1 - traj_follower_bad))))
    _check(
        "DAgger replay with bad offset: wrap_to_pi hides error",
        bad_diff < 1e-10,
        "wrapped comparison cannot detect 2pi offset (expected)",
    )

    raw_diff = float(np.max(np.abs(traj_follower_boot1 - traj_follower_bad)))
    _check("raw difference reveals 2pi jump", raw_diff > 6.0, f"raw_diff={raw_diff:.2f} rad")

    return _summary()


def run_read_only_hardware_checks(config_path: str) -> int:
    _reset_counters()
    _section("2  Read-Only Hardware Verification")

    cfg = _load_runtime_config(config_path)

    dyn_cfg = cfg.get("dynamixel", {})
    arm_cfg = cfg.get("arm_teleop", {})
    init_cfg = arm_cfg.get("initialization", {})
    teleop_cfg = cfg.get("teleop", {})
    mapping_cfg = teleop_cfg.get("mapping", {})
    robot_cfg = teleop_cfg.get("robot", {})

    num_arm = int(arm_cfg.get("num_arm_joints", 6))
    joint_signs = np.array(dyn_cfg.get("joint_signs", [1] * num_arm)[:num_arm], dtype=float)
    permanent_offsets = np.array(init_cfg.get("joint_offsets", [0.0] * num_arm)[:num_arm], dtype=float)
    calibration_pose = np.array(init_cfg.get("calibration_joint_pos", [0.0] * num_arm)[:num_arm], dtype=float)

    map_index, map_signs_cfg, map_offsets_cfg = _extract_mapping(mapping_cfg, num_arm)

    print(f"\n  Config: {config_path}")
    print(f"  Permanent offsets (pi): {[f'{x / np.pi:.2f}' for x in permanent_offsets]}")
    print(f"  Joint signs: {joint_signs.tolist()}")
    print(f"  Calibration pose (deg): {[f'{np.rad2deg(x):.1f}' for x in calibration_pose]}")
    print(f"  Map signs: {map_signs_cfg.tolist()}")
    print(f"  Map offsets: {[f'{x:.4f}' for x in map_offsets_cfg]}")

    grid = np.pi / 2.0
    grid_residual = np.abs(np.mod(permanent_offsets, grid))
    grid_residual = np.minimum(grid_residual, grid - grid_residual)
    on_grid = bool(np.all(grid_residual < 0.01))
    if on_grid:
        _check(
            "permanent offsets on pi/2 grid",
            True,
            f"max residual={np.rad2deg(float(np.max(grid_residual))):.2f}deg",
        )
    else:
        print(
            f"  {_YELLOW}WARNING: permanent offsets not on pi/2 grid "
            f"(max residual={np.rad2deg(float(np.max(grid_residual))):.2f}deg). "
            f"This is acceptable when offsets were measured directly.{_RESET}"
        )

    from gello.dynamixel.driver import DynamixelDriver

    port = _resolve_port(str(dyn_cfg.get("dynamixel_port", "/dev/ttyUSB0")))
    baudrate = int(dyn_cfg.get("baudrate", 4000000))
    servo_types = _extract_servo_types(dyn_cfg, num_arm)
    joint_ids = list(range(1, num_arm + 1))

    print(f"\n  Connecting to GELLO on {port}...")
    driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate, servo_types=servo_types)

    try:
        for _ in range(10):
            driver.get_joints()

        samples = []
        for _ in range(25):
            samples.append(np.array(driver.get_joints()[:num_arm], dtype=float))
            time.sleep(0.005)
        raw_mean = np.mean(np.array(samples), axis=0)
        raw_std = np.std(np.array(samples), axis=0)
    finally:
        driver.close()

    print(f"\n  Raw GELLO (deg):  {[f'{np.rad2deg(x):+.1f}' for x in raw_mean]}")
    print(f"  Raw noise (deg):  {[f'{np.rad2deg(x):.3f}' for x in raw_std]}")

    session_offsets = np.zeros(num_arm, dtype=float)
    k_values = np.zeros(num_arm, dtype=int)
    residuals = np.zeros(num_arm, dtype=float)

    for i in range(num_arm):
        so, k, res = disambiguate_leader_offset(
            raw_mean[i], permanent_offsets[i], joint_signs[i], calibration_pose[i]
        )
        session_offsets[i] = so
        k_values[i] = k
        residuals[i] = res

    decoded = decode_leader(raw_mean, session_offsets, joint_signs)

    print(f"\n  Turn indices k: {k_values.tolist()}")
    print(f"  Session offsets (deg): {[f'{np.rad2deg(x):+.1f}' for x in session_offsets]}")
    print(f"  Decoded (deg): {[f'{np.rad2deg(x):+.1f}' for x in decoded]}")
    print(f"  Residuals (deg): {[f'{np.rad2deg(x):+.2f}' for x in residuals]}")

    max_res_deg = float(np.max(np.abs(np.rad2deg(residuals))))
    _check("all residuals < 5deg", max_res_deg < 5.0, f"max={max_res_deg:.2f}deg")
    _check("all residuals < 10deg (relaxed)", max_res_deg < 10.0, f"max={max_res_deg:.2f}deg")

    ur_ip = str(robot_cfg.get("robot_ip", "172.22.22.2"))
    ur_q: Optional[np.ndarray] = None
    try:
        from gello.robots.ur5e import URRobot

        print(f"\n  Connecting to UR5e at {ur_ip}...")
        ur = URRobot(robot_ip=ur_ip, no_gripper=True)
        time.sleep(0.5)

        ur_samples = []
        for _ in range(10):
            ur_samples.append(np.array(ur.get_joint_state()[: len(map_index)], dtype=float))
            time.sleep(0.02)
        ur_q_local = np.mean(np.array(ur_samples), axis=0)
        ur_q = ur_q_local
        print(f"  UR5e joints (deg): {[f'{np.rad2deg(x):+.1f}' for x in ur_q_local]}")
    except Exception as e:
        print(f"  {_YELLOW}WARNING: Could not connect to UR5e: {e}{_RESET}")
        print("  Skipping follower mapping checks.")

    if ur_q is not None:
        leader_mapped = map_signs_cfg * decoded[map_index]
        expected_follower = leader_mapped + map_offsets_cfg

        mapping_err = wrap_to_pi(expected_follower - ur_q)
        max_mapping_err_deg = float(np.max(np.abs(np.rad2deg(mapping_err))))

        print("\n  Mapping check:")
        print(f"    Leader mapped (deg): {[f'{np.rad2deg(x):+.1f}' for x in leader_mapped]}")
        print(f"    Expected follower (deg): {[f'{np.rad2deg(x):+.1f}' for x in expected_follower]}")
        print(f"    Actual UR5e (deg): {[f'{np.rad2deg(x):+.1f}' for x in ur_q]}")
        print(f"    Mapping error (deg): {[f'{np.rad2deg(x):+.2f}' for x in mapping_err]}")

        _check("mapping error < 10deg", max_mapping_err_deg < 10.0, f"max={max_mapping_err_deg:.2f}deg")
        _check("mapping error < 5deg (tight)", max_mapping_err_deg < 5.0, f"max={max_mapping_err_deg:.2f}deg")

        corrected_offsets = turn_disambiguate_mapping(leader_mapped, ur_q, map_offsets_cfg)
        corrected_expected = leader_mapped + corrected_offsets
        corrected_err = wrap_to_pi(corrected_expected - ur_q)
        max_corrected_deg = float(np.max(np.abs(np.rad2deg(corrected_err))))

        offset_change = corrected_offsets - map_offsets_cfg
        k_mapping = np.round(offset_change / (2 * np.pi)).astype(int)

        print("\n  After mapping turn-disambiguation:")
        print(f"    Mapping k corrections: {k_mapping.tolist()}")
        print(f"    Corrected offsets: {[f'{x:.4f}' for x in corrected_offsets]}")
        print(f"    Corrected error (deg): {[f'{np.rad2deg(x):+.2f}' for x in corrected_err]}")

        _check("mapping turn-disambiguated error < 5deg", max_corrected_deg < 5.0, f"max={max_corrected_deg:.2f}deg")

    ckpt_path = _checkpoint_path()
    prev_checkpoint = None
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                prev_checkpoint = json.load(f)
        except Exception:
            prev_checkpoint = None

    checkpoint = {
        "raw_mean": raw_mean.tolist(),
        "session_offsets": session_offsets.tolist(),
        "k_values": k_values.tolist(),
        "decoded": decoded.tolist(),
        "residuals": residuals.tolist(),
        "ur_q": ur_q.tolist() if ur_q is not None else None,
        "timestamp": time.time(),
        "config_path": config_path,
    }

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n  Checkpoint saved to {ckpt_path}")
    print("  Re-run after reboot to verify cross-boot consistency.")

    if (
        prev_checkpoint is not None
        and prev_checkpoint.get("config_path") == config_path
        and prev_checkpoint.get("decoded") is not None
    ):
        prev_decoded = np.array(prev_checkpoint["decoded"], dtype=float)
        cross_boot_err = wrap_to_pi(decoded - prev_decoded)
        max_cross_deg = float(np.max(np.abs(np.rad2deg(cross_boot_err))))
        prev_k = prev_checkpoint.get("k_values", [])

        print("\n  Cross-checkpoint comparison:")
        print(f"    Previous k: {prev_k}")
        print(f"    Current k: {k_values.tolist()}")
        print(f"    Decoded diff (deg): {[f'{np.rad2deg(x):+.2f}' for x in cross_boot_err]}")

        _check("cross-session decoded diff < 10deg", max_cross_deg < 10.0, f"max={max_cross_deg:.2f}deg")
        if 5.0 <= max_cross_deg < 10.0:
            print(
                f"  {_YELLOW}WARNING: decoded diff exceeds tight 5deg target "
                f"(max={max_cross_deg:.2f}deg). Consider re-holding calibration pose more consistently.{_RESET}"
            )

    return _summary()


def run_simulated_multiboot_checks(config_path: str) -> int:
    _reset_counters()
    _section("3  Simulated Multi-Boot Reproducibility")

    cfg = _load_runtime_config(config_path)

    dyn_cfg = cfg.get("dynamixel", {})
    arm_cfg = cfg.get("arm_teleop", {})
    init_cfg = arm_cfg.get("initialization", {})
    teleop_cfg = cfg.get("teleop", {})
    mapping_cfg = teleop_cfg.get("mapping", {})

    num_arm = int(arm_cfg.get("num_arm_joints", 6))
    joint_signs = np.array(dyn_cfg.get("joint_signs", [1] * num_arm)[:num_arm], dtype=float)
    permanent_offsets = np.array(init_cfg.get("joint_offsets", [0.0] * num_arm)[:num_arm], dtype=float)
    calibration_pose = np.array(init_cfg.get("calibration_joint_pos", [0.0] * num_arm)[:num_arm], dtype=float)

    map_index, map_signs_cfg, map_offsets_cfg = _extract_mapping(mapping_cfg, num_arm)

    from gello.dynamixel.driver import DynamixelDriver

    port = _resolve_port(str(dyn_cfg.get("dynamixel_port", "/dev/ttyUSB0")))
    baudrate = int(dyn_cfg.get("baudrate", 4000000))
    servo_types = _extract_servo_types(dyn_cfg, num_arm)
    joint_ids = list(range(1, num_arm + 1))

    print("  Connecting to GELLO...")
    driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate, servo_types=servo_types)

    try:
        for _ in range(10):
            driver.get_joints()
        raw_now = np.mean(
            np.array([driver.get_joints()[:num_arm] for _ in range(25)], dtype=float),
            axis=0,
        )
    finally:
        driver.close()

    print(f"  Raw now (deg): {[f'{np.rad2deg(x):+.1f}' for x in raw_now]}")

    n_boots = 50
    rng = np.random.default_rng(12345)

    all_decoded = []
    all_follower = []

    print(f"\n  Simulating {n_boots} boots with random turn shifts...")

    for _ in range(n_boots):
        k_shift = rng.integers(-5, 6, size=num_arm)
        hold_var = rng.normal(0.0, np.deg2rad(1.0), size=num_arm)

        raw_sim = raw_now + k_shift * 2 * np.pi + hold_var

        session = np.zeros(num_arm, dtype=float)
        for i in range(num_arm):
            so, _, _ = disambiguate_leader_offset(
                raw_sim[i], permanent_offsets[i], joint_signs[i], calibration_pose[i]
            )
            session[i] = so

        decoded = decode_leader(raw_sim, session, joint_signs)
        leader_mapped = map_signs_cfg * decoded[map_index]
        follower = leader_mapped + map_offsets_cfg

        all_decoded.append(decoded)
        all_follower.append(follower)

    all_decoded_arr = np.array(all_decoded, dtype=float)
    all_follower_arr = np.array(all_follower, dtype=float)

    ref_decoded = all_decoded_arr[0]
    ref_follower = all_follower_arr[0]

    decoded_errs = np.array([wrap_to_pi(d - ref_decoded) for d in all_decoded_arr], dtype=float)
    follower_errs = np.array([wrap_to_pi(f - ref_follower) for f in all_follower_arr], dtype=float)

    max_decoded_err_deg = float(np.max(np.abs(np.rad2deg(decoded_errs))))
    max_follower_err_deg = float(np.max(np.abs(np.rad2deg(follower_errs))))
    mean_decoded_err_deg = float(np.mean(np.abs(np.rad2deg(decoded_errs))))
    std_decoded_err_deg = float(np.std(np.rad2deg(decoded_errs)))

    print(f"\n  Results over {n_boots} simulated boots:")
    print(f"    Decoded max deviation:  {max_decoded_err_deg:.2f}deg")
    print(f"    Decoded mean deviation: {mean_decoded_err_deg:.2f}deg")
    print(f"    Decoded std:            {std_decoded_err_deg:.2f}deg")
    print(f"    Follower max deviation: {max_follower_err_deg:.2f}deg")

    _check(
        f"decoded consistent across {n_boots} boots (max < 5deg)",
        max_decoded_err_deg < 5.0,
        f"max={max_decoded_err_deg:.2f}deg",
    )
    _check(
        f"follower consistent across {n_boots} boots (max < 5deg)",
        max_follower_err_deg < 5.0,
        f"max={max_follower_err_deg:.2f}deg",
    )
    _check("decoded std < 2deg (precision)", std_decoded_err_deg < 2.0, f"std={std_decoded_err_deg:.2f}deg")
    _check(
        "no 2pi blowups (max < 30deg)",
        max_decoded_err_deg < 30.0,
        "would be >100deg if turn disambiguation failed",
    )

    print("\n  Per-joint decoded deviation (deg):")
    print("   Joint      max     mean      std")
    print("   ---------------------------------")
    for j in range(num_arm):
        j_errs = np.rad2deg(decoded_errs[:, j])
        print(f"   J{j+1:>4} {np.max(np.abs(j_errs)):8.2f} {np.mean(np.abs(j_errs)):8.2f} {np.std(j_errs):8.2f}")

    return _summary()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turn-disambiguation verification script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Test levels (run in order):\n"
            "  math  Pure math tests, no hardware\n"
            "  read  Read-only hardware check (GELLO + optional UR5e)\n"
            "  sim   Simulated multi-boot reproducibility (reads GELLO once)\n"
        ),
    )
    parser.add_argument("level", choices=["math", "read", "sim", "all"], help="Test level to run")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/ur5e_gello_factr_hw_V2.yaml",
        help="YAML config path (for read/sim levels)",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("  TURN-DISAMBIGUATION VERIFICATION")
    print(f"  Level: {args.level}")
    print(f"{'=' * 60}")

    if args.level in ("read", "sim", "all") and not os.path.exists(args.config):
        print(f"{_RED}Config not found: {args.config}{_RESET}")
        return 1

    rc = 0
    if args.level in ("math", "all"):
        rc |= run_math_checks()
    if args.level in ("read", "all"):
        rc |= run_read_only_hardware_checks(args.config)
    if args.level in ("sim", "all"):
        rc |= run_simulated_multiboot_checks(args.config)

    return rc


if __name__ == "__main__":
    sys.exit(main())
