"""Visualize real leader joint angles on the leader URDF.

Goal
----
Quickly sanity-check whether the *real* leader joint angles (from Dynamixels) match
what the URDF expects (axes/signs/frames), by rendering the URDF and continuously
updating its configuration.

This uses Pinocchio for URDF loading + MeshCat for visualization.

Usage
-----
1) Install the optional visualizer dependency (once):
   pip install meshcat

2) Run:
   python scripts/visualize_leader_urdf_meshcat.py --config configs/ur5e_gello_factr_sim.yaml

Notes
-----
- This script is READ-ONLY with respect to the Dynamixels: it disables torque.
- It performs the same multi-turn offset calibration approach as
  gello/factr/gravity_compensation.py (coarse grid search around the calibration pose).
- If your URDF includes a gripper DOF (nq=7) while your arm has 6 joints,
  the script pads q with the live gripper position.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pinocchio as pin
import yaml

from gello.dynamixel.driver import DynamixelDriver


CALIBRATION_RANGE_MULTIPLIER = 10  # Range: -10π to 10π
CALIBRATION_STEP_COUNT = 721  # every ~5° over +/-10π


def _as_homogeneous(se3: Any) -> np.ndarray:
    """Convert a Pinocchio SE3 to a 4x4 homogeneous numpy array."""

    # Pinocchio SE3 supports .homogeneous in most versions.
    if hasattr(se3, "homogeneous"):
        H = se3.homogeneous
        return np.asarray(H, dtype=float)

    # Fallback: build from rotation + translation.
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(se3.rotation, dtype=float)
    H[:3, 3] = np.asarray(se3.translation, dtype=float).reshape(3)
    return H


def _setup_joint_frame_visuals(
    viz: Any,
    model: Any,
    num_arm_joints: int,
    axis_length: float = 0.06,
    axis_thickness: float = 0.004,
) -> list[int]:
    """Create XYZ triads for joint frames in MeshCat.

    Returns a list of Pinocchio joint ids that were visualized.
    """

    # MeshCat is an optional dependency; keep import local.
    import meshcat.geometry as g  # type: ignore
    import meshcat.transformations as tf  # type: ignore

    viewer = getattr(viz, "viewer", None)
    if viewer is None:
        raise RuntimeError("MeshCat viewer is not initialized")

    def _mat(rgb: tuple[float, float, float]) -> Any:
        return g.MeshLambertMaterial(color=int(rgb[0] * 255) * 65536 + int(rgb[1] * 255) * 256 + int(rgb[2] * 255))

    x_mat = _mat((1.0, 0.0, 0.0))
    y_mat = _mat((0.0, 1.0, 0.0))
    z_mat = _mat((0.0, 0.0, 1.0))

    # Boxes centered at +axis_length/2 along each axis.
    x_box = g.Box([axis_length, axis_thickness, axis_thickness])
    y_box = g.Box([axis_thickness, axis_length, axis_thickness])
    z_box = g.Box([axis_thickness, axis_thickness, axis_length])

    # Choose joints to visualize: only 1-DoF joints, and only the first num_arm_joints.
    joint_ids: list[int] = []
    dof_joints = [
        jid
        for jid in range(1, int(getattr(model, "njoints", 0)))
        if int(getattr(model.joints[jid], "nq", 0)) > 0
    ]
    joint_ids = dof_joints[:num_arm_joints]

    for jid in joint_ids:
        jname = str(model.names[jid])
        base_path = viewer["joint_frames"][jname]

        base_path["x"].set_object(x_box, x_mat)
        base_path["y"].set_object(y_box, y_mat)
        base_path["z"].set_object(z_box, z_mat)
        # Initialize transforms to identity; we'll update them each tick.
        I = tf.identity_matrix()
        base_path["x"].set_transform(I)
        base_path["y"].set_transform(I)
        base_path["z"].set_transform(I)

    return joint_ids


def _update_joint_frame_visuals(
    viz: Any,
    model: Any,
    data: Any,
    joint_ids: list[int],
    axis_length: float = 0.06,
) -> None:
    """Update the transforms of the joint frame triads in MeshCat."""

    import meshcat.transformations as tf  # type: ignore

    viewer = getattr(viz, "viewer", None)
    if viewer is None:
        return

    # Offsets: translate half length along axis.
    Tx = tf.translation_matrix([axis_length / 2.0, 0.0, 0.0])
    Ty = tf.translation_matrix([0.0, axis_length / 2.0, 0.0])
    Tz = tf.translation_matrix([0.0, 0.0, axis_length / 2.0])

    for jid in joint_ids:
        jname = str(model.names[jid])
        base_path = viewer["joint_frames"][jname]

        H = _as_homogeneous(data.oMi[jid])
        base_path["x"].set_transform(H @ Tx)
        base_path["y"].set_transform(H @ Ty)
        base_path["z"].set_transform(H @ Tz)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict")
    return cfg


def _resolve_urdf_path(config_path: str, urdf_filename: str) -> Path:
    cfg_dir = Path(config_path).expanduser().resolve().parent
    urdf_path = (cfg_dir / urdf_filename).expanduser().resolve()
    if urdf_path.exists():
        return urdf_path

    # Fall back to being relative to repo root (common when configs live at repo root)
    repo_root = Path(__file__).parent.parent
    candidate = (repo_root / urdf_filename).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"URDF not found. Tried: {urdf_path} and {candidate}. (leader_urdf={urdf_filename})"
    )


def _resolve_dynamixel_port(port_config: str) -> str:
    # Matches the logic used elsewhere in the repo: allow absolute /dev/... or
    # /dev/serial/by-id/<name>.
    if port_config.startswith("/"):
        return port_config
    return "/dev/serial/by-id/" + port_config


def _calibrate_offsets(
    driver: DynamixelDriver,
    joint_signs: np.ndarray,
    calibration_joint_pos: np.ndarray,
    num_arm_joints: int,
) -> np.ndarray:
    """Find multi-turn offsets so that signed (raw-offset) matches calibration pose."""

    # Warm up reads
    for _ in range(10):
        driver.get_positions_and_velocities()
        time.sleep(0.01)

    raw_pos, _ = driver.get_positions_and_velocities()
    if len(raw_pos) < num_arm_joints:
        raise RuntimeError(
            f"Expected at least {num_arm_joints} motors, got {len(raw_pos)}"
        )

    offsets: list[float] = []

    # Precompute candidate offsets (same grid for all joints)
    offsets_grid = np.linspace(
        -CALIBRATION_RANGE_MULTIPLIER * np.pi,
        CALIBRATION_RANGE_MULTIPLIER * np.pi,
        CALIBRATION_STEP_COUNT,
        dtype=float,
    )

    for i in range(num_arm_joints):
        sign = float(joint_signs[i])
        raw = float(raw_pos[i])
        target = float(calibration_joint_pos[i])

        # signed_joint = sign * (raw - offset)  => error = signed_joint - target
        signed_joint_grid = sign * (raw - offsets_grid)
        err = signed_joint_grid - target
        best_idx = int(np.argmin(np.abs(err)))
        offsets.append(float(offsets_grid[best_idx]))

    # Any remaining joints (e.g. gripper) use current position as offset
    for j in range(num_arm_joints, len(raw_pos)):
        offsets.append(float(raw_pos[j]))

    return np.asarray(offsets, dtype=float)


def _read_leader_state(
    driver: DynamixelDriver,
    offsets: np.ndarray,
    joint_signs: np.ndarray,
    num_arm_joints: int,
    dt: float,
    last_gripper_pos: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pos, vel = driver.get_positions_and_velocities()

    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)

    arm_pos = (pos[:num_arm_joints] - offsets[:num_arm_joints]) * joint_signs[:num_arm_joints]
    arm_vel = vel[:num_arm_joints] * joint_signs[:num_arm_joints]

    gripper_raw = float(pos[-1])
    gripper_pos = (float(pos[-1]) - float(offsets[-1])) * float(joint_signs[-1])
    gripper_vel = (gripper_pos - last_gripper_pos) / max(dt, 1e-6)

    return arm_pos, arm_vel, gripper_pos, gripper_vel


def _make_pinocchio_models(urdf_path: Path) -> Tuple[Any, Any, Any]:
    # Pinocchio expects package_dirs as a directory containing meshes (or packages).
    package_dir = str(urdf_path.parent)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(  # type: ignore[attr-defined]
        filename=str(urdf_path),
        package_dirs=package_dir,
    )
    return model, collision_model, visual_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize real leader joint angles on the URDF using MeshCat"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config (e.g. configs/ur5e_gello_factr_sim.yaml)",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=60.0,
        help="Viewer update rate (Hz)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help=(
            "Override the URDF path from the config (useful to try e.g. "
            "gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4_noGripper.urdf)"
        ),
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Do not connect to Dynamixels; instead animate joints with a sine wave.",
    )
    parser.add_argument(
        "--show-joint-frames",
        action="store_true",
        help="Visualize Pinocchio joint frames as XYZ triads in MeshCat.",
    )
    args = parser.parse_args()

    cfg_path = str(Path(args.config).expanduser().resolve())
    cfg = _load_yaml(cfg_path)

    name = cfg.get("name", "<unknown>")
    print(f"Loaded config: {name}")
    print(f"Config file: {cfg_path}")

    dt = 1.0 / float(cfg["controller"]["frequency"])
    num_arm_joints = int(cfg["arm_teleop"]["num_arm_joints"])

    calib_pose = np.asarray(
        cfg["arm_teleop"]["initialization"]["calibration_joint_pos"], dtype=float
    )

    joint_signs = np.asarray(cfg["dynamixel"]["joint_signs"], dtype=float)

    # Resolve URDF
    urdf_filename = str(args.urdf) if args.urdf is not None else str(cfg["arm_teleop"]["leader_urdf"])
    urdf_path = _resolve_urdf_path(cfg_path, urdf_filename)
    print(f"URDF: {urdf_path}")

    # Build Pinocchio models
    model, collision_model, visual_model = _make_pinocchio_models(urdf_path)
    nq = int(getattr(model, "nq", 0))
    nv = int(getattr(model, "nv", 0))
    if nq <= 0 or nv <= 0:
        raise RuntimeError("Pinocchio model has invalid nq/nv")

    # Optional viewer
    try:
        from pinocchio.visualize import MeshcatVisualizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "MeshCat visualizer is not available. Install it via: pip install meshcat\n"
            f"(Import error: {e})"
        ) from e

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Optional joint-frame visuals (helps debug joint origin placement vs mesh origin).
    data = model.createData()
    joint_frame_ids: list[int] = []
    if args.show_joint_frames:
        joint_frame_ids = _setup_joint_frame_visuals(viz, model, num_arm_joints)
        joint_names = [str(model.names[jid]) for jid in joint_frame_ids]
        print(
            "Joint-frame visuals enabled. "
            "In MeshCat, expand the object tree under: joint_frames/<joint_name>. "
            f"Joints: {joint_names}"
        )

    if args.no_hardware:
        print("Running in --no-hardware mode (sine wave animation).")
        t0 = time.time()
        while True:
            t = time.time() - t0
            q = pin.neutral(model)
            # Animate only first num_arm_joints
            for i in range(min(num_arm_joints, len(q))):
                q[i] = 0.5 * np.sin(0.5 * t + 0.3 * i)
            viz.display(q)

            if joint_frame_ids:
                pin.forwardKinematics(model, data, q)
                _update_joint_frame_visuals(viz, model, data, joint_frame_ids)

            time.sleep(max(0.0, (1.0 / args.rate_hz)))

    # Hardware mode
    dyn_cfg = cfg["dynamixel"]
    port = _resolve_dynamixel_port(str(dyn_cfg["dynamixel_port"]))
    servo_types = list(dyn_cfg["servo_types"])
    baudrate = int(dyn_cfg.get("baudrate", 57600))

    joint_ids = (np.arange(len(servo_types)) + 1).tolist()

    print(f"Connecting to Dynamixels on {port} @ {baudrate}...")
    driver = DynamixelDriver(joint_ids, servo_types, port, baudrate=baudrate)

    # Ensure we do not drive the motors
    try:
        driver.set_torque_mode(False)
    except Exception:
        pass

    print("Calibrating offsets (read-only)...")
    offsets = _calibrate_offsets(driver, joint_signs, calib_pose, num_arm_joints)
    print(f"Offsets calibrated (rad): {[float(f'{x:.4f}') for x in offsets]}")

    last_gripper_pos = 0.0
    last_print = 0.0

    print("Streaming leader state into MeshCat. Ctrl+C to stop.")
    try:
        while True:
            loop_start = time.time()

            arm_pos, arm_vel, gripper_pos, gripper_vel = _read_leader_state(
                driver,
                offsets,
                joint_signs,
                num_arm_joints,
                dt,
                last_gripper_pos,
            )
            last_gripper_pos = float(gripper_pos)

            # Build q sized to URDF
            q = np.zeros((nq,), dtype=float)
            q[: min(num_arm_joints, nq)] = arm_pos[: min(num_arm_joints, nq)]
            if nq == num_arm_joints + 1:
                q[num_arm_joints] = float(gripper_pos)

            viz.display(q)

            if joint_frame_ids:
                pin.forwardKinematics(model, data, q)
                _update_joint_frame_visuals(viz, model, data, joint_frame_ids)

            # Periodic console debug
            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                print(
                    "q(deg): "
                    + str([float(f"{x:.1f}") for x in np.rad2deg(arm_pos)])
                    + f" | gripper(deg)={float(np.rad2deg(gripper_pos)):.1f}"
                )

            sleep = (1.0 / args.rate_hz) - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            driver.set_torque_mode(False)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
