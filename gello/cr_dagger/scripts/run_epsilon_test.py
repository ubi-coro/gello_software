from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from threading import Thread
import threading
from typing import Any, Optional

import numpy as np

from gello.bilat_4ch.gello_ur5e_observer_shi import (
    MinimalistEstimatorConfig,
    MinimalistTorqueEstimator,
    MotorParams,
    MotorType,
)
from gello.cr_dagger.core.epsilon_recorder import EpsilonRecorder
from gello.cr_dagger.ipc.shared_observation_snapshot import SharedObservationSnapshot
from gello.cr_dagger.ipc.shared_trajectory_buffer import SharedTrajectoryBuffer
from gello.cr_dagger.policy.filtered_ddq import FilteredDDQComputer
from gello.cr_dagger.policy.policy_worker import policy_worker
from gello.cr_dagger.policy.trajectory_interpolator import TrajectoryInterpolator
from gello.factr.gravity_compensation import FACTRGravityCompensation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_leader_urdf(config_path: Path, leader_urdf: str) -> Path:
    for candidate in [
        (config_path.parent / leader_urdf).resolve(),
        (REPO_ROOT / "gello" / "factr" / "urdf" / Path(leader_urdf).name).resolve(),
        (REPO_ROOT / leader_urdf).resolve(),
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"URDF not found for leader_urdf='{leader_urdf}'")


def _infer_motor_params(servo_types: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    params_by_servo = {
        "XC330_T288_T": (288.35, 1.136 / 288.35),
        "XM430_W210_T": (212.6, 1.304 / 212.6),
        "XM430_W350_T": (353.5, 1.783 / 353.5),
    }
    gear_ratios, kts = [], []
    for s in servo_types[:n]:
        gr, kt = params_by_servo.get(s, (1.0, 0.00504))
        gear_ratios.append(float(gr))
        kts.append(float(kt))
    while len(gear_ratios) < n:
        gear_ratios.append(1.0)
        kts.append(0.00504)
    return np.asarray(gear_ratios, dtype=float), np.asarray(kts, dtype=float)


def _build_shi(
    system: FACTRGravityCompensation, config_path: Path, n: int
) -> MinimalistTorqueEstimator:
    leader_urdf = str(system.config["arm_teleop"]["leader_urdf"])
    urdf_path = _resolve_leader_urdf(config_path, leader_urdf)
    servo_types = list(system.config["dynamixel"]["servo_types"])
    gear_ratio, kt = _infer_motor_params(servo_types, n)

    return MinimalistTorqueEstimator(
        MinimalistEstimatorConfig(
            urdf_path=str(urdf_path),
            motor_params=MotorParams(
                kt=kt,
                gear_ratio=gear_ratio,
                eta=np.full(n, 0.65, dtype=float),
                motor_type=MotorType.CURRENT,
            ),
            alpha_ema=0.5,
            vel_threshold=0.05,
        )
    )


def _parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _expand_list(values: list[float], n: int, fill: float = 0.0) -> np.ndarray:
    if not values:
        return np.full(n, fill, dtype=float)
    if len(values) == 1:
        return np.full(n, values[0], dtype=float)
    if len(values) < n:
        return np.asarray(values + [fill] * (n - len(values)), dtype=float)
    return np.asarray(values[:n], dtype=float)


class ContactGate:
    """Lightweight online gate for contact hypotheses from residual torque."""

    NO_CONTACT = 0.0
    POSSIBLE_CONTACT = 1.0
    CONFIRMED_CONTACT = 2.0
    CORRECTION_ACTIVE = 3.0

    def __init__(
        self,
        n_joints: int,
        residual_low: float = 0.15,
        residual_high: float = 0.45,
        accel_gate: float = 8.0,
        window: int = 8,
    ):
        self.residual_low = float(residual_low)
        self.residual_high = max(float(residual_high), float(residual_low) + 1e-6)
        self.accel_gate = max(float(accel_gate), 1e-6)
        self.sign_hist = np.zeros((max(int(window), 1), int(n_joints)), dtype=float)
        self.idx = 0
        self.count = 0

    def update(
        self,
        tau_residual: np.ndarray,
        ddq_ref: np.ndarray,
        correction_active: bool = False,
    ) -> tuple[float, float]:
        residual = np.asarray(tau_residual, dtype=float)
        self.sign_hist[self.idx] = np.sign(residual)
        self.idx = (self.idx + 1) % self.sign_hist.shape[0]
        self.count = min(self.count + 1, self.sign_hist.shape[0])

        mag = float(np.linalg.norm(residual))
        mag_prob = np.clip(
            (mag - self.residual_low) / (self.residual_high - self.residual_low),
            0.0,
            1.0,
        )
        ddq_norm = float(np.linalg.norm(ddq_ref))
        motion_gate = 1.0 / np.sqrt(1.0 + (ddq_norm / self.accel_gate) ** 2)

        hist = self.sign_hist[: self.count]
        if self.count < 2 or mag <= self.residual_low:
            consistency = 0.0
        else:
            consistency = float(np.max(np.abs(np.sum(hist, axis=0))) / self.count)

        probability = float(np.clip(mag_prob * motion_gate * consistency, 0.0, 1.0))
        if correction_active and probability >= 0.6:
            state = self.CORRECTION_ACTIVE
        elif probability >= 0.6:
            state = self.CONFIRMED_CONTACT
        elif probability >= 0.25:
            state = self.POSSIBLE_CONTACT
        else:
            state = self.NO_CONTACT
        return probability, state


def _shi_torque_components(
    system: FACTRGravityCompensation,
    shi: Optional[MinimalistTorqueEstimator],
    q: np.ndarray,
    dq: np.ndarray,
    currents_arm: Optional[np.ndarray],
    tau_model: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(system.num_arm_joints)
    tau_ext = np.zeros(n, dtype=float)
    tau_meas = None

    if shi is not None and currents_arm is not None:
        try:
            tau_meas = shi.get_raw_motor_torque(dq, currents_arm)
            tau_ext = shi.update(q, dq, currents_arm)
        except Exception:
            tau_ext = np.zeros(n, dtype=float)
            tau_meas = None

    components = system.separate_contact_torque_components(
        tau_model=tau_model,
        tau_ext_shi=tau_ext,
        tau_meas=tau_meas,
    )
    return tau_ext, components


def _start_prepared_teleop(system: FACTRGravityCompensation) -> bool:
    if not system.teleop_enabled or not system.teleop_prepared:
        return False

    if system.teleop_thread is not None and system.teleop_thread.is_alive():
        return True

    system.running = True
    if getattr(system, "use_impedance_control", False):
        target = system._teleop_loop_impedance
        mode = "impedance"
    else:
        target = system._teleop_loop
        mode = "position"

    system.teleop_thread = Thread(target=target, daemon=True, name="factr-teleop")
    system.teleop_thread.start()
    print(f"[TELEOP] started follower mirroring thread ({mode} mode)")
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Epsilon tracking test (Phase B style)")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V3_PhaseB.yaml")
    p.add_argument("--duration", type=float, default=30.0, help="0 means run until Ctrl+C")
    p.add_argument("--settle-time", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=120.0)

    p.add_argument("--target", choices=["leader", "follower"], default="leader")
    p.add_argument(
        "--mode",
        choices=[
            "leader_tracking",
            "leader_admittance",
            "four_channel",
            "follower_tracking",
            "observer_only",
        ],
        default=None,
        help="Architecture mode. If omitted, legacy --target/--four-channel flags are used.",
    )
    p.add_argument(
        "--test-mode",
        choices=["static_hold", "sine", "chirp", "ramp", "multi_joint", "taskspace"],
        default="sine",
    )

    p.add_argument(
        "--four-channel",
        action="store_true",
        help="Use 4-channel dual-impedance loop (epsilon = q_ur5e - q_cmd).",
    )
    p.add_argument(
        "--mirror-source",
        choices=["cmd", "actual", "policy", "blend"],
        default=None,
        help="Override 4-channel mirror source (default: config).",
    )
    p.add_argument("--mirror-blend-cmd", type=float, default=0.5)
    p.add_argument("--mirror-blend-actual", type=float, default=0.3)
    p.add_argument("--mirror-blend-policy", type=float, default=0.2)

    p.add_argument("--compensation", choices=["none", "velocity_ff", "feedforward", "computed_torque"], default="none")
    p.add_argument("--kp", type=float, default=5.0)
    p.add_argument("--kd", type=float, default=0.5)

    p.add_argument("--adm-mass", type=float, default=1.0)
    p.add_argument("--adm-damp", type=float, default=8.0)
    p.add_argument("--adm-stiff", type=float, default=25.0)
    p.add_argument("--adm-leak", type=float, default=0.02)
    p.add_argument("--adm-delta-max", type=float, default=0.25)

    p.add_argument("--follower-kp", type=float, default=150.0)
    p.add_argument("--follower-kd", type=float, default=12.0)

    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--frequency", type=float, default=0.2)
    p.add_argument("--joint-index", type=int, default=1)

    p.add_argument("--chirp-f-start", type=float, default=0.1)
    p.add_argument("--chirp-f-end", type=float, default=2.0)
    p.add_argument("--chirp-duration", type=float, default=30.0)

    p.add_argument("--ramp-delta", type=float, default=0.2)
    p.add_argument("--ramp-duration", type=float, default=5.0)

    p.add_argument("--multi-amplitudes", type=str, default=None)
    p.add_argument("--multi-frequencies", type=str, default=None)

    p.add_argument("--taskspace-motion", choices=["line", "circle", "figure8"], default="circle")
    p.add_argument("--taskspace-amplitude", type=float, default=0.08, help="meters")
    p.add_argument("--taskspace-axis", choices=["xy", "xz", "yz"], default="xy")

    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-dt", type=float, default=0.1)

    p.add_argument("--log-dir", type=str, default="epsilon_measurements")
    p.add_argument("--no-follower", action="store_true", help="Leader mode: do not start follower teleop")
    p.add_argument("--no-shi", action="store_true", help="Skip Shi torque estimator (records zeros)")

    return p.parse_args()


def _resolve_arch_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        if args.mode == "observer_only":
            print(
                "[WARN] observer_only is currently an alias for follower_tracking; "
                "no sensor-only correction path is enabled yet"
            )
            return "follower_tracking"
        return str(args.mode)
    if args.four_channel:
        return "four_channel"
    if args.target == "follower":
        return "follower_tracking"
    return "leader_tracking"


def _build_policy(
    args: argparse.Namespace,
    n: int,
    q0: np.ndarray,
    config_path: Path,
    system_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if args.test_mode == "static_hold":
        return "dummy_hold", {"q_hold": q0.tolist()}

    if args.test_mode == "sine":
        amplitudes = np.zeros(n, dtype=float)
        frequencies = np.zeros(n, dtype=float)
        if 0 <= int(args.joint_index) < n:
            amplitudes[int(args.joint_index)] = float(args.amplitude)
            frequencies[int(args.joint_index)] = float(args.frequency)
        else:
            amplitudes[:] = float(args.amplitude)
            frequencies[:] = float(args.frequency)
        return "dummy_sine", {
            "center": q0.tolist(),
            "amplitude": amplitudes.tolist(),
            "frequency": frequencies.tolist(),
        }

    if args.test_mode == "chirp":
        return "dummy_chirp", {
            "center": q0.tolist(),
            "amplitude": float(args.amplitude),
            "f_start": float(args.chirp_f_start),
            "f_end": float(args.chirp_f_end),
            "duration": float(args.chirp_duration),
            "joint_index": int(args.joint_index),
        }

    if args.test_mode == "ramp":
        q_end = q0.copy()
        if 0 <= int(args.joint_index) < n:
            q_end[int(args.joint_index)] += float(args.ramp_delta)
        return "dummy_ramp", {
            "q_start": q0.tolist(),
            "q_end": q_end.tolist(),
            "duration": float(args.ramp_duration),
        }

    if args.test_mode == "multi_joint":
        amplitudes = _expand_list(_parse_float_list(args.multi_amplitudes), n, 0.0)
        frequencies = _expand_list(_parse_float_list(args.multi_frequencies), n, 0.2)
        return "dummy_multi_sine", {
            "center": q0.tolist(),
            "amplitudes": amplitudes.tolist(),
            "frequencies": frequencies.tolist(),
        }

    if args.test_mode == "taskspace":
        leader_urdf = str(system_config["arm_teleop"]["leader_urdf"])
        urdf_path = _resolve_leader_urdf(config_path, leader_urdf)
        return "dummy_taskspace", {
            "center": q0.tolist(),
            "urdf_path": str(urdf_path),
            "motion_type": str(args.taskspace_motion),
            "amplitude": float(args.taskspace_amplitude),
            "frequency": float(args.frequency),
            "axis": str(args.taskspace_axis),
        }

    raise ValueError(f"Unknown test_mode: {args.test_mode}")


def _sleep_remaining(dt: float, t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    sleep_s = dt - elapsed
    if sleep_s > 0:
        time.sleep(sleep_s)


def _leader_gravity_hold(system: FACTRGravityCompensation, dt: float) -> None:
    q, dq, _, _ = system.get_leader_joint_states()
    tau_grav = system.gravity_compensation(q, dq)
    tau_fric = system.friction_compensation(dq)
    tau_damp = -float(system.gravity_comp_velocity_damping) * dq
    system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)
    time.sleep(max(0.0, dt - 0.0005))


def _run_leader_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
    shi: Optional[MinimalistTorqueEstimator],
) -> None:
    n = system.num_arm_joints
    dt = float(system.dt)

    if not args.no_follower:
        if not _start_prepared_teleop(system):
            print("[WARN] teleop could not be started; continuing without follower")

    print("[SETTLE] gravity comp warmup")
    t_settle = time.perf_counter()
    while time.perf_counter() - t_settle < float(args.settle_time):
        _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0
    contact_gate = ContactGate(n)

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        currents_arm = None
        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, cur_raw = system.driver.get_positions_velocities_and_currents()
            q = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
            dq = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

            if len(pos_raw) > n:
                grip_raw = float(pos_raw[-1])
                grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
                system.leader_gripper_raw_rad = grip_raw
            else:
                grip = 0.0
                grip_raw = 0.0

            if len(vel_raw) > n:
                grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
            else:
                grip_vel = 0.0

            system.gripper_pos = float(grip)
            system._last_gripper_vel = float(grip_vel)

            if shi is not None:
                currents_arm = np.asarray(cur_raw[:n], dtype=float) * system.joint_signs[:n]
        else:
            q, dq, grip, grip_vel = system.get_leader_joint_states()
            if shi is not None and system.driver is not None:
                currents = system.driver.get_currents()
                currents_arm = np.asarray(currents[:n], dtype=float) * system.joint_signs[:n]

        q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref, t_mono)

        tau_cmd = system.control_loop_step_with_policy(
            q_ref_policy=q_ref,
            dq_ref_policy=dq_ref,
            ddq_ref_policy=ddq_ref,
            compensation_mode=str(args.compensation),
            leader_state=(q, dq, float(grip), float(grip_vel)),
        )

        tau_ext, tau_components = _shi_torque_components(
            system=system,
            shi=shi,
            q=q,
            dq=dq,
            currents_arm=currents_arm,
            tau_model=tau_cmd,
        )
        contact_probability, contact_state = contact_gate.update(
            tau_components["tau_residual"],
            ddq_ref,
        )

        q_follower = np.zeros(n)
        dq_follower = np.zeros(n)
        if system.teleop_enabled:
            try:
                q_follower, dq_follower = system.get_follower_arm_state()
            except Exception:
                pass

        epsilon = q - q_ref
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_ref=ddq_ref,
            q_leader=q,
            dq_leader=dq,
            q_cmd_leader=q_ref,
            q_c_leader=q_ref,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=np.full(n, np.nan),
            delta_corr=np.zeros(n),
            tau_cmd=tau_cmd,
            tau_ext_shi=tau_ext,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=contact_probability,
            contact_state=contact_state,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q,
            dq=dq,
            grip=float(grip),
            tau_ext=tau_ext,
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)


def _run_follower_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
) -> None:
    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("Follower mode requires teleop.enable=true with direct RTDE")

    follower = system._direct_follower_robot
    n = system.num_arm_joints
    dt = float(system.dt)

    print("[SETTLE] gravity comp warmup")
    t_settle = time.perf_counter()
    while time.perf_counter() - t_settle < float(args.settle_time):
        _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0

    tau_max = np.array([100.0, 100.0, 60.0, 25.0, 25.0, 25.0], dtype=float)

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        q_leader, dq_leader, grip, grip_vel = system.get_leader_joint_states()

        tau_grav = system.gravity_compensation(q_leader, dq_leader)
        tau_fric = system.friction_compensation(dq_leader)
        tau_limit, torque_gripper = system.joint_limit_barrier(
            q_leader, dq_leader, float(grip), float(grip_vel)
        )
        tau_damp = -float(system.gravity_comp_velocity_damping) * dq_leader
        system.set_leader_joint_torque(tau_grav + tau_fric + tau_limit + tau_damp, float(torque_gripper))

        q_ref_leader, dq_ref_leader, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref_leader, t_mono)

        q_ref_follower = system._build_follower_action(q_ref_leader, float(grip))[:n]
        if system.map_signs is not None and system.map_index is not None:
            dq_ref_follower = system.map_signs * dq_ref_leader[system.map_index]
        else:
            dq_ref_follower = dq_ref_leader[:n]

        q_follower, dq_follower = system.get_follower_arm_state()
        q_follower = np.asarray(q_follower[:n], dtype=float)
        dq_follower = np.asarray(dq_follower[:n], dtype=float)

        follower.command_joint_state_impedance(
            target_joints=q_ref_follower,
            target_velocities=dq_ref_follower,
            kp=float(args.follower_kp),
            kd=float(args.follower_kd),
        )

        tau_cmd = float(args.follower_kp) * (q_ref_follower - q_follower) + float(
            args.follower_kd
        ) * (dq_ref_follower - dq_follower)
        tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
        tau_components = system.separate_contact_torque_components(
            tau_model=tau_cmd,
            tau_ext_shi=np.zeros(n),
        )

        epsilon = q_follower - q_ref_follower
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref_follower,
            dq_ref=dq_ref_follower,
            ddq_ref=ddq_ref,
            q_leader=q_leader,
            dq_leader=dq_leader,
            q_cmd_leader=q_ref_leader,
            q_c_leader=q_ref_leader,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=q_ref_follower,
            delta_corr=np.zeros(n),
            tau_cmd=tau_cmd,
            tau_ext_shi=np.zeros(n),
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=0.0,
            contact_state=ContactGate.NO_CONTACT,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q_leader,
            dq=dq_leader,
            grip=float(grip),
            tau_ext=np.zeros(n),
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)

    try:
        follower.robot.stopJ(2.0)
    except Exception:
        pass


def _run_leader_admittance_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
    shi: Optional[MinimalistTorqueEstimator],
) -> None:
    n = system.num_arm_joints
    dt = float(system.dt)

    if not args.no_follower:
        if not _start_prepared_teleop(system):
            print("[WARN] teleop could not be started; continuing without follower")

    print("[SETTLE] gravity comp warmup")
    t_settle = time.perf_counter()
    while time.perf_counter() - t_settle < float(args.settle_time):
        _leader_gravity_hold(system, dt)

    q0, _, _, _ = system.get_leader_joint_states()
    q_c = np.asarray(q0[:n], dtype=float).copy()
    dq_c = np.zeros(n, dtype=float)
    contact_gate = ContactGate(n)
    last_tau_residual = np.zeros(n, dtype=float)

    mass = np.full(n, max(float(args.adm_mass), 1e-6), dtype=float)
    damp = np.full(n, max(float(args.adm_damp), 0.0), dtype=float)
    stiff = np.full(n, max(float(args.adm_stiff), 0.0), dtype=float)
    leak = max(float(args.adm_leak), 0.0)
    delta_max = max(float(args.adm_delta_max), 1e-6)

    t_start = time.monotonic()
    last_warn = 0.0

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        currents_arm = None
        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, cur_raw = system.driver.get_positions_velocities_and_currents()
            q = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
            dq = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

            if len(pos_raw) > n:
                grip_raw = float(pos_raw[-1])
                grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
                system.leader_gripper_raw_rad = grip_raw
            else:
                grip = 0.0
                grip_raw = 0.0

            if len(vel_raw) > n:
                grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
            else:
                grip_vel = 0.0

            system.gripper_pos = float(grip)
            system._last_gripper_vel = float(grip_vel)

            if shi is not None:
                currents_arm = np.asarray(cur_raw[:n], dtype=float) * system.joint_signs[:n]
        else:
            q, dq, grip, grip_vel = system.get_leader_joint_states()
            if shi is not None and system.driver is not None:
                currents = system.driver.get_currents()
                currents_arm = np.asarray(currents[:n], dtype=float) * system.joint_signs[:n]

        q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
        ddq_ref = ddq_computer.update(dq_ref, t_mono)

        contact_probability, contact_state = contact_gate.update(
            last_tau_residual,
            ddq_ref,
            correction_active=True,
        )

        tau_contact = last_tau_residual if contact_probability >= 0.6 else np.zeros(n)
        ddq_c = (tau_contact - damp * dq_c - stiff * (q_c - q_ref)) / mass
        dq_c = (1.0 - leak * dt) * (dq_c + ddq_c * dt)
        q_c = q_c + dq_c * dt

        delta = np.clip(q_c - q_ref, -delta_max, delta_max)
        q_c = q_ref + delta

        tau_cmd = system.control_loop_step_with_policy(
            q_ref_policy=q_c,
            dq_ref_policy=dq_c,
            ddq_ref_policy=ddq_c,
            compensation_mode=str(args.compensation),
            leader_state=(q, dq, float(grip), float(grip_vel)),
        )
        tau_ext, tau_components = _shi_torque_components(
            system=system,
            shi=shi,
            q=q,
            dq=dq,
            currents_arm=currents_arm,
            tau_model=tau_cmd,
        )
        last_tau_residual = tau_components["tau_residual"].copy()

        q_follower = np.zeros(n)
        dq_follower = np.zeros(n)
        if system.teleop_enabled:
            try:
                q_follower, dq_follower = system.get_follower_arm_state()
            except Exception:
                pass

        epsilon = q - q_c
        recorder.record(
            t_mono=t_mono,
            q_ref=q_ref,
            dq_ref=dq_ref,
            ddq_ref=ddq_ref,
            q_leader=q,
            dq_leader=dq,
            q_cmd_leader=q_c,
            q_c_leader=q_c,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=np.full(n, np.nan),
            delta_corr=delta,
            tau_cmd=tau_cmd,
            tau_ext_shi=tau_ext,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=contact_probability,
            contact_state=contact_state,
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q,
            dq=dq,
            grip=float(grip),
            tau_ext=tau_components["tau_residual"],
            wrench=np.zeros(6),
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)


def _run_four_channel_mode(
    system: FACTRGravityCompensation,
    args: argparse.Namespace,
    traj_interp: TrajectoryInterpolator,
    recorder: EpsilonRecorder,
    ddq_computer: FilteredDDQComputer,
    obs_snap: SharedObservationSnapshot,
) -> None:
    from gello.cr_dagger.core.yamane_observer import GeneralizedMomentumObserver

    if not system.teleop_enabled or system._direct_follower_robot is None:
        raise RuntimeError("4-channel mode requires teleop.enable=true with direct RTDE")

    follower = system._direct_follower_robot
    n = system.num_arm_joints
    dt = 1.0 / 500.0

    four_cfg = system.config.get("teleop", {}).get("four_channel", {})
    K_virtual = float(four_cfg.get("K_virtual", 40.0))
    correction_gain = float(four_cfg.get("correction_gain", 1.0))
    force_feedback_gain = float(four_cfg.get("force_feedback_gain", 0.3))
    velocity_gate_threshold = float(four_cfg.get("velocity_gate_threshold", 1.5))
    deadband_tau = float(four_cfg.get("deadband_tau", 0.15))
    mirror_source = str(four_cfg.get("mirror_source", "cmd")).lower()
    yamane_K_obs = float(four_cfg.get("yamane_K_obs", 50.0))
    yamane_filter_fc = float(four_cfg.get("yamane_filter_fc", 5.0))
    if args.mirror_source:
        mirror_source = str(args.mirror_source)

    kp_mirror = float(system.policy_impedance_kp)
    kd_mirror = float(system.policy_impedance_kd)

    yamane = GeneralizedMomentumObserver(
        pin_model=system.pin_model,
        num_arm_joints=n,
        K_obs=yamane_K_obs,
        dt=dt,
        filter_fc=yamane_filter_fc,
    )

    has_gripper = bool(
        hasattr(follower, "gripper")
        and hasattr(follower, "_use_gripper")
        and follower._use_gripper
    )
    gripper_lock = threading.Lock()
    gripper_state = {"latest": None}
    if has_gripper:
        def gripper_worker() -> None:
            while True:
                pos = None
                with gripper_lock:
                    if gripper_state["latest"] is not None:
                        pos = gripper_state["latest"]
                        gripper_state["latest"] = None
                if pos is not None:
                    try:
                        follower.gripper.move(pos, 255, 10)
                    except Exception:
                        pass
                time.sleep(1.0 / 30.0)

        Thread(target=gripper_worker, daemon=True, name="epsilon-4ch-gripper").start()

    print("[SETTLE] gravity comp warmup")
    t_settle = time.perf_counter()
    while time.perf_counter() - t_settle < float(args.settle_time):
        _leader_gravity_hold(system, dt)

    t_start = time.monotonic()
    last_warn = 0.0
    q_mirror_target = np.zeros(n, dtype=float)
    dq_mirror_target = np.zeros(n, dtype=float)
    delta_follower = np.zeros(n, dtype=float)
    initialized = False

    while True:
        loop_t0 = time.perf_counter()
        t_mono = time.monotonic()

        if float(args.duration) > 0 and (t_mono - t_start) > float(args.duration):
            break

        if system.driver is not None and hasattr(system.driver, "get_positions_velocities_and_currents"):
            pos_raw, vel_raw, _ = system.driver.get_positions_velocities_and_currents()
        else:
            pos_raw, vel_raw = system.driver.get_positions_and_velocities()

        q_leader = (np.asarray(pos_raw[:n], dtype=float) - system.joint_offsets[:n]) * system.joint_signs[:n]
        dq_leader = np.asarray(vel_raw[:n], dtype=float) * system.joint_signs[:n]

        if len(pos_raw) > n:
            grip_raw = float(pos_raw[-1])
            grip = (grip_raw - float(system.joint_offsets[-1])) * float(system.joint_signs[-1])
            system.leader_gripper_raw_rad = grip_raw
        else:
            grip_raw = 0.0
            grip = 0.0

        if len(vel_raw) > n:
            grip_vel = float(vel_raw[-1]) * float(system.joint_signs[-1])
        else:
            grip_vel = 0.0

        system.gripper_pos = float(grip)
        system._last_gripper_vel = float(grip_vel)

        if not initialized:
            yamane.reset(q_leader, dq_leader)
            q_mirror_target = q_leader.copy()
            dq_mirror_target = np.zeros(n, dtype=float)
            initialized = True

        q_ref_leader, dq_ref_leader, is_stale = traj_interp.get_reference(t_mono)
        q_ref_leader = np.asarray(q_ref_leader[:n], dtype=float)
        dq_ref_leader = (
            np.asarray(dq_ref_leader[:n], dtype=float)
            if dq_ref_leader is not None
            else np.zeros(n, dtype=float)
        )
        ddq_ref = ddq_computer.update(dq_ref_leader, t_mono)

        q_ref_follower = system._build_follower_action(q_ref_leader, float(grip))[:n]
        if system.map_signs is not None and system.map_index is not None:
            dq_ref_follower = system.map_signs * dq_ref_leader[system.map_index]
        else:
            dq_ref_follower = dq_ref_leader[:n]

        if mirror_source == "cmd":
            delta_leader = np.zeros(n, dtype=float)
            if system.map_signs is not None and system.map_index is not None:
                lim = min(len(system.map_index), len(system.map_signs), n)
                for i in range(lim):
                    leader_idx = int(system.map_index[i])
                    sign = float(system.map_signs[i])
                    if leader_idx < n and abs(sign) > 1e-6:
                        delta_leader[leader_idx] = delta_follower[i] / sign
            q_mirror_target = q_ref_leader + delta_leader
            dq_mirror_target = dq_ref_leader.copy()
        elif mirror_source in ("actual", "blend"):
            q_actual_target = q_mirror_target.copy()
            dq_actual_target = dq_mirror_target.copy()
            try:
                q_ur5e, dq_ur5e = system.get_follower_arm_state()
                q_actual_target = np.zeros(n, dtype=float)
                dq_actual_target = np.zeros(n, dtype=float)
                if (
                    system.map_index is not None
                    and system.map_signs is not None
                    and system.map_offsets is not None
                ):
                    lim = min(len(system.map_index), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            q_actual_target[leader_idx] = (
                                q_ur5e[i] - system.map_offsets[i]
                            ) / sign
                            dq_actual_target[leader_idx] = dq_ur5e[i] / sign
            except Exception:
                pass
            if mirror_source == "actual":
                q_mirror_target = q_actual_target
                dq_mirror_target = dq_actual_target
            else:
                delta_leader = np.zeros(n, dtype=float)
                if system.map_signs is not None and system.map_index is not None:
                    lim = min(len(system.map_index), len(system.map_signs), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            delta_leader[leader_idx] = delta_follower[i] / sign
                q_cmd_target = q_ref_leader + delta_leader
                dq_cmd_target = dq_ref_leader.copy()
                weights = np.array(
                    [
                        float(args.mirror_blend_cmd),
                        float(args.mirror_blend_actual),
                        float(args.mirror_blend_policy),
                    ],
                    dtype=float,
                )
                weights = np.maximum(weights, 0.0)
                denom = float(np.sum(weights))
                weights = weights / denom if denom > 1e-9 else np.array([1.0, 0.0, 0.0])
                q_mirror_target = (
                    weights[0] * q_cmd_target
                    + weights[1] * q_actual_target
                    + weights[2] * q_ref_leader
                )
                dq_mirror_target = (
                    weights[0] * dq_cmd_target
                    + weights[1] * dq_actual_target
                    + weights[2] * dq_ref_leader
                )
        else:
            q_mirror_target = q_ref_leader.copy()
            dq_mirror_target = dq_ref_leader.copy()

        tau_force_feedback = np.zeros(n, dtype=float)
        wrench_ur5e = np.zeros(6, dtype=float)
        if force_feedback_gain > 0.0:
            try:
                wrench_ur5e, tcp_jt = system.get_follower_tcp_force()
                if system.map_index is not None and system.map_signs is not None:
                    lim = min(len(system.map_index), n)
                    for i in range(lim):
                        leader_idx = int(system.map_index[i])
                        sign = float(system.map_signs[i])
                        if leader_idx < n and abs(sign) > 1e-6:
                            tau_force_feedback[leader_idx] = (
                                force_feedback_gain * tcp_jt[i] / sign
                            )
            except Exception:
                pass

        tau_gravity = system.gravity_compensation(q_leader, dq_leader)
        tau_friction = system.friction_compensation(dq_leader)
        tau_limit, torque_gripper = system.joint_limit_barrier(
            q_leader, dq_leader, float(grip), float(grip_vel)
        )
        tau_damping = np.zeros(n, dtype=float)
        if system.gravity_comp_velocity_damping != 0.0:
            tau_damping = -system.gravity_comp_velocity_damping * dq_leader

        tau_mirror = (
            kp_mirror * (q_mirror_target - q_leader)
            + kd_mirror * (dq_mirror_target - dq_leader)
        )

        tau_cmd_total = (
            tau_gravity
            + tau_friction
            + tau_limit
            + tau_damping
            + tau_mirror
            + tau_force_feedback
        )
        system.set_leader_joint_torque(tau_cmd_total, float(torque_gripper))

        tau_ext_human = yamane.update(q=q_leader, dq=dq_leader, tau_cmd=tau_cmd_total, dt=dt)
        tau_components = system.separate_contact_torque_components(
            tau_model=tau_cmd_total,
            tau_ext_shi=tau_ext_human,
        )

        tau_ext_db = np.where(
            np.abs(tau_ext_human) > deadband_tau,
            tau_ext_human - np.sign(tau_ext_human) * deadband_tau,
            0.0,
        )
        if velocity_gate_threshold > 1e-6:
            speed = float(np.linalg.norm(dq_leader))
            v_gate = 1.0 / np.sqrt(1.0 + (speed / velocity_gate_threshold) ** 2)
        else:
            v_gate = 1.0
        tau_ext_gated = tau_ext_db * v_gate

        k_virtual_safe = K_virtual if abs(K_virtual) > 1e-6 else 1.0
        delta_leader_new = tau_ext_gated / k_virtual_safe
        if system.map_signs is not None and system.map_index is not None:
            delta_follower = system.map_signs * delta_leader_new[system.map_index] * correction_gain
        else:
            delta_follower = delta_leader_new * correction_gain

        q_cmd_ur5e = q_ref_follower + delta_follower
        follower.command_joint_state_impedance(
            target_joints=q_cmd_ur5e,
            target_velocities=dq_ref_follower,
            kp=float(args.follower_kp),
            kd=float(args.follower_kd),
        )

        if has_gripper:
            gripper_action = system._build_follower_action(q_leader, float(grip))
            if len(gripper_action) > 6:
                with gripper_lock:
                    gripper_state["latest"] = int(
                        np.clip(gripper_action[-1] * 255, 0, 255)
                    )

        q_follower, dq_follower = system.get_follower_arm_state()
        q_follower = np.asarray(q_follower[:n], dtype=float)
        dq_follower = np.asarray(dq_follower[:n], dtype=float)

        epsilon = q_follower - q_cmd_ur5e
        recorder.record(
            t_mono=t_mono,
            q_ref=q_cmd_ur5e,
            dq_ref=dq_ref_follower,
            ddq_ref=ddq_ref,
            q_leader=q_leader,
            dq_leader=dq_leader,
            q_cmd_leader=q_mirror_target,
            q_c_leader=q_mirror_target,
            q_follower=q_follower,
            dq_follower=dq_follower,
            q_cmd_follower=q_cmd_ur5e,
            delta_corr=delta_follower,
            tau_cmd=tau_cmd_total,
            tau_ext_shi=tau_ext_human,
            tau_model=tau_components["tau_model"],
            tau_meas=tau_components["tau_meas"],
            tau_residual=tau_components["tau_residual"],
            contact_probability=float(np.clip(np.linalg.norm(tau_ext_gated) / 0.45, 0.0, 1.0)),
            contact_state=(
                ContactGate.CORRECTION_ACTIVE
                if np.linalg.norm(delta_follower) > 1e-6
                else ContactGate.NO_CONTACT
            ),
            epsilon=epsilon,
        )

        obs_snap.write(
            timestamp=t_mono,
            q=q_leader,
            dq=dq_leader,
            grip=float(grip),
            tau_ext=tau_ext_human,
            wrench=wrench_ur5e,
        )

        if is_stale and (t_mono - last_warn) > 1.0:
            last_warn = t_mono
            print("[WARN] trajectory stale")

        _sleep_remaining(dt, loop_t0)

    try:
        follower.robot.stopJ(2.0)
    except Exception:
        pass


def main() -> int:
    args = _parse_args()
    arch_mode = _resolve_arch_mode(args)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    running = True
    system: Optional[FACTRGravityCompensation] = None
    traj_buf: SharedTrajectoryBuffer | None = None
    obs_snap: SharedObservationSnapshot | None = None
    policy_proc: mp.Process | None = None
    stop_event: Any | None = None

    def _sig(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")

        n = int(system.num_arm_joints)
        system.policy_impedance_kp = float(args.kp)
        system.policy_impedance_kd = float(args.kd)

        if system.driver is not None:
            system.driver.set_torque_mode(False)
            time.sleep(0.05)
            system.driver.set_operating_mode(0)
            time.sleep(0.05)
            system.driver.set_torque_mode(True)
            time.sleep(0.05)

        q0, _, _, _ = system.get_leader_joint_states()

        horizon = int(args.horizon)
        traj_buf = SharedTrajectoryBuffer(
            name="epsilon_traj",
            horizon=horizon,
            n_joints=n,
            create=True,
        )
        obs_snap = SharedObservationSnapshot(
            name="epsilon_obs",
            n_joints=n,
            img_height=480,
            img_width=640,
            create=True,
        )

        initial_traj = np.tile(q0, (horizon, 1))
        traj_buf.write(initial_traj, time.monotonic())

        traj_interp = TrajectoryInterpolator(
            traj_buf=traj_buf,
            action_dt=float(args.action_dt),
            stale_threshold_s=5.0,
            fallback_q=q0.copy(),
        )

        policy_type, policy_config = _build_policy(args, n, q0, config_path, system.config)

        stop_event = mp.Event()
        policy_proc = mp.Process(
            target=policy_worker,
            args=(
                "epsilon_traj",
                "epsilon_obs",
                horizon,
                n,
                float(args.action_dt),
                policy_type,
                policy_config,
                stop_event,
            ),
            daemon=True,
        )
        policy_proc.start()
        print(f"[POLICY] started pid={policy_proc.pid} ({policy_type})")

        rate_hz = max(1.0, 1.0 / float(system.dt))
        max_duration = float(args.duration) if float(args.duration) > 0 else float(args.max_duration)
        recorder = EpsilonRecorder(
            n_joints=n,
            max_duration_s=max_duration,
            rate_hz=rate_hz,
        )
        ddq_computer = FilteredDDQComputer(n_joints=n)

        shi: Optional[MinimalistTorqueEstimator] = None
        if not args.no_shi and arch_mode in ("leader_tracking", "leader_admittance"):
            try:
                shi = _build_shi(system, config_path, n)
            except Exception as exc:
                print(f"[WARN] Shi estimator unavailable: {exc}")

        target_label = arch_mode
        mirror_source_used = None
        if arch_mode == "four_channel":
            if args.target != "follower":
                print("[WARN] four_channel records follower epsilon; ignoring --target leader")
            mirror_source_used = str(
                args.mirror_source
                or system.config.get("teleop", {}).get("four_channel", {}).get("mirror_source", "cmd")
            )
            _run_four_channel_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap)
        elif arch_mode == "leader_tracking":
            _run_leader_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap, shi)
        elif arch_mode == "leader_admittance":
            _run_leader_admittance_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap, shi)
        elif arch_mode == "follower_tracking":
            _run_follower_mode(system, args, traj_interp, recorder, ddq_computer, obs_snap)
        else:
            raise ValueError(f"Unknown mode: {arch_mode}")

        out_dir = Path(args.log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.monotonic())
        fname = f"epsilon_{target_label}_{args.test_mode}_{timestamp}.npz"
        out_path = out_dir / fname
        metadata = {
            "mode": arch_mode,
            "target": target_label,
            "test_mode": args.test_mode,
            "compensation": args.compensation,
            "kp": float(args.kp),
            "kd": float(args.kd),
            "follower_kp": float(args.follower_kp),
            "follower_kd": float(args.follower_kd),
            "four_channel": bool(arch_mode == "four_channel"),
            "mirror_source": mirror_source_used,
            "mirror_blend_cmd": float(args.mirror_blend_cmd),
            "mirror_blend_actual": float(args.mirror_blend_actual),
            "mirror_blend_policy": float(args.mirror_blend_policy),
            "adm_mass": float(args.adm_mass),
            "adm_damp": float(args.adm_damp),
            "adm_stiff": float(args.adm_stiff),
            "adm_leak": float(args.adm_leak),
            "adm_delta_max": float(args.adm_delta_max),
            "action_dt": float(args.action_dt),
            "horizon": int(args.horizon),
            "config": str(config_path),
        }
        if args.test_mode == "taskspace":
            metadata.update(
                {
                    "taskspace_motion": str(args.taskspace_motion),
                    "taskspace_amplitude": float(args.taskspace_amplitude),
                    "taskspace_axis": str(args.taskspace_axis),
                    "frequency": float(args.frequency),
                    "urdf_path": str(policy_config.get("urdf_path", "")),
                }
            )
        recorder.save(out_path, metadata=metadata)
        print(f"Saved: {out_path}")
        if recorder.dropped_samples:
            print("[WARN] recorder buffer overflowed; increase --max-duration")

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Run failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if stop_event is not None:
            stop_event.set()
        if policy_proc is not None:
            try:
                policy_proc.join(timeout=2.0)
                if policy_proc.is_alive():
                    policy_proc.terminate()
            except Exception:
                pass

        if traj_buf is not None:
            try:
                traj_buf.close()
                traj_buf.unlink()
            except Exception:
                pass
        if obs_snap is not None:
            try:
                obs_snap.close()
                obs_snap.unlink()
            except Exception:
                pass

        if system is not None:
            try:
                system.set_leader_joint_torque(np.zeros(system.num_arm_joints), 0.0)
                time.sleep(0.05)
                if system.driver is not None:
                    system.driver.set_operating_mode(3)
                time.sleep(0.05)
                system.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
