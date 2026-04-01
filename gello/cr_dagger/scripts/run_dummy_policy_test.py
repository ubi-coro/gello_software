from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gello.bilat_4ch.gello_ur5e_observer_shi import (
    MinimalistEstimatorConfig,
    MinimalistTorqueEstimator,
    MotorParams,
    MotorType,
)
from gello.cr_dagger.core.admittance_controller import (
    AdmittanceParams,
    JointSpaceAdmittanceController,
)
from gello.cr_dagger.core.correction_recorder import CorrectionRecorder
from gello.cr_dagger.core.intervention_detector import (
    FusedDetectorParams,
    FusedInterventionDetector,
)
from gello.cr_dagger.core.lerobot_recorder import LeRobotCorrectionRecorder
from gello.cr_dagger.ipc.shared_observation_snapshot import SharedObservationSnapshot
from gello.cr_dagger.ipc.shared_trajectory_buffer import SharedTrajectoryBuffer
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


def _build_shi(system: FACTRGravityCompensation, config_path: Path, n: int) -> MinimalistTorqueEstimator:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CR-DAgger dummy policy integration test")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--duration", type=float, default=30.0, help="0 means run until Ctrl+C")
    p.add_argument("--settle-time", type=float, default=3.0)

    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=20.0)

    p.add_argument("--policy-type", choices=["dummy_sine", "dummy_hold"], default="dummy_sine")
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--frequency", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-dt", type=float, default=0.1)

    p.add_argument("--min-votes", type=int, default=2)

    p.add_argument("--lerobot-repo", type=str, default=None)
    p.add_argument("--task-description", type=str, default="Dummy sine wave compliance test")
    p.add_argument("--no-npz", action="store_true")
    p.add_argument("--log-dir", type=str, default="cr_dagger_data")

    p.add_argument("--obs-decimation", type=int, default=33)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

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
    npz_recorder: CorrectionRecorder | None = None
    lerobot_recorder: LeRobotCorrectionRecorder | None = None
    n = 6

    def _sig(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        n = int(system.num_arm_joints)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")

        shi = _build_shi(system, config_path, n)
        q0, _, grip0, _ = system.get_leader_joint_states()

        horizon = int(args.horizon)
        traj_buf = SharedTrajectoryBuffer(
            name="cr_dagger_traj",
            horizon=horizon,
            n_joints=n,
            create=True,
        )
        obs_snap = SharedObservationSnapshot(
            name="cr_dagger_obs",
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

        detector = FusedInterventionDetector(
            params=FusedDetectorParams(min_votes=int(args.min_votes)),
            n_joints=n,
            dt=float(system.dt),
        )

        if not args.no_npz:
            npz_recorder = CorrectionRecorder(
                n_joints=n,
                log_dir=str(args.log_dir),
                latency_compensation_s=0.008,
            )
        if args.lerobot_repo:
            lerobot_recorder = LeRobotCorrectionRecorder(
                repo_id=str(args.lerobot_repo),
                fps=max(1, int(round(1.0 / float(system.dt)))),
                task_description=str(args.task_description),
                n_joints=n,
                camera_names=None,
            )

        print(f"[SETTLE] gravity comp for {float(args.settle_time):.1f}s")
        t_settle_start = time.perf_counter()
        while running and (time.perf_counter() - t_settle_start < float(args.settle_time)):
            q, dq, _, _ = system.get_leader_joint_states()
            tau_grav = system.gravity_compensation(q, dq)
            tau_fric = system.friction_compensation(dq)
            tau_damp = -float(system.gravity_comp_velocity_damping) * dq
            system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)
            time.sleep(max(0.0, float(system.dt) - 0.0005))

        q_now, _, _, _ = system.get_leader_joint_states()

        system.driver.set_torque_mode(False)
        time.sleep(0.05)
        system.driver.set_operating_mode(3)
        time.sleep(0.05)
        system.driver.set_torque_mode(True)
        time.sleep(0.05)

        shi.reset()

        admittance = JointSpaceAdmittanceController(
            params=AdmittanceParams(
                mass=float(args.mass),
                damping=float(args.damping),
                stiffness=float(args.stiffness),
            ),
            n_joints=n,
            q_init=q_now.copy(),
            q_min=system.arm_joint_limits_min,
            q_max=system.arm_joint_limits_max,
        )

        stop_event = mp.Event()
        policy_config = {
            "center": q_now.tolist(),
            "amplitude": [float(args.amplitude)] * n,
            "frequency": [float(args.frequency)] * n,
        }
        policy_proc = mp.Process(
            target=policy_worker,
            args=(
                "cr_dagger_traj",
                "cr_dagger_obs",
                horizon,
                n,
                float(args.action_dt),
                str(args.policy_type),
                policy_config,
                stop_event,
            ),
            daemon=True,
        )
        policy_proc.start()
        print(f"[POLICY] started pid={policy_proc.pid}")

        if npz_recorder:
            npz_recorder.start_episode("dummy_test")
        if lerobot_recorder:
            lerobot_recorder.start_episode()

        print(f"[REPLAY] running for {float(args.duration):.1f}s (0 means until Ctrl+C)")

        t_replay_start = time.perf_counter()
        last_step_t = time.perf_counter()
        obs_write_counter = 0
        step_count = 0
        timing_overruns = 0

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf

            t_elapsed = now_perf - t_replay_start
            if float(args.duration) > 0 and t_elapsed > float(args.duration):
                print("Duration reached.")
                break

            q, dq, grip, _ = system.get_leader_joint_states()
            currents_all = system.driver.get_currents()
            currents_arm = currents_all[:n] * system.joint_signs[:n]
            tau_ext = shi.update(q, dq, currents_arm)

            t_mono = time.monotonic()
            q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
            if is_stale and step_count % 330 == 0:
                print("[WARN] trajectory stale")

            q_c = admittance.step(q_ref=q_ref, tau_ext_human=tau_ext, dt=measured_dt)
            delta_q = admittance.get_delta_q()
            adm_state = admittance.get_state()

            is_corr = detector.update(
                tau_ext=tau_ext,
                q_c=adm_state["q_c"],
                q_ref=q_ref,
                dq_c=adm_state["dq_c"],
                wrench=None,
            )
            diag = detector.get_diagnostics()

            target_hw = np.zeros(system.num_motors)
            target_hw[:n] = q_c * system.joint_signs[:n] + system.joint_offsets[:n]
            if system.num_motors > n:
                target_hw[-1] = system.leader_gripper_raw_rad
            system.driver.set_joints(target_hw.tolist())

            if npz_recorder is not None:
                npz_recorder.record(
                    timestamp=t_mono,
                    q_ref=q_ref,
                    q_actual=q,
                    dq_actual=dq,
                    q_compliant=adm_state["q_c"],
                    dq_compliant=adm_state["dq_c"],
                    tau_ext_gello=tau_ext,
                    q_follower=np.zeros(6),
                    wrench_ur5e=np.zeros(6),
                    is_correction=bool(is_corr),
                    detector_diagnostics=diag,
                )

            if lerobot_recorder is not None:
                votes = diag.get("votes", {})
                detector_votes = np.array(
                    [
                        bool(votes.get("torque", False)),
                        bool(votes.get("delta_q", votes.get("delta", False))),
                        bool(votes.get("energy", False)),
                        bool(votes.get("wrench", False)),
                    ],
                    dtype=bool,
                )
                lerobot_recorder.add_frame(
                    timestamp=t_mono,
                    q=q,
                    dq=dq,
                    gripper=float(grip),
                    tau_ext=tau_ext,
                    wrench_ur5e=np.zeros(6),
                    q_ref=q_ref,
                    q_compliant=adm_state["q_c"],
                    dq_compliant=adm_state["dq_c"],
                    is_correction=bool(is_corr),
                    detector_votes=detector_votes,
                )

            obs_write_counter += 1
            if obs_write_counter >= int(args.obs_decimation):
                obs_write_counter = 0
                obs_snap.write(
                    timestamp=t_mono,
                    q=q,
                    dq=dq,
                    grip=float(grip),
                    tau_ext=tau_ext,
                    wrench=np.zeros(6),
                )

            step_count += 1
            if step_count % 330 == 0:
                print(
                    f"[{t_elapsed:6.1f}s] INT={'YES' if is_corr else 'no '} "
                    f"|delta_q|={np.linalg.norm(delta_q):.4f} "
                    f"|tau|={np.linalg.norm(tau_ext):.3f} stale={is_stale}"
                )

            loop_time = time.perf_counter() - now_perf
            if loop_time > float(system.dt):
                timing_overruns += 1
            sleep_s = max(0.0, float(system.dt) - loop_time)
            if sleep_s > 0:
                time.sleep(sleep_s)

        if stop_event is not None:
            stop_event.set()
        if policy_proc is not None:
            policy_proc.join(timeout=3.0)
            if policy_proc.is_alive():
                policy_proc.terminate()

        if npz_recorder is not None:
            ep_path = npz_recorder.end_episode()
            print(f"NPZ saved: {ep_path}")

        if lerobot_recorder is not None:
            lerobot_recorder.end_episode()
            local_path = lerobot_recorder.finalize()
            print(f"LeRobot dataset: {local_path}")

        print(
            f"\nTiming overruns: {timing_overruns}/{step_count} "
            f"({100.0 * timing_overruns / max(step_count, 1):.1f}%)"
        )

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
                policy_proc.join(timeout=1.0)
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
                system.set_leader_joint_torque(np.zeros(n), 0.0)
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
