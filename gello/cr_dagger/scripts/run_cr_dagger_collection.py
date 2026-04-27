from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import signal
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Optional

import numpy as np

try:
    from pynput import keyboard as kb
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    kb = None  # type: ignore[assignment]

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
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.factr.gravity_compensation import FACTRGravityCompensation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# repo_id Validation
# ══════════════════════════════════════════════════════════════════════════════

_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _validate_repo_id(repo_id: str) -> str:
    """
    Validate that repo_id follows LeRobot's required 'username/dataset_name' format.

    LeRobot's LeRobotDataset and HuggingFace Hub both require this format.
    A bare path like '/media/.../my_dataset' is NOT a valid repo_id.

    Correct:  jstranghoener/cr_dagger_insertion
    Wrong:    /media/internal/nvme/.../cr_dagger_teleoperation
    """
    if not _REPO_ID_PATTERN.match(repo_id):
        raise ValueError(
            f"\n[ERROR] Invalid repo_id: '{repo_id}'\n"
            f"        repo_id must be in 'username/dataset_name' format.\n"
            f"        Example: --lerobot-repo jstranghoener/cr_dagger_insertion\n"
            f"        Use --lerobot-root for the local storage path.\n"
            f"        Example: --lerobot-root /media/internal/nvme/jstranghoener/data"
        )
    return repo_id


# ══════════════════════════════════════════════════════════════════════════════
# Keyboard Controller
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeKeyboardController:
    """
    Non-blocking pynput keyboard controller for in-episode actions.

    Keybindings (mirrors UR5eInsertionConfig key_mapping):
        →  Right Arrow  →  stop recording, save episode, end collection
        ←  Left Arrow   →  discard episode, re-record same index
        Space           →  mark episode as success, continue to next
    """

    def __init__(self) -> None:
        self.stop_recording: bool = False
        self.rerecord_episode: bool = False
        self.mark_success: bool = False
        self._listener: Optional[Any] = None

    def start(self) -> None:
        if not _PYNPUT_AVAILABLE:
            print("[KB] pynput not installed – keyboard episode controls disabled.")
            return
        self._listener = kb.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()
        print("[KB] In-episode keyboard controls active:")
        print("       →  Right Arrow  →  stop & save episode")
        print("       ←  Left Arrow   →  discard & re-record")
        print("       Space           →  mark success & continue")

    def _on_press(self, key: Any) -> None:
        if key == kb.Key.right:
            self.stop_recording = True
        elif key == kb.Key.left:
            self.rerecord_episode = True
        elif key == kb.Key.space:
            self.mark_success = True

    def reset_episode(self) -> None:
        """Reset per-episode flags. Call once before starting the inner loop."""
        self.rerecord_episode = False
        self.mark_success = False
        # stop_recording intentionally NOT reset – once set it stops the run

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


# ══════════════════════════════════════════════════════════════════════════════
# Episode discard helper
# ══════════════════════════════════════════════════════════════════════════════

def _discard_episode(
    lerobot_recorder: LeRobotCorrectionRecorder,
    npz_recorder: Optional[CorrectionRecorder],
) -> None:
    """Best-effort discard of the current in-progress episode buffers."""
    try:
        if hasattr(lerobot_recorder, "discard_episode"):
            lerobot_recorder.discard_episode()
        elif hasattr(lerobot_recorder, "clear_episode_buffer"):
            lerobot_recorder.clear_episode_buffer()
        else:
            print("[WARN] Recorder has no discard/clear method; buffer may persist.")
    except Exception as exc:
        print(f"[WARN] Could not discard lerobot episode: {exc}")

    if npz_recorder is not None:
        try:
            if hasattr(npz_recorder, "discard_episode"):
                npz_recorder.discard_episode()
        except Exception as exc:
            print(f"[WARN] Could not discard npz episode: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CR-DAgger production data collection")
    p.add_argument(
        "--interventions",
        action="store_true",
        help="Enable CR-DAgger intervention/correction mode. If unset, records Phase A teleop data.",
    )
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V3.yaml")
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=20.0)
    p.add_argument("--policy-type", choices=["dummy_sine", "dummy_hold"], default="dummy_hold")
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--frequency", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-dt", type=float, default=0.1)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--obs-decimation", type=int, default=33)

    # ── LeRobot dataset ───────────────────────────────────────────────────────
    p.add_argument(
        "--lerobot-repo",
        type=str,
        required=True,
        help=(
            "LeRobot repo_id in 'username/dataset_name' format. "
            "This is used as the dataset identifier (HuggingFace-style). "
            "Example: jstranghoener/cr_dagger_insertion"
        ),
    )
    p.add_argument(
        "--lerobot-root",
        type=str,
        default=None,
        help=(
            "Local root directory where the dataset is stored on disk. "
            "Defaults to ~/.cache/lerobot/datasets/{repo_id}. "
            "Example: /media/internal/nvme/jstranghoener/data"
        ),
    )
    # ─────────────────────────────────────────────────────────────────────────

    p.add_argument("--task-description", type=str, default="CR-DAgger correction episode")
    p.add_argument("--log-dir", type=str, default="cr_dagger_data")
    p.add_argument("--no-npz", action="store_true")
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--max-episode-duration", type=float, default=30.0)
    p.add_argument("--dataset-fps", type=int, default=30)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--camera-names", nargs="*", default=None)
    p.add_argument("--camera-device-ids", nargs="*", default=None)
    p.add_argument("--camera-flips", nargs="*", default=None)
    p.add_argument("--camera-warmup-s", type=float, default=2.0)
    p.add_argument("--enable-wrench", action="store_true")
    p.add_argument("--enable-wrench-feedback", action="store_true")
    return p.parse_args()


class MultiRealSenseRig:
    def __init__(
        self,
        camera_names: list[str],
        camera_device_ids: list[str],
        camera_flips: list[bool],
    ):
        if len(camera_names) != len(camera_device_ids):
            raise ValueError("camera_names and camera_device_ids must have the same length")
        if len(camera_names) != len(camera_flips):
            raise ValueError("camera_names and camera_flips must have the same length")

        self.camera_names = camera_names
        self._cameras: dict[str, RealSenseCamera] = {}
        self._threads: list[Thread] = []
        self._latest_rgb: dict[str, np.ndarray] = {}
        self._lock = Lock()
        self._stop = Event()

        for name, device_id, flip in zip(camera_names, camera_device_ids, camera_flips):
            self._cameras[name] = RealSenseCamera(device_id=device_id, flip=flip)

    def start(self) -> None:
        for name, cam in self._cameras.items():
            t = Thread(target=self._reader_loop, args=(name, cam), daemon=True, name=f"rs-{name}")
            t.start()
            self._threads.append(t)

    def _reader_loop(self, name: str, cam: RealSenseCamera) -> None:
        while not self._stop.is_set():
            try:
                rgb, _ = cam.read()
                with self._lock:
                    self._latest_rgb[name] = rgb
            except Exception:
                time.sleep(0.01)

    def get_images(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            if any(name not in self._latest_rgb for name in self.camera_names):
                return None
            return {name: self._latest_rgb[name].copy() for name in self.camera_names}

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for cam in self._cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass


def _resolve_camera_config(args: argparse.Namespace) -> tuple[list[str], list[str], list[bool]]:
    camera_names = list(args.camera_names) if args.camera_names else []
    camera_device_ids = list(args.camera_device_ids) if args.camera_device_ids else []

    if camera_names and not camera_device_ids:
        discovered = get_device_ids()
        if len(discovered) < len(camera_names):
            raise RuntimeError(
                f"Requested {len(camera_names)} cameras but found only "
                f"{len(discovered)} RealSense device(s): {discovered}"
            )
        camera_device_ids = discovered[: len(camera_names)]
    elif camera_device_ids and not camera_names:
        camera_names = [f"cam{i}" for i in range(len(camera_device_ids))]

    if len(camera_names) != len(camera_device_ids):
        raise ValueError("camera_names and camera_device_ids must have the same length")

    if not camera_names:
        return [], [], []

    if args.camera_flips:
        raw_flips = [bool(int(v)) for v in args.camera_flips]
        if len(raw_flips) == 1:
            camera_flips = raw_flips * len(camera_names)
        elif len(raw_flips) == len(camera_names):
            camera_flips = raw_flips
        else:
            raise ValueError("camera_flips must have length 1 or match number of cameras")
    else:
        camera_flips = [False] * len(camera_names)

    return camera_names, camera_device_ids, camera_flips


def _start_prepared_teleop(system: FACTRGravityCompensation) -> bool:
    """Start follower mirroring if FACTR teleop was prepared during init."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = _parse_args()

    # ── Validate repo_id early – fail fast before any hardware init ───────────
    try:
        repo_id = _validate_repo_id(args.lerobot_repo)
    except ValueError as e:
        print(e)
        return 1

    # ── Resolve local dataset root ────────────────────────────────────────────
    if args.lerobot_root is not None:
        lerobot_root = Path(args.lerobot_root)
        lerobot_root.mkdir(parents=True, exist_ok=True)
    else:
        # LeRobot default: ~/.cache/lerobot/datasets/{repo_id}
        lerobot_root = Path.home() / ".cache" / "lerobot" / "datasets" / repo_id
        lerobot_root.mkdir(parents=True, exist_ok=True)

    print(f"[DATASET] repo_id  : {repo_id}")
    print(f"[DATASET] local root: {lerobot_root}")

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
    camera_rig: MultiRealSenseRig | None = None
    traj_interp: TrajectoryInterpolator | None = None
    detector: FusedInterventionDetector | None = None
    admittance: JointSpaceAdmittanceController | None = None
    kbd: Optional[EpisodeKeyboardController] = None
    n = 6
    teleop_started = False

    def _sig(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        # ── Keyboard setup ────────────────────────────────────────────────────
        kbd = EpisodeKeyboardController()
        kbd.start()

        mode_str = "CR-DAgger interventions" if args.interventions else "Phase A baseline recording"
        print(f"[MODE] {mode_str}")

        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        n = int(system.num_arm_joints)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")
        if not system.teleop_enabled:
            print("[WARN] teleop is disabled in config; follower motion and wrench feedback are unavailable.")
        shi = _build_shi(system, config_path, n)

        q0, _, _, _ = system.get_leader_joint_states()

        if args.interventions:
            traj_buf = SharedTrajectoryBuffer(
                name="cr_dagger_traj",
                horizon=int(args.horizon),
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
            traj_buf.write(np.tile(q0, (int(args.horizon), 1)), time.monotonic())
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

        camera_names, camera_device_ids, camera_flips = _resolve_camera_config(args)
        if camera_names:
            camera_rig = MultiRealSenseRig(
                camera_names=camera_names,
                camera_device_ids=camera_device_ids,
                camera_flips=camera_flips,
            )
            camera_rig.start()
            print(f"[CAM] started {len(camera_names)} camera(s): {camera_names}")
            print(f"[CAM] device ids: {camera_device_ids}")
            time.sleep(max(0.0, float(args.camera_warmup_s)))

        dataset_fps = int(args.dataset_fps)
        if dataset_fps <= 0:
            raise ValueError("dataset-fps must be > 0")

        # ── LeRobot recorder: repo_id + explicit local root ───────────────────
        lerobot_recorder = LeRobotCorrectionRecorder(
            repo_id=repo_id,
            root=lerobot_root,
            fps=dataset_fps,
            task_description=str(args.task_description),
            n_joints=n,
            camera_names=camera_names if camera_names else None,
        )
        # ─────────────────────────────────────────────────────────────────────

        if args.interventions and not args.no_npz:
            npz_recorder = CorrectionRecorder(
                n_joints=n,
                log_dir=str(args.log_dir),
                latency_compensation_s=0.008,
            )

        shi.reset()

        if args.interventions:
            q_now, _, _, _ = system.get_leader_joint_states()
            system.driver.set_torque_mode(False)
            time.sleep(0.05)
            system.driver.set_operating_mode(3)
            time.sleep(0.05)
            system.driver.set_torque_mode(True)
            time.sleep(0.05)

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
                "q_hold": q_now.tolist(),
            }
            policy_proc = mp.Process(
                target=policy_worker,
                args=(
                    "cr_dagger_traj",
                    "cr_dagger_obs",
                    int(args.horizon),
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

        control_hz = max(1.0, 1.0 / float(system.dt))
        record_every_n = max(1, int(round(control_hz / float(dataset_fps))))
        print(
            f"[DATASET] control_hz={control_hz:.1f}, dataset_fps={dataset_fps}, "
            f"record_every_n={record_every_n}"
        )

        total_overruns = 0
        total_steps = 0

        # ── Episode loop (while instead of for → supports re-record) ─────────
        ep = 0
        num_episodes = int(args.num_episodes)

        while ep < num_episodes and running:

            input(
                f"\n{'=' * 40}\n"
                f"Press Enter to start episode {ep + 1}/{num_episodes}...\n"
                f"{'=' * 40}"
            )
            if not running:
                break

            # Reset per-episode keyboard flags AFTER the Enter prompt so any
            # accidental key presses during the pause don't affect this episode.
            if kbd is not None:
                kbd.reset_episode()

            if system.teleop_enabled and not teleop_started:
                teleop_started = _start_prepared_teleop(system)
                if not teleop_started:
                    raise RuntimeError(
                        "teleop is enabled but follower mirroring thread could not be started"
                    )

            if args.interventions:
                assert traj_interp is not None
                assert admittance is not None
                assert detector is not None
                t_mono = time.monotonic()
                q_ref_now, _, _ = traj_interp.get_reference(t_mono)
                admittance.reset(q_ref_now)
                detector.reset()

            lerobot_recorder.start_episode(task_description=str(args.task_description))
            if npz_recorder is not None:
                npz_recorder.start_episode(f"episode_{ep:04d}")

            ep_start = time.perf_counter()
            last_step_t = time.perf_counter()
            obs_write_counter = 0
            ep_steps = 0
            ep_overruns = 0
            corr_steps = 0
            ep_recorded_frames = 0
            rerecord = False
            marked_success = False

            # ── Inner control loop ──────────────────────────────────────────
            while running:
                now_perf = time.perf_counter()
                measured_dt = max(now_perf - last_step_t, 1e-4)
                last_step_t = now_perf

                # Time limit
                if now_perf - ep_start > float(args.max_episode_duration):
                    break

                # ── Keyboard event handling ─────────────────────────────────
                if kbd is not None:
                    if kbd.stop_recording:
                        print("\n[KB] STOP RECORDING – saving episode & ending run.")
                        running = False
                        break
                    if kbd.rerecord_episode:
                        print("\n[KB] RE-RECORD – discarding current episode.")
                        rerecord = True
                        break
                    if kbd.mark_success:
                        print("\n[KB] Episode marked as SUCCESS – saving & continuing.")
                        marked_success = True
                        kbd.mark_success = False
                        break
                # ───────────────────────────────────────────────────────────

                q, dq, grip, _ = system.get_leader_joint_states()
                currents_all = system.driver.get_currents()
                currents_arm = currents_all[:n] * system.joint_signs[:n]
                tau_ext = shi.update(q, dq, currents_arm)

                q_follower = np.zeros(n)
                dq_follower = np.zeros(n)
                try:
                    q_follower_raw, dq_follower_raw = system.get_follower_arm_state()
                    q_follower = np.asarray(q_follower_raw[:n], dtype=float)
                    dq_follower = np.asarray(dq_follower_raw[:n], dtype=float)
                except Exception:
                    pass

                wrench_ur5e = np.zeros(6)
                tau_wrench_fb = None
                if args.enable_wrench or args.enable_wrench_feedback:
                    wrench_ur5e, tcp_joint_torques = system.get_follower_tcp_force()
                    if args.enable_wrench_feedback:
                        tau_wrench_fb = np.asarray(tcp_joint_torques[:n], dtype=float)

                t_mono = time.monotonic()
                if args.interventions:
                    assert traj_interp is not None
                    assert admittance is not None
                    assert detector is not None
                    q_ref, dq_ref, is_stale = traj_interp.get_reference(t_mono)
                    _ = dq_ref
                    q_c = admittance.step(
                        q_ref=q_ref,
                        tau_ext_human=tau_ext,
                        dt=measured_dt,
                        tau_wrench_fb=tau_wrench_fb,
                    )
                    adm_state = admittance.get_state()

                    is_corr = detector.update(
                        tau_ext=tau_ext,
                        q_c=adm_state["q_c"],
                        q_ref=q_ref,
                        dq_c=adm_state["dq_c"],
                        wrench=wrench_ur5e if args.enable_wrench else None,
                    )
                    diag = detector.get_diagnostics()
                    if is_corr:
                        corr_steps += 1

                    target_hw = np.zeros(system.num_motors)
                    target_hw[:n] = q_c * system.joint_signs[:n] + system.joint_offsets[:n]
                    if system.num_motors > n:
                        target_hw[-1] = system.leader_gripper_raw_rad
                    system.driver.set_joints(target_hw.tolist())
                else:
                    is_stale = False
                    q_ref = q_follower.copy()
                    q_c = q_follower.copy()
                    adm_state = {
                        "q_c": q_follower.copy(),
                        "dq_c": dq_follower.copy(),
                    }
                    is_corr = False
                    diag = {
                        "votes": {
                            "torque": False,
                            "delta": False,
                            "energy": False,
                            "wrench": False,
                        }
                    }

                if npz_recorder is not None:
                    npz_recorder.record(
                        timestamp=t_mono,
                        q_ref=q_ref,
                        q_actual=q,
                        dq_actual=dq,
                        q_compliant=adm_state["q_c"],
                        dq_compliant=adm_state["dq_c"],
                        tau_ext_gello=tau_ext,
                        q_follower=q_follower,
                        wrench_ur5e=wrench_ur5e,
                        is_correction=bool(is_corr),
                        detector_diagnostics=diag,
                    )

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

                if ep_steps % record_every_n == 0:
                    images_for_frame = None
                    can_record_frame = True
                    if camera_rig is not None:
                        images_for_frame = camera_rig.get_images()
                        if images_for_frame is None:
                            can_record_frame = False

                    if can_record_frame:
                        obs_q = q if args.interventions else q_follower
                        obs_dq = dq if args.interventions else dq_follower
                        lerobot_recorder.add_frame(
                            timestamp=t_mono,
                            q=obs_q,
                            dq=obs_dq,
                            gripper=float(grip),
                            tau_ext=tau_ext,
                            wrench_ur5e=wrench_ur5e,
                            q_ref=q_ref,
                            q_compliant=adm_state["q_c"],
                            dq_compliant=adm_state["dq_c"],
                            is_correction=bool(is_corr),
                            detector_votes=detector_votes,
                            images=images_for_frame,
                        )
                        ep_recorded_frames += 1

                if args.interventions and obs_snap is not None:
                    obs_write_counter += 1
                    if obs_write_counter >= int(args.obs_decimation):
                        obs_write_counter = 0
                        obs_image = None
                        if camera_rig is not None:
                            latest_images = camera_rig.get_images()
                            if latest_images is not None and camera_names:
                                obs_image = latest_images[camera_names[0]]
                        obs_snap.write(
                            timestamp=t_mono,
                            q=q,
                            dq=dq,
                            grip=float(grip),
                            tau_ext=tau_ext,
                            wrench=wrench_ur5e,
                            image=obs_image,
                        )

                ep_steps += 1
                if ep_steps % 330 == 0:
                    if args.interventions and admittance is not None:
                        print(
                            f"[ep {ep:03d} {now_perf - ep_start:6.1f}s] "
                            f"INT={'YES' if is_corr else 'no '} "
                            f"|delta_q|={np.linalg.norm(admittance.get_delta_q()):.4f} "
                            f"|tau|={np.linalg.norm(tau_ext):.3f} stale={is_stale}"
                        )
                    else:
                        print(
                            f"[ep {ep:03d} {now_perf - ep_start:6.1f}s] "
                            f"PHASE_A |tau|={np.linalg.norm(tau_ext):.3f}"
                        )

                loop_time = time.perf_counter() - now_perf
                if loop_time > float(system.dt):
                    ep_overruns += 1
                sleep_s = max(0.0, float(system.dt) - loop_time)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            # ── Post-episode handling ───────────────────────────────────────

            if rerecord:
                _discard_episode(lerobot_recorder, npz_recorder)
                print(f"[KB] Episode {ep + 1} discarded – will re-record.\n")
                if not running:
                    break
                continue  # ep NOT incremented → same episode retried

            # Save episode
            if ep_recorded_frames > 0:
                ep_idx = lerobot_recorder.end_episode()
            else:
                ep_idx = -1
                print("[WARN] No dataset frames recorded in this episode; skipping save.")
            if npz_recorder is not None:
                npz_recorder.end_episode()

            duration_s = time.perf_counter() - ep_start
            end_reason = (
                "SUCCESS (keyboard)"
                if marked_success
                else "STOP (keyboard)"
                if kbd is not None and kbd.stop_recording
                else "time limit"
            )
            print(f"\nEpisode {ep_idx} complete ({end_reason}):")
            print(f"  Steps:            {ep_steps}")
            print(f"  Duration:         {duration_s:.1f}s")
            print(f"  Dataset frames:   {ep_recorded_frames}")
            print(
                f"  Correction steps: {corr_steps} "
                f"({100.0 * corr_steps / max(ep_steps, 1):.1f}%)"
            )
            print(
                f"  Timing overruns:  {ep_overruns}/{ep_steps} "
                f"({100.0 * ep_overruns / max(ep_steps, 1):.1f}%)"
            )

            total_steps += ep_steps
            total_overruns += ep_overruns
            ep += 1

        # ── Finalize dataset ──────────────────────────────────────────────────
        local_path = lerobot_recorder.finalize()
        print(f"\nDataset finalized at: {local_path}")
        print(f"  repo_id   : {repo_id}")
        print(f"  local root: {lerobot_root}")
        print(f"Total episodes recorded: {ep}/{num_episodes} requested")
        print(
            f"Total timing overruns: {total_overruns}/{total_steps} "
            f"({100.0 * total_overruns / max(total_steps, 1):.1f}%)"
        )

        if args.push_to_hub:
            url = lerobot_recorder.push_to_hub(private=True)
            print(f"Pushed to Hub: {url}")

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Run failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if kbd is not None:
            kbd.stop()

        if system is not None:
            system.running = False

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

        if camera_rig is not None:
            try:
                camera_rig.close()
            except Exception:
                pass

        if system is not None:
            try:
                system.set_leader_joint_torque(np.zeros(n), 0.0)
                time.sleep(0.05)
                if args.interventions and system.driver is not None:
                    system.driver.set_operating_mode(3)
                time.sleep(0.05)
                system.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())