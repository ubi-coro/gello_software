# run_cr_dagger_collection.py
from __future__ import annotations

import argparse
import enum
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
    import yaml
except ImportError:  # pragma: no cover - production env includes PyYAML
    yaml = None

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
from gello.cr_dagger.core.correction_recorder import CorrectionRecorder
from gello.cr_dagger.core.intervention_detector import (
    FusedDetectorParams,
    FusedInterventionDetector,
)
from gello.cr_dagger.core.lerobot_recorder import LeRobotCorrectionRecorder
from gello.cr_dagger.core.bota_se3_residual import run_phase_b_bota_se3_loop
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
# High-frequency sensor reader (decouples I/O from recording logic)
# ══════════════════════════════════════════════════════════════════════════════

# TODO: Deprecated — superseded by the unified Phase B loop. Keep for
# now for debugging, but consider removing this class entirely.
class SensorSnapshot:
    """Immutable snapshot of all sensor data at one timestep."""
    __slots__ = (
        "t_mono", "t_perf",
        "q_leader", "dq_leader", "grip_leader", "leader_gripper_raw_rad",
        "currents_arm",
        "q_follower", "dq_follower", "gripper_follower",
        "wrench_ur5e", "tcp_joint_torques",
        "tau_ext",
    )

    def __init__(self, n: int = 6):
        self.t_mono: float = 0.0
        self.t_perf: float = 0.0
        self.q_leader: np.ndarray = np.zeros(n)
        self.dq_leader: np.ndarray = np.zeros(n)
        self.grip_leader: float = 0.0
        self.leader_gripper_raw_rad: float = 0.0
        self.currents_arm: np.ndarray = np.zeros(n)
        self.q_follower: np.ndarray = np.zeros(n)
        self.dq_follower: np.ndarray = np.zeros(n)
        self.gripper_follower: float = 0.0
        self.wrench_ur5e: np.ndarray = np.zeros(6)
        self.tcp_joint_torques: np.ndarray = np.zeros(n)
        self.tau_ext: np.ndarray = np.zeros(n)


# TODO: Deprecated — superseded by the unified Phase B loop. Keep for
# now for debugging, but consider removing this class entirely.
class AsyncSensorReader:
    """
    Reads all sensors in a dedicated thread at maximum rate.
    The control/recording loop just grabs the latest snapshot (near-zero cost read).

    Solves:
    1. Bus contention (single thread owns Dynamixel + RTDE reads)
    2. Latency stacking (parallel I/O instead of sequential)
    3. Consistent timestamps (all data from ~same instant)
    """

    def __init__(
        self,
        system: FACTRGravityCompensation,
        shi: MinimalistTorqueEstimator,
        n: int,
        enable_wrench: bool = False,
    ):
        self._system = system
        self._shi = shi
        self._n = n
        self._enable_wrench = enable_wrench

        self._lock = Lock()
        self._snapshot = SensorSnapshot(n)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._read_count = 0
        self._overrun_count = 0
        self._start_t = 0.0

    @property
    def snapshot(self) -> SensorSnapshot:
        """Get latest sensor snapshot (thread-safe, near-zero cost)."""
        with self._lock:
            snap = SensorSnapshot(self._n)
            # Shallow copy all attributes
            snap.t_mono = self._snapshot.t_mono
            snap.t_perf = self._snapshot.t_perf
            snap.q_leader = self._snapshot.q_leader.copy()
            snap.dq_leader = self._snapshot.dq_leader.copy()
            snap.grip_leader = self._snapshot.grip_leader
            snap.leader_gripper_raw_rad = self._snapshot.leader_gripper_raw_rad
            snap.currents_arm = self._snapshot.currents_arm.copy()
            snap.q_follower = self._snapshot.q_follower.copy()
            snap.dq_follower = self._snapshot.dq_follower.copy()
            snap.gripper_follower = self._snapshot.gripper_follower
            snap.wrench_ur5e = self._snapshot.wrench_ur5e.copy()
            snap.tcp_joint_torques = self._snapshot.tcp_joint_torques.copy()
            snap.tau_ext = self._snapshot.tau_ext.copy()
            return snap

    @property
    def read_hz(self) -> float:
        """Actual achieved read rate."""
        if self._read_count == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_t
        return self._read_count / max(elapsed, 1e-6)

    def start(self) -> None:
        """Start the sensor reading thread."""
        self._start_t = time.perf_counter()
        self._thread = Thread(target=self._reader_loop, daemon=True, name="async-sensor-reader")
        self._thread.start()

    def stop(self) -> None:
        """Stop the sensor reading thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _reader_loop(self) -> None:
        """
        Tight sensor read loop. Targets system.dt but doesn't sleep if reads
        are already slow — just provides data as fast as possible.
        """
        system = self._system
        shi = self._shi
        n = self._n
        dt_target = float(system.dt)
        driver = system.driver
        if driver is None:
            return

        while not self._stop.is_set():
            t0 = time.perf_counter()

            snap = SensorSnapshot(n)
            snap.t_perf = t0
            snap.t_mono = time.monotonic()

            # ── Dynamixel: combined position + velocity + current read ────
            try:
                if hasattr(driver, "get_positions_velocities_and_currents"):
                    # Optimized: single bulk read
                    pos, vel, cur = driver.get_positions_velocities_and_currents()
                    q_raw = np.asarray(pos[:n], dtype=float)
                    dq_raw = np.asarray(vel[:n], dtype=float)
                    snap.q_leader = (q_raw - system.joint_offsets[:n]) * system.joint_signs[:n]
                    snap.dq_leader = dq_raw * system.joint_signs[:n]
                    snap.currents_arm = np.asarray(cur[:n], dtype=float) * system.joint_signs[:n]
                    snap.leader_gripper_raw_rad = float(pos[-1]) if len(pos) > n else 0.0
                    snap.grip_leader = (pos[-1] - system.joint_offsets[-1]) * system.joint_signs[-1] if len(pos) > n else 0.0
                else:
                    # Fallback: two separate reads
                    q_arm, dq_arm, grip, _ = system.get_leader_joint_states()
                    snap.q_leader = q_arm.copy()
                    snap.dq_leader = dq_arm.copy()
                    snap.grip_leader = grip
                    snap.leader_gripper_raw_rad = system.leader_gripper_raw_rad

                    cur_all = driver.get_currents()
                    snap.currents_arm = np.asarray(cur_all[:n], dtype=float) * system.joint_signs[:n]
            except Exception:
                # On read failure, keep previous snapshot
                pass

            # ── Shi torque estimator (pure computation, <0.1ms) ───────────
            try:
                snap.tau_ext = shi.update(snap.q_leader, snap.dq_leader, snap.currents_arm)
            except Exception:
                snap.tau_ext = np.zeros(n)

            # ── RTDE follower reads (separate interface, no bus conflict) ──
            try:
                if system._direct_follower_robot is not None and hasattr(system._direct_follower_robot, "r_inter"):
                    r = system._direct_follower_robot.r_inter
                    snap.q_follower = np.asarray(r.getActualQ()[:n], dtype=float)
                    snap.dq_follower = np.asarray(r.getActualQd()[:n], dtype=float)

                    if self._enable_wrench:
                        snap.wrench_ur5e = np.asarray(r.getActualTCPForce(), dtype=float)
                        if hasattr(system, "J_semi") and system.J_semi is not None:
                            tcp_jt = system.J_semi.T @ snap.wrench_ur5e
                            if hasattr(system, "J_semi_tare") and system.J_semi_tare is not None:
                                tcp_jt -= system.J_semi_tare
                            snap.tcp_joint_torques = tcp_jt[:n]
                else:
                    q_f, dq_f = system.get_follower_arm_state()
                    snap.q_follower = np.asarray(q_f[:n], dtype=float)
                    snap.dq_follower = np.asarray(dq_f[:n], dtype=float)
            except Exception:
                pass

            # ── Publish snapshot ──────────────────────────────────────────
            with self._lock:
                self._snapshot = snap

            self._read_count += 1

            # ── Timing ────────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            sleep_s = dt_target - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                self._overrun_count += 1


# ══════════════════════════════════════════════════════════════════════════════
# repo_id Validation
# ══════════════════════════════════════════════════════════════════════════════

_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _validate_repo_id(repo_id: str) -> str:
    if not _REPO_ID_PATTERN.match(repo_id):
        raise ValueError(
            f"\n[ERROR] Invalid repo_id: '{repo_id}'\n"
            f"        repo_id must be in 'username/dataset_name' format.\n"
            f"        Example: --lerobot-repo jstranghoener/cr_dagger_insertion\n"
            f"        Use --lerobot-root for the local storage path."
        )
    return repo_id


def _slug_filename_part(value: object, default: str = "run") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_.-")
    return text or default


# ══════════════════════════════════════════════════════════════════════════════
# Episode State Machine
# ══════════════════════════════════════════════════════════════════════════════

class EpisodeState(enum.Enum):
    """State machine for episode lifecycle."""
    IDLE = "idle"           # Waiting for user to start recording
    RECORDING = "recording" # Actively recording an episode
    RESET = "reset"         # Post-episode reset phase (teleop still active)


class EpisodeController:
    """
    Unified controller for episode lifecycle via keyboard AND foot pedal.

    The foot pedal (a USB HID device that sends a configurable key) has
    context-dependent behavior:

        State       | Pedal / Enter        | ← Left Arrow    | → Right Arrow
        ------------|----------------------|------------------|------------------
        IDLE        | Start recording      | —                | Quit collection
        RECORDING   | Mark success & stop  | Discard & retry  | Stop & save
        RESET       | Start next recording | —                | Quit collection

    This replaces the old EpisodeKeyboardController + blocking input() approach.
    """

    def __init__(self, foot_pedal_key: str = "b") -> None:
        self._state = EpisodeState.IDLE
        self._lock = Lock()
        self._listener: Optional[Any] = None

        # ── Events (thread-safe, one-shot) ────────────────────────────────
        self.start_requested = Event()     # IDLE/RESET → start recording
        self.success_requested = Event()   # RECORDING → mark success
        self.discard_requested = Event()   # RECORDING → discard & retry
        self.stop_requested = Event()      # any state → save & quit

        # ── Foot pedal key configuration ──────────────────────────────────
        self._pedal_key = self._resolve_key(foot_pedal_key)
        self._pedal_key_name = foot_pedal_key

    # ── Key resolution ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_key(key_str: str) -> Any:
        """Resolve a key string to a pynput Key or KeyCode."""
        if not _PYNPUT_AVAILABLE:
            return None
        # Try special keys first (f1-f12, space, enter, etc.)
        special = getattr(kb.Key, key_str.lower(), None)
        if special is not None:
            return special
        # Single character → KeyCode
        if len(key_str) == 1:
            return kb.KeyCode.from_char(key_str)
        return None

    # ── State management ──────────────────────────────────────────────────

    @property
    def state(self) -> EpisodeState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, new_state: EpisodeState) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            print(f"[STATE] {old.value} → {new_state.value}")

    def reset_events(self) -> None:
        """Clear all one-shot events (call when entering a new state)."""
        self.start_requested.clear()
        self.success_requested.clear()
        self.discard_requested.clear()
        # NOTE: stop_requested is intentionally NOT cleared

    # ── Keyboard / pedal listener ─────────────────────────────────────────

    def start(self) -> None:
        if not _PYNPUT_AVAILABLE:
            print("[KB] pynput not installed – keyboard/pedal controls disabled.")
            print("[KB] Falling back to blocking input() prompts.")
            return

        self._listener = kb.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

        print(f"[CONTROLS] Foot pedal key: '{self._pedal_key_name}'")
        print(f"[CONTROLS] Key bindings (context-dependent):")
        print(f"  IDLE/RESET:  Pedal/Enter → start recording")
        print(f"  RECORDING:   Pedal       → mark SUCCESS & save")
        print(f"  RECORDING:   ← Left      → DISCARD & retry")
        print(f"  RECORDING:   → Right     → STOP collection & save")
        print(f"  ANY:         Ctrl+C      → emergency stop")

    def _on_press(self, key: Any) -> None:
        state = self.state

        # ── Foot pedal or Enter key ───────────────────────────────────────
        is_pedal = (key == self._pedal_key)
        is_enter = (key == kb.Key.enter)

        if is_pedal or is_enter:
            if state == EpisodeState.IDLE or state == EpisodeState.RESET:
                self.start_requested.set()
            elif state == EpisodeState.RECORDING:
                self.success_requested.set()
            return

        # ── Arrow keys (recording-only) ──────────────────────────────────
        if state == EpisodeState.RECORDING:
            if key == kb.Key.left:
                self.discard_requested.set()
            elif key == kb.Key.right:
                self.stop_requested.set()

        # ── Right arrow in IDLE/RESET → quit ──────────────────────────────
        if state in (EpisodeState.IDLE, EpisodeState.RESET):
            if key == kb.Key.right:
                self.stop_requested.set()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    # ── Blocking wait helpers (fallback when pynput unavailable) ──────────

    def wait_for_start(self, timeout: float | None = None) -> bool:
        """
        Wait for user to request episode start.

        Returns True if start was requested, False if stop was requested.
        Uses pynput events if available, otherwise falls back to input().
        """
        if self._listener is not None:
            # Non-blocking pynput mode: wait on events
            while True:
                if self.stop_requested.is_set():
                    return False
                if self.start_requested.wait(timeout=0.1):
                    self.start_requested.clear()
                    return True
        else:
            # Fallback: blocking input()
            try:
                input("  Press Enter to start recording (Ctrl+C to quit)... ")
                return True
            except (EOFError, KeyboardInterrupt):
                return False


# ══════════════════════════════════════════════════════════════════════════════
# Episode discard helper
# ══════════════════════════════════════════════════════════════════════════════

def _discard_episode(
    lerobot_recorder: LeRobotCorrectionRecorder,
    npz_recorder: Optional[CorrectionRecorder],
) -> None:
    try:
        if hasattr(lerobot_recorder, "discard_episode"):
            lerobot_recorder.discard_episode()
        elif hasattr(lerobot_recorder, "clear_episode_buffer"):
            lerobot_recorder.clear_episode_buffer()
        else:
            print("[WARN] Recorder has no discard/clear method.")
    except Exception as exc:
        print(f"[WARN] Could not discard lerobot episode: {exc}")

    if npz_recorder is not None:
        try:
            if hasattr(npz_recorder, "discard_episode"):
                npz_recorder.discard_episode()
        except Exception as exc:
            print(f"[WARN] Could not discard npz episode: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight Phase A recording loop (RTDE + cameras only)
# ══════════════════════════════════════════════════════════════════════════════

def _run_recording_loop_phase_a(
    system: FACTRGravityCompensation,
    ctrl: EpisodeController,
    lerobot_recorder: LeRobotCorrectionRecorder,
    camera_rig: Optional[MultiRealSenseRig],
    args: argparse.Namespace,
    ep: int,
    n: int,
) -> dict:
    """Phase A recording: teleop owns Dynamixel I/O; this loop reads RTDE and cameras."""
    dataset_fps = int(args.dataset_fps)
    frame_dt = 1.0 / max(dataset_fps, 1)
    max_duration = float(args.max_episode_duration)
    ep_start = time.perf_counter()
    ep_steps = 0
    ep_overruns = 0
    ep_recorded_frames = 0

    while True:
        frame_start = time.perf_counter()

        if frame_start - ep_start > max_duration:
            print(f"[REC] Time limit reached ({max_duration}s)")
            break

        if ctrl.stop_requested.is_set():
            return {
                "rerecord": False,
                "success": False,
                "running": False,
                "steps": ep_steps,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
            }
        if ctrl.discard_requested.is_set():
            return {
                "rerecord": True,
                "success": False,
                "running": True,
                "steps": ep_steps,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
            }
        if ctrl.success_requested.is_set():
            return {
                "rerecord": False,
                "success": True,
                "running": True,
                "steps": ep_steps,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
            }

        t_mono = time.monotonic()
        q_follower = np.zeros(n, dtype=float)
        dq_follower = np.zeros(n, dtype=float)
        try:
            q_follower_raw, dq_follower_raw = system.get_follower_arm_state()
            q_follower = np.asarray(q_follower_raw[:n], dtype=float)
            dq_follower = np.asarray(dq_follower_raw[:n], dtype=float)
        except Exception:
            pass

        gripper_follower = 0.0
        try:
            fb = system.get_follower_gripper_feedback()
            gripper_follower = float(fb.get("position", 0.0))
        except Exception:
            pass

        action = getattr(system, "_teleop_last_action", None)
        if action is None:
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
            ep_steps += 1
            continue
        action = np.asarray(action, dtype=float).copy()

        images = None
        if camera_rig is not None:
            images = camera_rig.get_images()
            if images is None:
                elapsed = time.perf_counter() - frame_start
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                ep_steps += 1
                continue

        lerobot_recorder.add_frame(
            timestamp=t_mono,
            q=q_follower,
            dq=dq_follower,
            gripper=gripper_follower,
            action=action,
            tau_ext=np.zeros(n, dtype=float),
            wrench_ur5e=np.zeros(6, dtype=float),
            q_ref=q_follower,
            q_compliant=q_follower,
            dq_compliant=dq_follower,
            is_correction=False,
            images=images,
            detector_votes=np.zeros(4, dtype=np.float32),
        )
        ep_recorded_frames += 1
        ep_steps += 1

        if ep_recorded_frames % dataset_fps == 0:
            elapsed_s = frame_start - ep_start
            print(
                f"  [ep {ep+1:03d} {elapsed_s:5.1f}s] "
                f"frames={ep_recorded_frames}"
            )

        elapsed = time.perf_counter() - frame_start
        if elapsed > frame_dt:
            ep_overruns += 1
        else:
            time.sleep(frame_dt - elapsed)

    return {
        "rerecord": False,
        "success": False,
        "running": True,
        "steps": ep_steps,
        "overruns": ep_overruns,
        "frames": ep_recorded_frames,
    }


def _run_recording_loop_phase_b(
    ctrl: EpisodeController,
    lerobot_recorder: LeRobotCorrectionRecorder,
    camera_rig: Optional[MultiRealSenseRig],
    state_cache: dict[str, Any],
    args: argparse.Namespace,
    ep: int,
    n: int,
    obs_snap: SharedObservationSnapshot | None = None,
    npz_recorder: CorrectionRecorder | None = None,
) -> dict:
    """Phase B recording: read only from shared cache and cameras."""
    dataset_fps = int(args.dataset_fps)
    frame_dt = 1.0 / max(dataset_fps, 1)
    max_duration = float(args.max_episode_duration)
    ep_start = time.perf_counter()
    ep_recorded_frames = 0
    ep_overruns = 0
    corr_steps = 0
    cache_lock = state_cache["lock"]

    while True:
        frame_start = time.perf_counter()

        if frame_start - ep_start > max_duration:
            print(f"[REC] Time limit reached ({max_duration}s)")
            break

        if ctrl.stop_requested.is_set():
            return {
                "rerecord": False,
                "success": False,
                "running": False,
                "steps": ep_recorded_frames,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
                "corr_steps": corr_steps,
            }
        if ctrl.discard_requested.is_set():
            return {
                "rerecord": True,
                "success": False,
                "running": True,
                "steps": ep_recorded_frames,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
                "corr_steps": corr_steps,
            }
        if ctrl.success_requested.is_set():
            return {
                "rerecord": False,
                "success": True,
                "running": True,
                "steps": ep_recorded_frames,
                "overruns": ep_overruns,
                "frames": ep_recorded_frames,
                "corr_steps": corr_steps,
            }

        t_mono = time.monotonic()
        with cache_lock:
            if not state_cache.get("updated", False):
                time.sleep(0.001)
                continue

            # Staleness check: ensure the unified loop updated recently.
            cache_t = float(state_cache.get("t_mono", 0.0))
            if t_mono - cache_t > 0.1:
                print("[WARN] Unified loop stale (no updates in >100ms) — skipping frame")
                time.sleep(0.01)
                continue

            q_follower = state_cache.get("q_follower", np.zeros(n)).copy()
            dq_follower = state_cache.get("dq_follower", np.zeros(n)).copy()
            q_leader = state_cache.get("q_leader", np.zeros(n)).copy()
            dq_leader = state_cache.get("dq_leader", np.zeros(n)).copy()
            tau_ext = state_cache.get("tau_ext", np.zeros(n)).copy()
            wrench_ur5e = state_cache.get("wrench_ur5e", np.zeros(6)).copy()
            wrench_sensor_raw = state_cache.get("wrench_sensor_raw", np.zeros(6)).copy()
            wrench_base_raw = state_cache.get("wrench_base_raw", np.zeros(6)).copy()
            wrench_base = state_cache.get("wrench_base", wrench_ur5e).copy()
            task_delta = state_cache.get("task_delta", np.zeros(6)).copy()
            task_pose_error = state_cache.get("task_pose_error", np.zeros(6)).copy()
            delta_leader = state_cache.get("delta_leader", np.zeros(n)).copy()
            delta_leader_raw = state_cache.get("delta_leader_raw", np.zeros(n)).copy()
            gripper_follower = float(state_cache.get("gripper_follower", 0.0))
            is_corr = bool(state_cache.get("is_correction", False))
            diag = state_cache.get("detector_diag", {})
            policy_action = state_cache.get("policy_action", np.zeros(n + 1)).copy()
            compliant_action = state_cache.get("compliant_action", np.zeros(n + 1)).copy()
            q_ref_leader = state_cache.get("q_ref", np.zeros(n)).copy()
            q_cmd_leader = state_cache.get("q_cmd_leader", q_ref_leader + delta_leader).copy()
            q_cmd_ur5e = state_cache.get("q_cmd_ur5e", compliant_action[:n]).copy()
            epsilon_leader = state_cache.get("epsilon_leader", q_leader - q_cmd_leader).copy()
            epsilon_ur5e = state_cache.get("epsilon_ur5e", q_follower - q_cmd_ur5e).copy()

            # Reset the updated flag to detect future updates and stale state.
            try:
                state_cache["updated"] = False
            except Exception:
                pass

        if is_corr:
            corr_steps += 1

        images = None
        if camera_rig is not None:
            images = camera_rig.get_images()
            if images is None:
                elapsed = time.perf_counter() - frame_start
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                continue

        obs_image = None
        if images is not None:
            obs_image = next(iter(images.values())) if len(images) > 0 else None

        if obs_snap is not None:
            obs_snap.write(
                timestamp=t_mono,
                q=q_follower,
                dq=dq_follower,
                grip=gripper_follower,
                tau_ext=tau_ext,
                wrench=wrench_ur5e,
                image=obs_image,
                images=images,
            )

        if npz_recorder is not None:
            npz_recorder.record(
                timestamp=t_mono,
                q_ref=policy_action[:n],
                q_actual=q_leader,
                dq_actual=dq_leader,
                q_compliant=compliant_action[:n],
                dq_compliant=np.zeros(n),
                tau_ext_gello=tau_ext,
                q_follower=q_follower,
                wrench_ur5e=wrench_ur5e,
                is_correction=bool(is_corr),
                detector_diagnostics=diag,
                q_ref_leader=q_ref_leader,
                q_cmd_leader=q_cmd_leader,
                q_cmd_ur5e=q_cmd_ur5e,
                epsilon_leader=epsilon_leader,
                epsilon_ur5e=epsilon_ur5e,
                wrench_sensor_raw=wrench_sensor_raw,
                wrench_base_raw=wrench_base_raw,
                wrench_base=wrench_base,
                task_delta=task_delta,
                task_pose_error=task_pose_error,
                delta_leader=delta_leader,
                delta_leader_raw=delta_leader_raw,
            )

        votes = diag.get("votes", {})
        detector_votes = np.array(
            [
                float(bool(votes.get("torque", False))),
                float(bool(votes.get("delta_q", votes.get("delta", False)))),
                float(bool(votes.get("energy", False))),
                float(bool(votes.get("wrench", False))),
            ],
            dtype=np.float32,
        )

        lerobot_recorder.add_frame(
            timestamp=t_mono,
            q=q_follower,
            dq=dq_follower,
            gripper=gripper_follower,
            action=policy_action,
            tau_ext=tau_ext,
            wrench_ur5e=wrench_ur5e,
            q_ref=policy_action,
            q_compliant=compliant_action[:n],
            dq_compliant=np.zeros(n),
            gripper_ref=float(policy_action[-1]) if len(policy_action) > n else 0.0,
            gripper_compliant=float(compliant_action[-1]) if len(compliant_action) > n else 0.0,
            is_correction=bool(is_corr),
            detector_votes=detector_votes,
            images=images,
        )
        ep_recorded_frames += 1

        if ep_recorded_frames % dataset_fps == 0:
            elapsed_s = frame_start - ep_start
            delta_norm = np.linalg.norm(compliant_action[:n] - policy_action[:n])
            print(
                f"  [ep {ep+1:03d} {elapsed_s:5.1f}s] "
                f"INT={'YES' if is_corr else 'no '} "
                f"|Δ|={delta_norm:.4f} "
                f"frames={ep_recorded_frames}"
            )

        elapsed = time.perf_counter() - frame_start
        if elapsed > frame_dt:
            ep_overruns += 1
        else:
            time.sleep(frame_dt - elapsed)

    return {
        "rerecord": False,
        "success": False,
        "running": True,
        "steps": ep_recorded_frames,
        "overruns": ep_overruns,
        "frames": ep_recorded_frames,
        "corr_steps": corr_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (unchanged)
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


def _load_default_config(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


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
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CR-DAgger production data collection")
    p.add_argument(
        "--interventions",
        action="store_true",
        help="Enable CR-DAgger intervention/correction mode.",
    )
    p.add_argument(
        "--four-channel",
        action="store_true",
        help="Use 4-channel dual-impedance loop instead of standard Phase B.",
    )
    p.add_argument(
        "--mirror-source",
        choices=["cmd", "actual", "policy"],
        default=None,
        help="Override 4-channel mirror source (default: config).",
    )
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V3.yaml")
    p.add_argument(
        "--cr-dagger-defaults",
        type=str,
        default="gello/cr_dagger/config/cr_dagger_defaults.yaml",
        help="YAML file containing validated CR-DAgger defaults, including the Phase-B BOTA-SE3 parameters.",
    )
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=20.0)
    p.add_argument("--policy-type", choices=["dummy_sine", "dummy_hold", "lerobot_act", "act"], default="dummy_hold")
    p.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Path or Hub ID of a LeRobot ACT policy. Required for --policy-type lerobot_act/act.",
    )
    p.add_argument(
        "--policy-device",
        type=str,
        default="cuda",
        help="Torch device used by the LeRobot policy worker.",
    )
    p.add_argument(
        "--policy-dataset-repo",
        type=str,
        default=None,
        help="LeRobot dataset repo used to load policy feature metadata. Defaults to --lerobot-repo.",
    )
    p.add_argument(
        "--policy-dataset-root",
        type=str,
        default=None,
        help="Local root for --policy-dataset-repo metadata. Defaults to --lerobot-root.",
    )
    p.add_argument(
        "--policy-robot-type",
        type=str,
        default="",
        help="Optional robot_type string passed into LeRobot policy preprocessing.",
    )
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--frequency", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--action-dt", type=float, default=0.1)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--obs-decimation", type=int, default=33)

    # ── Foot pedal ────────────────────────────────────────────────────────
    p.add_argument(
        "--foot-pedal-key",
        type=str,
        default="b",
        help=(
            "Key that the USB foot pedal sends. Examples: 'f10', 'b', 'space'. "
            "Pedal starts/stops episodes depending on context. (default: b)"
        ),
    )

    # ── LeRobot dataset ───────────────────────────────────────────────────
    p.add_argument("--lerobot-repo", type=str, required=True)
    p.add_argument("--lerobot-root", type=str, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume recording into an existing LeRobot dataset root.",
    )

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
    p.add_argument("--camera-warmup-s", type=float, default=3.0)
    p.add_argument("--enable-wrench", action="store_true")
    p.add_argument("--enable-wrench-feedback", action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = _parse_args()

    defaults_path = Path(args.cr_dagger_defaults)
    if not defaults_path.is_absolute():
        defaults_path = (REPO_ROOT / defaults_path).resolve()
    defaults_cfg = _load_default_config(defaults_path)
    phase_b_cfg = defaults_cfg.get("phase_b", {}) if isinstance(defaults_cfg.get("phase_b", {}), dict) else {}

    if str(args.policy_type) in ("lerobot_act", "act") and not args.policy_path:
        print("[ERROR] --policy-path is required for --policy-type lerobot_act/act")
        return 2

    try:
        repo_id = _validate_repo_id(args.lerobot_repo)
    except ValueError as e:
        print(e)
        return 1

    if args.lerobot_root is not None:
        dataset_root = Path(args.lerobot_root) / repo_id
    else:
        dataset_root = Path.home() / ".cache" / "lerobot" / "datasets" / repo_id

    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    if dataset_root.exists():
        if args.resume:
            print(f"[DATASET] Resuming existing dataset root: {dataset_root}")
        else:
            raise FileExistsError(
                f"\n[ERROR] Dataset root already exists: {dataset_root}\n"
                f"        Use --resume to append episodes to this dataset, or choose a different"
                f"        --lerobot-repo/--lerobot-root to avoid overwriting existing data."
                f"        rm -rf {dataset_root} to delete the existing dataset (be careful with this command!)"
            )
    elif args.resume:
        print(
            f"[WARN] --resume was requested, but dataset root does not exist: {dataset_root}. "
            f"Creating a new dataset."
        )

    print(f"[DATASET] repo_id  : {repo_id}")
    print(f"[DATASET] local root: {dataset_root}")

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
    npz_session_prefix: str | None = None
    camera_rig: MultiRealSenseRig | None = None
    traj_interp: TrajectoryInterpolator | None = None
    detector: FusedInterventionDetector | None = None
    ctrl: Optional[EpisodeController] = None
    sensor_reader: Optional[AsyncSensorReader] = None
    unified_stop: Event | None = None
    unified_thread: Thread | None = None
    state_cache: dict[str, Any] | None = None
    n = 6
    teleop_started = False
    phase_b_runtime_stopped = False

    def _sig(*_: object) -> None:
        nonlocal running
        running = False
        if ctrl is not None:
            ctrl.stop_requested.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def _stop_phase_b_runtime(reason: str) -> None:
        nonlocal unified_stop, unified_thread, stop_event, policy_proc, phase_b_runtime_stopped
        if not args.interventions or phase_b_runtime_stopped:
            return
        phase_b_runtime_stopped = True
        print(f"[PHASE B] stopping runtime before {reason}")
        if unified_stop is not None:
            unified_stop.set()
        if unified_thread is not None:
            try:
                unified_thread.join(timeout=3.0)
                if unified_thread.is_alive():
                    print("[WARN] Phase-B control thread did not stop within 3s")
            except Exception as exc:
                print(f"[WARN] Phase-B control thread join failed: {exc}")
            unified_thread = None
        try:
            follower = getattr(system, "_direct_follower_robot", None) if system is not None else None
            robot = getattr(follower, "robot", None)
            if robot is not None and hasattr(robot, "stopJ"):
                robot.stopJ(2.0)
                print("[FOLLOWER] stopJ confirmed before saving/finalize")
        except Exception as exc:
            print(f"[WARN] Follower stopJ before saving/finalize failed: {exc}")
        if stop_event is not None:
            stop_event.set()
        if policy_proc is not None:
            try:
                policy_proc.join(timeout=2.0)
                if policy_proc.is_alive():
                    policy_proc.terminate()
                    policy_proc.join(timeout=1.0)
            except Exception as exc:
                print(f"[WARN] Policy worker stop failed: {exc}")
            policy_proc = None


    try:
        # ── Episode controller (keyboard + foot pedal) ────────────────────
        ctrl = EpisodeController(foot_pedal_key=str(args.foot_pedal_key))
        ctrl.start()

        mode_str = "CR-DAgger interventions" if args.interventions else "Phase A baseline recording"
        print(f"[MODE] {mode_str}")

        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        n = int(system.num_arm_joints)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")
        if not system.teleop_enabled:
            print("[WARN] teleop is disabled in config.")
        shi = _build_shi(system, config_path, n)

        if args.interventions:
            q0, _, _, _ = system.get_leader_joint_states()
            traj_buf = SharedTrajectoryBuffer(
                name="cr_dagger_traj",
                horizon=int(args.horizon),
                n_joints=n,
                create=True,
            )
            obs_camera_names = list(args.camera_names or [])
            if not obs_camera_names and args.camera_device_ids:
                obs_camera_names = [f"cam{i}" for i in range(len(args.camera_device_ids))]
            obs_snap = SharedObservationSnapshot(
                name="cr_dagger_obs",
                n_joints=n,
                img_height=480,
                img_width=640,
                camera_names=obs_camera_names,
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
            state_cache = {
                "lock": Lock(),
                "updated": False,
                "t_mono": 0.0,
                "q_leader": np.zeros(n),
                "dq_leader": np.zeros(n),
                "grip": 0.0,
                "grip_raw": 0.0,
                "grip_vel": 0.0,
                "tau_ext": np.zeros(n),
                "q_ref": np.zeros(n),
                "dq_ref": np.zeros(n),
                "is_stale": False,
                "is_correction": False,
                "detector_diag": {},
                "policy_action": np.zeros(n + 1),
                "compliant_action": np.zeros(n + 1),
                "q_follower": np.zeros(n),
                "dq_follower": np.zeros(n),
                "gripper_follower": 0.0,
                "wrench_ur5e": np.zeros(6),
                "wrench_sensor_raw": np.zeros(6),
                "wrench_base_raw": np.zeros(6),
                "wrench_base": np.zeros(6),
                "task_delta": np.zeros(6),
                "task_pose_error": np.zeros(6),
                "delta_leader": np.zeros(n),
                "delta_leader_raw": np.zeros(n),
                "q_cmd_leader": np.zeros(n),
                "tcp_joint_torques": np.zeros(n),
                "delta_human": np.zeros(n),
                "q_cmd_ur5e": np.zeros(n),
                "epsilon_ur5e": np.zeros(n),
                "tau_ext_gated": np.zeros(n),
                "tau_force_feedback": np.zeros(n),
                "q_mirror_target": np.zeros(n),
                "tau_cmd_gello": np.zeros(n),
                "tau_mirror": np.zeros(n),
                "velocity_gate": 1.0,
                "mirror_source": "cmd",
                "phase_b_ready": False,
                "phase_b_error": "",
                "epsilon_leader": np.zeros(n),
                "wrench_bota_raw": np.zeros(6),
                "delta_human_raw": np.zeros(n),
                "intervention_active": 0.0,
                "intervention_source": 0.0,
            }

        camera_names, camera_device_ids, camera_flips = _resolve_camera_config(args)
        if camera_names:
            camera_rig = MultiRealSenseRig(
                camera_names=camera_names,
                camera_device_ids=camera_device_ids,
                camera_flips=camera_flips,
            )
            camera_rig.start()
            print(f"[CAM] started {len(camera_names)} camera(s): {camera_names}")
            time.sleep(max(0.0, float(args.camera_warmup_s)))

        dataset_fps = int(args.dataset_fps)
        if dataset_fps <= 0:
            raise ValueError("dataset-fps must be > 0")

        # Determine whether the follower robot actually exposes a gripper.
        has_gripper = False
        try:
            if getattr(system, "_direct_follower_robot", None) is not None:
                has_gripper = bool(getattr(system._direct_follower_robot, "_use_gripper", False))
            elif getattr(system, "teleop_client", None) is not None:
                try:
                    follower_dofs = system.teleop_client.num_dofs()
                    has_gripper = follower_dofs > n
                except Exception:
                    has_gripper = False
        except Exception:
            has_gripper = False

        lerobot_recorder = LeRobotCorrectionRecorder(
            repo_id=repo_id,
            root=str(dataset_root),
            resume=bool(args.resume),
            fps=dataset_fps,
            task_description=str(args.task_description),
            n_joints=n,
            include_gripper_action=(bool(args.interventions) and has_gripper),
            camera_names=camera_names if camera_names else None,
        )

        if args.interventions and not args.no_npz:
            task_label = _slug_filename_part(args.task_description, "task")
            policy_label = _slug_filename_part(str(args.policy_type).replace("/", "_"), "policy")
            timestamp = int(time.monotonic())
            npz_session_prefix = f"crdagger_phaseB_{policy_label}_{task_label}_{timestamp}"
            bota_cfg = phase_b_cfg.get("bota", {}) if isinstance(phase_b_cfg.get("bota", {}), dict) else {}
            residual_cfg = phase_b_cfg.get("residual", {}) if isinstance(phase_b_cfg.get("residual", {}), dict) else {}
            follower_cfg = phase_b_cfg.get("follower", {}) if isinstance(phase_b_cfg.get("follower", {}), dict) else {}
            intervention_cfg = phase_b_cfg.get("intervention", {}) if isinstance(phase_b_cfg.get("intervention", {}), dict) else {}
            npz_metadata = {
                "mode": "phase_b_intervention_collection",
                "architecture": "bota_se3_residual" if not bool(args.four_channel) else "four_channel",
                "leader_command_mode": "position",
                "wrench_source": "bota",
                "repo_id": repo_id,
                "dataset_root": str(dataset_root),
                "task_description": str(args.task_description),
                "policy_type": str(args.policy_type),
                "policy_path": str(args.policy_path or ""),
                "policy_output_frame": "follower" if str(args.policy_type) in ("lerobot_act", "act") else "leader",
                "dataset_fps": int(dataset_fps),
                "max_episode_duration": float(args.max_episode_duration),
                "num_episodes": int(args.num_episodes),
                "camera_names": list(camera_names),
                "camera_device_ids": list(camera_device_ids),
                "config": str(config_path),
                "cr_dagger_defaults": str(defaults_path),
                "bota_config": str((REPO_ROOT / str(bota_cfg.get("config", "configs/bota_binary.json"))).resolve()),
                "bota_axis_mask": bota_cfg.get("axis_mask", []),
                "bota_cart_mass": bota_cfg.get("cart_mass", []),
                "bota_cart_damp": bota_cfg.get("cart_damp", []),
                "bota_cart_stiff": bota_cfg.get("cart_stiff", []),
                "bota_cart_max": bota_cfg.get("cart_max", []),
                "bota_dls_damping": float(bota_cfg.get("dls_damping", 0.10)),
                "bota_filter_cutoff": float(bota_cfg.get("filter_cutoff_hz", 25.0)),
                "bota_deadband": bota_cfg.get("deadband", []),
                "bota_saturation": bota_cfg.get("saturation", []),
                "bota_wrench_sign": float(bota_cfg.get("wrench_sign", 1.0)),
                "bota_se3_output_mode": str(bota_cfg.get("output_mode", "offset")),
                "bota_contact_force_scale": float(bota_cfg.get("contact_force_scale", 8.0)),
                "bota_contact_torque_scale": float(bota_cfg.get("contact_torque_scale", 0.35)),
                "adm_delta_max": residual_cfg.get("delta_max", 0.0),
                "adm_delta_rate_max": residual_cfg.get("delta_rate_max", []),
                "adm_delta_release_rate_max": residual_cfg.get("delta_release_rate_max", []),
                "adm_delta_release_tau": float(residual_cfg.get("delta_release_tau_s", 0.0)),
                "adm_delta_release_contact_threshold": float(residual_cfg.get("delta_release_contact_threshold", 0.0)),
                "intervention_contact_threshold": float(intervention_cfg.get("contact_threshold", 0.25)),
                "intervention_delta_threshold": float(intervention_cfg.get("delta_threshold", 0.01)),
                "intervention_source_encoding": "0=none,1=contact,2=delta,3=both",
                "follower_kp": float(follower_cfg.get("kp", 190.0)),
                "follower_kd": float(follower_cfg.get("kd", 15.0)),
                "action_dt": float(args.action_dt),
                "horizon": int(args.horizon),
                "latency_compensation_s": 0.008,
            }
            npz_recorder = CorrectionRecorder(
                n_joints=n,
                log_dir=str(args.log_dir),
                latency_compensation_s=0.008,
                metadata=npz_metadata,
            )
            print(f"[NPZ] episode files: {args.log_dir}/{npz_session_prefix}_epXXXX.npz")

        shi.reset()

        if args.interventions:
            q_now, _, _, _ = system.get_leader_joint_states()
            system.driver.set_torque_mode(False)
            time.sleep(0.05)
            system.driver.set_operating_mode(0)
            time.sleep(0.05)
            system.driver.set_torque_mode(True)
            time.sleep(0.05)

            stop_event = mp.Event()
            policy_output_frame = (
                "follower" if str(args.policy_type) in ("lerobot_act", "act") else "leader"
            )
            policy_config = {
                "center": q_now.tolist(),
                "amplitude": [float(args.amplitude)] * n,
                "frequency": [float(args.frequency)] * n,
                "q_hold": q_now.tolist(),
                # Dummy policies publish leader-frame q references. LeRobot ACT
                # policies trained on Phase-A data publish follower-frame actions.
                "output_frame": policy_output_frame,
                "policy_path": args.policy_path,
                "device": str(args.policy_device),
                "task": str(args.task_description),
                "robot_type": str(args.policy_robot_type),
                "camera_names": camera_names if camera_names else list(args.camera_names or []),
                "dataset_features": lerobot_recorder.features,
                "dataset_repo": str(args.policy_dataset_repo or repo_id),
                "dataset_root": str(args.policy_dataset_root or args.lerobot_root or ""),
            }
            if getattr(system, "map_index", None) is not None:
                policy_config["map_index"] = np.asarray(system.map_index, dtype=int).tolist()
            if getattr(system, "map_signs", None) is not None:
                policy_config["map_signs"] = np.asarray(system.map_signs, dtype=float).tolist()
            if getattr(system, "map_offsets", None) is not None:
                policy_config["map_offsets"] = np.asarray(system.map_offsets, dtype=float).tolist()
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
        record_interval = 1.0 / float(dataset_fps)  # Time-based recording
        print(
            f"[DATASET] control_hz={control_hz:.1f}, dataset_fps={dataset_fps}, "
            f"record_interval={record_interval:.6f}s"
        )

        if args.interventions:
            if hasattr(system.driver, "stop_reading"):
                system.driver.stop_reading()

            if state_cache is None:
                raise RuntimeError("Phase B state cache was not initialized")

            unified_stop = Event()
            # The validated production Phase-B path is BOTA-SE3 residual control.
            # The legacy 4-channel loop is kept only as an explicit debugging option.
            use_4ch = bool(args.four_channel)
            if use_4ch:
                if args.mirror_source:
                    four_ch_cfg = system.config.setdefault("teleop", {}).setdefault("four_channel", {})
                    four_ch_cfg["mirror_source"] = str(args.mirror_source)
                unified_thread = Thread(
                    target=system._phase_b_4channel_loop,
                    args=(
                        traj_interp,
                        detector,
                        state_cache,
                        unified_stop,
                        bool(args.enable_wrench or args.enable_wrench_feedback),
                    ),
                    daemon=True,
                    name="phase-b-4channel",
                )
            else:
                unified_thread = Thread(
                    target=run_phase_b_bota_se3_loop,
                    args=(
                        system,
                        traj_interp,
                        state_cache,
                        unified_stop,
                        phase_b_cfg,
                        REPO_ROOT,
                    ),
                    daemon=True,
                    name="phase-b-bota-se3",
                )
            unified_thread.start()
            if use_4ch:
                print("[PHASE B] 4-channel control thread started")
            else:
                print("[PHASE B] BOTA-SE3 residual control thread started")
                t_ready = time.perf_counter()
                while time.perf_counter() - t_ready < 30.0:
                    with state_cache["lock"]:
                        ready = bool(state_cache.get("phase_b_ready", False))
                        error = str(state_cache.get("phase_b_error", ""))
                    if ready:
                        break
                    if error:
                        raise RuntimeError(f"Phase B BOTA-SE3 loop failed during startup: {error}")
                    time.sleep(0.05)
                else:
                    raise TimeoutError("Phase B BOTA-SE3 loop did not arm within 30s")
        elif system.teleop_enabled and not teleop_started:
            teleop_started = _start_prepared_teleop(system)
            if not teleop_started:
                raise RuntimeError("teleop could not be started")

        total_overruns = 0
        total_steps = 0
        ep = 0
        num_episodes = int(args.num_episodes)

        # ══════════════════════════════════════════════════════════════════
        # EPISODE LOOP with State Machine
        # ══════════════════════════════════════════════════════════════════

        while ep < num_episodes and running:

            # ── IDLE STATE: Wait for user to start ────────────────────────
            ctrl.state = EpisodeState.IDLE
            ctrl.reset_events()

            print(
                f"\n{'=' * 50}\n"
                f"  Episode {ep + 1}/{num_episodes} — READY\n"
                f"  Press foot pedal or Enter to start recording.\n"
                f"  Press → to end collection.\n"
                f"{'=' * 50}"
            )

            if not ctrl.wait_for_start():
                # stop_requested was set
                print("[CTRL] Collection stopped by user.")
                break

            if not running:
                break

            # ── RECORDING STATE ───────────────────────────────────────────
            ctrl.state = EpisodeState.RECORDING
            ctrl.reset_events()

            if args.interventions:
                assert traj_interp is not None
                assert detector is not None
                t_mono = time.monotonic()
                q_ref_now, _, _ = traj_interp.get_reference(t_mono)
                detector.reset()

            lerobot_recorder.start_episode(task_description=str(args.task_description))
            if npz_recorder is not None:
                episode_npz_id = (
                    f"{npz_session_prefix}_ep{ep:04d}"
                    if npz_session_prefix
                    else f"episode_{ep:04d}"
                )
                npz_recorder.start_episode(episode_npz_id)

            ep_start = time.perf_counter()
            ep_steps = 0
            ep_overruns = 0
            corr_steps = 0
            ep_recorded_frames = 0
            rerecord = False
            marked_success = False

            print(f"[REC] ● Recording episode {ep + 1}...")

            if not args.interventions:
                result = _run_recording_loop_phase_a(
                    system=system,
                    ctrl=ctrl,
                    lerobot_recorder=lerobot_recorder,
                    camera_rig=camera_rig,
                    args=args,
                    ep=ep,
                    n=n,
                )
            else:
                result = _run_recording_loop_phase_b(
                    ctrl=ctrl,
                    lerobot_recorder=lerobot_recorder,
                    camera_rig=camera_rig,
                    state_cache=state_cache if state_cache is not None else {"lock": Lock(), "updated": False},
                    args=args,
                    ep=ep,
                    n=n,
                    obs_snap=obs_snap,
                    npz_recorder=npz_recorder,
                )

            rerecord = result["rerecord"]
            marked_success = result["success"]
            if not result["running"]:
                running = False
            ep_steps = result["steps"]
            ep_overruns = result["overruns"]
            ep_recorded_frames = result["frames"]
            corr_steps = result.get("corr_steps", 0)
            total_overruns += ep_overruns
            total_steps += ep_steps

            # ══════════════════════════════════════════════════════════════
            # POST-EPISODE: Discard or Save, then RESET phase
            # ══════════════════════════════════════════════════════════════

            if rerecord:
                _discard_episode(lerobot_recorder, npz_recorder)
                print(f"[DISCARD] Episode {ep + 1} discarded — will re-record.\n")

                # ── RESET STATE (even after discard) ──────────────────────
                # Teleop stays active so user can reposition!
                ctrl.state = EpisodeState.RESET
                ctrl.reset_events()
                print(
                    "  [RESET] Teleop still active — reposition robot.\n"
                    "  Press foot pedal or Enter when ready to re-record."
                )
                # Don't wait here; the IDLE wait at loop top handles it.
                # But we DO stay here briefly so user sees the message.
                if not running:
                    break
                continue  # ep NOT incremented

            # ── Save episode ──────────────────────────────────────────────
            final_episode = (ep >= num_episodes - 1) or (not running)
            if final_episode:
                _stop_phase_b_runtime("final episode save")

            if ep_recorded_frames > 0:
                ep_idx = lerobot_recorder.end_episode()
            else:
                ep_idx = -1
                print("[WARN] No frames recorded; skipping save.")
            if npz_recorder is not None:
                npz_recorder.end_episode()

            duration_s = time.perf_counter() - ep_start
            end_reason = (
                "SUCCESS ✓"
                if marked_success
                else "STOP (user)"
                if not running
                else "time limit"
            )
            print(f"\n  Episode {ep_idx} complete ({end_reason}):")
            print(f"    Steps:            {ep_steps}")
            print(f"    Duration:         {duration_s:.1f}s")
            print(f"    Dataset frames:   {ep_recorded_frames}")
            print(
                f"    Correction steps: {corr_steps} "
                f"({100.0 * corr_steps / max(ep_steps, 1):.1f}%)"
            )
            print(
                f"    Timing overruns:  {ep_overruns}/{ep_steps} "
                f"({100.0 * ep_overruns / max(ep_steps, 1):.1f}%)"
            )

            ep += 1

            # ── RESET STATE: teleop active, user repositions ──────────────
            if ep < num_episodes and running:
                ctrl.state = EpisodeState.RESET
                ctrl.reset_events()
                print(
                    f"\n  [RESET] Teleop still active — reposition for next episode.\n"
                    f"  Press foot pedal or Enter when ready."
                )
                # The wait_for_start() at the top of the while loop handles
                # the actual waiting. We just set the state here.

        # ── Finalize dataset ──────────────────────────────────────────────
        _stop_phase_b_runtime("dataset finalize")
        local_path = lerobot_recorder.finalize()
        print(f"\nDataset finalized at: {local_path}")
        print(f"  repo_id   : {repo_id}")
        print(f"  local root: {dataset_root}")
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
        try:
            _stop_phase_b_runtime("process shutdown")
        except Exception:
            pass

        if ctrl is not None:
            ctrl.stop()

        if system is not None:
            system.running = False

        if unified_stop is not None:
            unified_stop.set()
        if unified_thread is not None:
            try:
                unified_thread.join(timeout=1.0)
            except Exception:
                pass

        if sensor_reader is not None:
            try:
                sensor_reader.stop()
            except Exception:
                pass

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