#!/usr/bin/env python3
"""Admittance Controller Testing Script on real GELLO hardware.

Records a human-guided trajectory in gravity compensation,
then replays it using either joint-space or task-space admittance control.

The admittance loop runs on top of Dynamixel position-mode tracking:
  1. Shi observer reads motor current → estimates τ_ext
  2. Admittance dynamics integrate τ_ext → compliant reference q_c
  3. q_c is sent as a position target to the Dynamixel hardware

Phases:
  1. PREP   — Settle in gravity comp
  2. RECORD — User moves arm, trajectory is recorded
  3. HOME   — PD control moves arm back to trajectory start
  4. REPLAY — Admittance control tracks the recorded trajectory

"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pinocchio as pin
except ImportError:
    pin = None

from gello.factr.gravity_compensation import FACTRGravityCompensation  # noqa: E402
from gello.bilat_4ch.gello_ur5e_observer_shi import (  # noqa: E402
    MinimalistEstimatorConfig,
    MinimalistTorqueEstimator,
    MotorParams,
    MotorType,
)


# ── helpers ──────────────────────────────────────────────────────────

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


def _init_plot(mode: str, window_s: float):
    if plt is None:
        return None
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 1.2,
    })
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Admittance Replay ({mode.capitalize()} Space)", fontweight="bold")

    n_dims = 6 if mode == "joint" else 3
    colors = plt.cm.tab10(np.linspace(0, 1, n_dims))
    lines_ref, lines_act = [], []
    for i in range(n_dims):
        lbl = f"Joint {i+1}" if mode == "joint" else ["X", "Y", "Z"][i]
        lr, = ax.plot([], [], "--", lw=1.5, alpha=0.5, color=colors[i], label=f"{lbl} Ref")
        la, = ax.plot([], [], "-",  lw=1.5,            color=colors[i], label=f"{lbl} Act")
        lines_ref.append(lr)
        lines_act.append(la)

    ax.set_ylabel("Joint Angle (rad)" if mode == "joint" else "TCP Position (m)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    fig.tight_layout()
    return {"fig": fig, "ax": ax, "lines_ref": lines_ref, "lines_act": lines_act}


def _update_plot(ps, t_hist, ref_hist, act_hist):
    if ps is None or t_hist.size == 0:
        return
    tx = t_hist - t_hist[0]
    for i in range(ref_hist.shape[1]):
        ps["lines_ref"][i].set_data(tx, ref_hist[:, i])
        ps["lines_act"][i].set_data(tx, act_hist[:, i])
    ps["ax"].set_xlim(tx[0], max(tx[-1], 1e-3))
    ps["ax"].relim()
    ps["ax"].autoscale_view(scaley=True)
    ps["fig"].canvas.draw_idle()
    ps["fig"].canvas.flush_events()


def compute_task_kinematics(system: FACTRGravityCompensation, q: np.ndarray):
    if pin is None:
        raise RuntimeError("pinocchio required for task-space mode")
    n = system.num_arm_joints
    q_full = np.zeros(system._pin_nq)
    q_full[: min(len(q), system._pin_nq)] = q[: min(len(q), system._pin_nq)]
    pin.forwardKinematics(system.pin_model, system.pin_data, q_full)
    pin.computeJointJacobians(system.pin_model, system.pin_data, q_full)
    pin.updateFramePlacements(system.pin_model, system.pin_data)
    ee_id = system.pin_model.nframes - 1
    x = np.array(system.pin_data.oMf[ee_id].translation)
    J_full = pin.getFrameJacobian(
        system.pin_model, system.pin_data, ee_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    return x, J_full[:3, :n]


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record & Replay via Admittance Control")
    p.add_argument("--config", type=str,
                   default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--mode", choices=["joint", "task"], default="joint")

    p.add_argument("--prep-time",   type=float, default=3.0)
    p.add_argument("--record-time", type=float, default=5.0)
    p.add_argument("--home-time",   type=float, default=4.0)

    # Low-level position PD (used during HOME phase only)
    p.add_argument("--kp-low", type=float, default=30.0)
    p.add_argument("--kd-low", type=float, default=2.0)

    # Admittance parameters — joint space
    p.add_argument("--mass-j",  type=float, default=1.0)
    p.add_argument("--damp-j",  type=float, default=5.0)
    p.add_argument("--stiff-j", type=float, default=20.0)

    # Admittance parameters — task space
    p.add_argument("--mass-t",  type=float, default=2.0)
    p.add_argument("--damp-t",  type=float, default=20.0)
    p.add_argument("--stiff-t", type=float, default=100.0)

    p.add_argument("--plot-rate", type=float, default=10.0)
    p.add_argument("--no-plot",   action="store_true")
    return p.parse_args()


# ── main loop ────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    print("=" * 70)
    print(f"Trajectory Record & Replay — ADMITTANCE [{args.mode.upper()} SPACE]")
    print(f"Config: {config_path}")
    print("=" * 70)

    system: Optional[FACTRGravityCompensation] = None
    n = 6  # fallback for cleanup
    running = True

    def _sig(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        system = FACTRGravityCompensation(str(config_path),
                                          enable_visualization=False)
        n = int(system.num_arm_joints)

        # ── Shi observer (current-based, works in position mode) ─────
        leader_urdf = str(system.config["arm_teleop"]["leader_urdf"])
        urdf_path = _resolve_leader_urdf(config_path, leader_urdf)
        servo_types = list(system.config["dynamixel"]["servo_types"])
        gear_ratio, kt = _infer_motor_params(servo_types, n)

        shi = MinimalistTorqueEstimator(
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

        # ── phase bookkeeping ────────────────────────────────────────
        t_start = time.time()
        last_step_t = time.perf_counter()

        phase = "PREP"
        trajectory_q: List[Tuple[float, np.ndarray, np.ndarray]] = []

        print(f"[{phase}] Settle in gravity comp for {args.prep_time}s …")

        plot_state = None if args.no_plot else _init_plot(args.mode,
                                                          args.record_time)
        plot_period = 1.0 / max(args.plot_rate, 1e-3)
        last_plot_t = time.time()

        t_buf   = deque(maxlen=int(args.record_time * 500))
        ref_buf = deque(maxlen=int(args.record_time * 500))
        act_buf = deque(maxlen=int(args.record_time * 500))

        # Variables set during phase transitions
        record_start_t = 0.0
        home_start_t   = 0.0
        home_start_q   = np.zeros(n)
        target_q       = np.zeros(n)
        replay_start_t = 0.0

        # Admittance state (initialised properly when entering REPLAY)
        q_c  = np.zeros(n)
        dq_c = np.zeros(n)
        x_c  = np.zeros(3)
        dx_c = np.zeros(3)

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf
            t_now = time.time() - t_start

            q, dq, _, _ = system.get_leader_joint_states()

            # Motor currents (needed for Shi observer in every phase)
            currents_all = (system.driver.get_currents()
                           if system.driver is not None
                           else np.zeros(system.num_motors))
            currents_arm = currents_all[:n] * system.joint_signs[:n]

            # Shi observer — runs every iteration so it stays warm
            tau_ext = shi.update(q, dq, currents_arm)

            # ── PHASE: PREP ──────────────────────────────────────────
            if phase == "PREP":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_cmd = tau_grav + tau_fric + tau_damp
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_now >= args.prep_time:
                    phase = "RECORD"
                    record_start_t = t_now
                    print(f"[{phase}] Recording for {args.record_time}s "
                          "— MOVE THE ARM NOW.")

            # ── PHASE: RECORD ────────────────────────────────────────
            elif phase == "RECORD":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_cmd = tau_grav + tau_fric + tau_damp
                system.set_leader_joint_torque(tau_cmd, 0.0)

                t_rel = t_now - record_start_t
                trajectory_q.append((t_rel, q.copy(), dq.copy()))

                if t_rel >= args.record_time:
                    phase = "HOME"
                    home_start_t = t_now
                    home_start_q = q.copy()
                    target_q = trajectory_q[0][1]
                    print(f"[{phase}] Returning to start for "
                          f"{args.home_time}s …")

            # ── PHASE: HOME ──────────────────────────────────────────
            elif phase == "HOME":
                t_rel = t_now - home_start_t
                alpha_t = np.clip(t_rel / args.home_time, 0.0, 1.0)
                # Minimum-jerk profile
                s = 10*alpha_t**3 - 15*alpha_t**4 + 6*alpha_t**5
                des_q = home_start_q + s * (target_q - home_start_q)

                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_pd   = 20.0 * (des_q - q) - 2.0 * dq
                tau_cmd  = tau_grav + tau_fric + tau_pd
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_rel >= args.home_time + 1.0:
                    # -- transition to REPLAY --
                    phase = "REPLAY"
                    print(f"[{phase}] Switching to POSITION MODE "
                          "for admittance tracking.")

                    # Switch Dynamixel to position mode
                    system.driver.set_torque_mode(False)
                    time.sleep(0.05)
                    system.driver.set_operating_mode(3)
                    time.sleep(0.05)
                    system.driver.set_torque_mode(True)
                    time.sleep(0.05)

                    replay_start_t = time.time() - t_start

                    # Reset observer so EMA starts clean in position mode
                    shi.reset()

                    # Initialise admittance state at trajectory start
                    q_c  = trajectory_q[0][1].copy()
                    dq_c = np.zeros(n)
                    if args.mode == "task":
                        x_c, _ = compute_task_kinematics(system, q_c)
                        dx_c = np.zeros(3)

            # ── PHASE: REPLAY (admittance loop) ──────────────────────
            elif phase == "REPLAY":
                t_rel = t_now - replay_start_t

                if t_rel > trajectory_q[-1][0]:
                    print("Replay finished — holding position. "
                          "Ctrl+C to exit.")
                    running = False
                    continue

                # Look up reference from recorded trajectory
                idx = np.searchsorted(
                    [p[0] for p in trajectory_q], t_rel
                )
                idx = min(idx, len(trajectory_q) - 1)
                _, q_ref, dq_ref = trajectory_q[idx]

                # --- Joint-Space Admittance ---
                if args.mode == "joint":
                    ddq_c = (
                        tau_ext
                        - args.damp_j  * dq_c
                        - args.stiff_j * (q_c - q_ref)

                    ) / args.mass_j

                    dq_c = dq_c + ddq_c * measured_dt
                    q_c  = q_c  + dq_c  * measured_dt

                    # Safety clamp — stay within joint limits
                    q_c = np.clip(
                        q_c,
                        system.arm_joint_limits_min,
                        system.arm_joint_limits_max,
                    )

                    t_buf.append(t_rel)
                    ref_buf.append(q_ref.copy())
                    act_buf.append(q.copy())

                # --- Task-Space Admittance ---
                elif args.mode == "task":
                    x_ref, _     = compute_task_kinematics(system, q_ref)
                    x_act, J_act = compute_task_kinematics(system, q)

                    # τ_ext → F_ext  via  (J^T)^+
                    F_ext = np.linalg.pinv(J_act.T) @ tau_ext

                    ddx_c = (
                        F_ext
                        - args.damp_t  * dx_c
                        - args.stiff_t * (x_c - x_ref)

                    ) / args.mass_t

                    dx_c = dx_c + ddx_c * measured_dt
                    x_c  = x_c  + dx_c  * measured_dt

                    # Differential IK  dq = J^+ dx
                    dq_c_des = np.linalg.pinv(J_act) @ dx_c
                    q_c = q_c + dq_c_des * measured_dt

                    q_c = np.clip(
                        q_c,
                        system.arm_joint_limits_min,
                        system.arm_joint_limits_max,
                    )

                    t_buf.append(t_rel)
                    ref_buf.append(x_ref.copy())
                    act_buf.append(x_act.copy())

                # --- Send position command to Dynamixel hardware ---
                # Inverse of get_leader_joint_states() mapping:
                #   user = (hw - offset) * sign  ⟹  hw = user * sign + offset
                n_motors = system.num_motors  # FIX: was system.joint_names
                target_hw = np.zeros(n_motors)
                target_hw[:n] = (
                    q_c * system.joint_signs[:n]
                    + system.joint_offsets[:n]

                )
                # Keep gripper at its current raw position
                if n_motors > n:
                    target_hw[-1] = system.leader_gripper_raw_rad
                system.driver.set_joints(target_hw.tolist())

                # --- Live plot ---
                wall_now = time.time()
                if (not args.no_plot
                        and plot_state is not None
                        and wall_now - last_plot_t >= plot_period):
                    _update_plot(
                        plot_state,
                        np.asarray(t_buf, dtype=float),
                        np.asarray(ref_buf, dtype=float),
                        np.asarray(act_buf, dtype=float),
                    )
                    last_plot_t = wall_now

            # ── loop timing ──────────────────────────────────────────
            sleep_s = max(0.0,
                          float(system.dt) - (time.perf_counter() - now_perf))
            if sleep_s > 0:
                time.sleep(sleep_s)

        print("Finished.")
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Test failed: {exc}")
        import traceback; traceback.print_exc()
        return 1
    finally:
        if system is not None:
            try:
                system.set_leader_joint_torque(np.zeros(n), 0.0)
                time.sleep(0.05)
                system.driver.set_operating_mode(3)
                time.sleep(0.05)
                system.shutdown()
            except Exception:
                pass

        if plt is not None:
            if "plot_state" in dir() and plot_state is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"trajectory_admittance_{args.mode}_{ts}.svg"
                plot_state["fig"].savefig(path, format="svg",
                                          bbox_inches="tight")
                print(f"\nSaved plot → {path}")
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())