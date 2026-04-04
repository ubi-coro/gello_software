#!/usr/bin/env python3
"""Impedance Controller - Record & Replay on GELLO hardware.

Records a human-guided trajectory in gravity compensation,
then replays it using joint-space or task-space impedance control.

All phases run in current control mode (no mode switch needed).

Phases:
  1. PREP    - Settle in gravity comp
  2. RECORD  - User moves arm, trajectory is recorded
  3. HOME    - Smooth return to trajectory start (PD + pure gravity)
  4. SETTLE  - Hold at start position (1s)
  5. REPLAY  - Impedance tracking of recorded trajectory
  6. HOLD    - Hold final position until Ctrl+C
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

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


def compute_task_kinematics(system: FACTRGravityCompensation, q: np.ndarray):
    """Compute EE position and linear velocity Jacobian."""
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
    j_full = pin.getFrameJacobian(
        system.pin_model,
        system.pin_data,
        ee_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    return x, j_full[:3, :n]


def _init_plot(mode: str):
    """Create publication-quality live plot."""
    if plt is None:
        return None

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.3,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "legend.fontsize": 9,
        }
    )
    plt.ion()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    n_dims = 6 if mode == "joint" else 3
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:n_dims]
    lines_ref, lines_act, lines_err = [], [], []

    for i in range(n_dims):
        lbl = f"J{i+1}" if mode == "joint" else ["X", "Y", "Z"][i]
        lr, = axes[0].plot([], [], "--", lw=1.2, alpha=0.5, color=colors[i], label=f"{lbl} ref")
        la, = axes[0].plot([], [], "-", lw=1.3, color=colors[i], label=f"{lbl} act")
        le, = axes[1].plot([], [], "-", lw=1.0, color=colors[i], label=lbl)
        lines_ref.append(lr)
        lines_act.append(la)
        lines_err.append(le)

    axes[0].set_ylabel("Joint Angle (rad)" if mode == "joint" else "TCP Position (m)")
    axes[0].set_title("Impedance Replay: Reference vs. Actual", loc="left")
    axes[0].legend(ncol=6, loc="upper right", framealpha=0.85, edgecolor="none")
    axes[0].grid(True, ls="--")

    axes[1].set_ylabel("Tracking Error (rad)" if mode == "joint" else "Error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Tracking Error", loc="left")
    axes[1].legend(ncol=6, loc="upper right", framealpha=0.85, edgecolor="none")
    axes[1].grid(True, ls="--")

    fig.tight_layout()
    return {
        "fig": fig,
        "axes": axes,
        "lines_ref": lines_ref,
        "lines_act": lines_act,
        "lines_err": lines_err,
    }


def _update_plot(plot_state, t: np.ndarray, refs: np.ndarray, acts: np.ndarray):
    if plot_state is None or t.size == 0:
        return

    tx = t - t[0]
    errs = acts - refs

    for i in range(refs.shape[1]):
        plot_state["lines_ref"][i].set_data(tx, refs[:, i])
        plot_state["lines_act"][i].set_data(tx, acts[:, i])
        plot_state["lines_err"][i].set_data(tx, errs[:, i])

    for ax in plot_state["axes"]:
        ax.set_xlim(tx[0], max(tx[-1], 0.1))
        ax.relim()
        ax.autoscale_view(scaley=True)

    plot_state["fig"].canvas.draw_idle()
    plot_state["fig"].canvas.flush_events()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record & Replay via Impedance Control")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--mode", choices=["joint", "task"], default="joint")

    p.add_argument("--prep-time", type=float, default=3.0)
    p.add_argument("--record-time", type=float, default=5.0)
    p.add_argument("--home-time", type=float, default=5.0)

    p.add_argument("--kp-j", type=float, default=15.0, help="Joint Kp (Nm/rad). Lower = more compliant.")
    p.add_argument("--kd-j", type=float, default=3.0, help="Joint Kd (Nm*s/rad)")

    p.add_argument("--kp-t", type=float, default=150.0)
    p.add_argument("--kd-t", type=float, default=10.0)
    p.add_argument("--kd-null", type=float, default=0.5)

    p.add_argument("--home-kp", type=float, default=5.0)
    p.add_argument("--home-kd", type=float, default=2.0)

    p.add_argument("--plot-rate", type=float, default=10.0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    print("=" * 70)
    print(f"Trajectory Record & Replay - IMPEDANCE [{args.mode.upper()} SPACE]")
    print(f"Config: {config_path}")
    print("=" * 70)

    system: Optional[FACTRGravityCompensation] = None
    n = 6
    running = True

    plot_state = None
    t_buf: deque = deque()
    ref_buf: deque = deque()
    act_buf: deque = deque()

    def _sig(signum, frame):
        del signum, frame
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        if system.driver is None:
            raise RuntimeError("Driver not initialized")
        n = int(system.num_arm_joints)

        servo_scale = np.array([0.8, 1.0, 1.0, 0.5, 0.5, 0.3], dtype=float)[:n]
        kp = args.kp_j * servo_scale
        kd = args.kd_j * servo_scale
        home_kp = args.home_kp * servo_scale
        home_kd = args.home_kd * servo_scale
        tau_max = np.array([0.8, 1.5, 1.5, 0.5, 0.5, 0.3], dtype=float)[:n]

        print(f"Replay Kp: {[f'{x:.1f}' for x in kp]}")
        print(f"Replay Kd: {[f'{x:.1f}' for x in kd]}")
        print(f"Torque limits: {[f'{x:.1f}' for x in tau_max]} Nm")

        t_start = time.time()
        last_step_t = time.perf_counter()

        phase = "PREP"
        trajectory_q: List[Tuple[float, np.ndarray, np.ndarray]] = []

        record_start_t = 0.0
        home_start_t = 0.0
        settle_start_t = 0.0
        replay_start_t = 0.0
        home_start_q = np.zeros(n)
        target_q = np.zeros(n)
        hold_q = np.zeros(n)
        debug_counter = 0

        plot_state = None if args.no_plot else _init_plot(args.mode)
        plot_period = 1.0 / max(args.plot_rate, 1e-3)
        last_plot_t = time.time()

        t_buf = deque(maxlen=10000)
        ref_buf = deque(maxlen=10000)
        act_buf = deque(maxlen=10000)

        print(f"[PREP] Settle in gravity comp for {args.prep_time}s ...")

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf
            t_now = time.time() - t_start

            q, dq, _, _ = system.get_leader_joint_states()

            if phase == "PREP":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_cmd = tau_grav + tau_fric + tau_damp
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_now >= args.prep_time:
                    phase = "RECORD"
                    record_start_t = t_now
                    print(f"[RECORD] Recording for {args.record_time}s - MOVE THE ARM NOW.")

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
                    target_q = trajectory_q[0][1].copy()

                    # Ensure HOME interpolation takes the shortest angular path.
                    for i in range(n):
                        delta = target_q[i] - home_start_q[i]
                        while delta > np.pi:
                            target_q[i] -= 2.0 * np.pi
                            delta -= 2.0 * np.pi
                        while delta < -np.pi:
                            target_q[i] += 2.0 * np.pi
                            delta += 2.0 * np.pi

                    print(f"[HOME] Returning to start ({args.home_time}s)")

            elif phase == "HOME":
                t_rel = t_now - home_start_t
                alpha_t = np.clip(t_rel / args.home_time, 0.0, 1.0)
                s = 10 * alpha_t**3 - 15 * alpha_t**4 + 6 * alpha_t**5
                des_q = home_start_q + s * (target_q - home_start_q)

                tau_grav = system.gravity_compensation(q, np.zeros(n))
                tau_pd = home_kp * (des_q - q) - home_kd * dq
                tau_pd = np.clip(tau_pd, -tau_max, tau_max)
                tau_cmd = tau_grav + tau_pd
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_rel >= args.home_time:
                    phase = "SETTLE"
                    settle_start_t = t_now
                    print("[SETTLE] Holding start position (1s) ...")

            elif phase == "SETTLE":
                t_rel = t_now - settle_start_t
                tau_grav = system.gravity_compensation(q, np.zeros(n))
                tau_pd = home_kp * (target_q - q) - home_kd * dq
                tau_pd = np.clip(tau_pd, -tau_max, tau_max)
                tau_cmd = tau_grav + tau_pd
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_rel >= 1.0:
                    phase = "REPLAY"
                    replay_start_t = time.time() - t_start
                    debug_counter = 0
                    print(f"[REPLAY] Impedance tracking [{args.mode.upper()}] active.")

            elif phase == "REPLAY":
                t_rel = t_now - replay_start_t

                if t_rel > trajectory_q[-1][0]:
                    print("Replay finished - holding. Ctrl+C to exit.")
                    phase = "HOLD"
                    hold_q = trajectory_q[-1][1].copy()
                    continue

                idx = np.searchsorted([p[0] for p in trajectory_q], t_rel)
                idx = min(idx, len(trajectory_q) - 1)
                _, q_ref, dq_ref = trajectory_q[idx]

                tau_grav = system.gravity_compensation(q, np.zeros(n))

                if args.mode == "joint":
                    # Wrap replay reference to the nearest turn relative to q.
                    q_ref_near = q_ref.copy()
                    for i in range(n):
                        while q_ref_near[i] - q[i] > np.pi:
                            q_ref_near[i] -= 2.0 * np.pi
                        while q_ref_near[i] - q[i] < -np.pi:
                            q_ref_near[i] += 2.0 * np.pi

                    tau_pd = kp * (q_ref_near - q) + kd * (dq_ref - dq)
                    tau_pd = np.clip(tau_pd, -tau_max, tau_max)
                    tau_cmd = tau_grav + tau_pd
                    system.set_leader_joint_torque(tau_cmd, 0.0)

                    t_buf.append(t_rel)
                    ref_buf.append(q_ref_near.copy())
                    act_buf.append(q.copy())

                else:
                    x_ref, j_ref = compute_task_kinematics(system, q_ref)
                    x_act, j_act = compute_task_kinematics(system, q)
                    dx_ref = j_ref @ dq_ref
                    dx_act = j_act @ dq

                    f_task = args.kp_t * (x_ref - x_act) + args.kd_t * (dx_ref - dx_act)
                    tau_task = j_act.T @ f_task
                    null_proj = np.eye(n) - np.linalg.pinv(j_act) @ j_act
                    tau_null = null_proj @ (-args.kd_null * dq)

                    tau_imp = tau_task + tau_null
                    tau_imp = np.clip(tau_imp, -tau_max, tau_max)

                    tau_cmd = tau_grav + tau_imp
                    system.set_leader_joint_torque(tau_cmd, 0.0)

                    t_buf.append(t_rel)
                    ref_buf.append(x_ref.copy())
                    act_buf.append(x_act.copy())

                debug_counter += 1
                if debug_counter % 30 == 1:
                    if args.mode == "joint":
                        err_terms = q - q_ref_near
                        tau_print = np.clip(
                            kp * (q_ref_near - q) + kd * (dq_ref - dq),
                            -tau_max,
                            tau_max,
                        )
                        err_str = " ".join(f"{e*57.3:+5.1f}deg" for e in err_terms)
                        tau_str = " ".join(f"{t:+.2f}" for t in tau_print)
                        print(f"  [t={t_rel:5.2f}] err=[{err_str}]  tau_pd=[{tau_str}]")

                wall_now = time.time()
                if (not args.no_plot and plot_state is not None and
                        wall_now - last_plot_t >= plot_period):
                    _update_plot(
                        plot_state,
                        np.asarray(t_buf, dtype=float),
                        np.asarray(ref_buf, dtype=float),
                        np.asarray(act_buf, dtype=float),
                    )
                    last_plot_t = wall_now

            elif phase == "HOLD":
                tau_grav = system.gravity_compensation(q, np.zeros(n))
                tau_pd = kp * (hold_q - q) - kd * dq
                tau_pd = np.clip(tau_pd, -tau_max, tau_max)
                tau_cmd = tau_grav + tau_pd
                system.set_leader_joint_torque(tau_cmd, 0.0)

            sleep_s = max(0.0, float(system.dt) - (time.perf_counter() - now_perf))
            if sleep_s > 0:
                time.sleep(sleep_s)

        print("Finished.")
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Test failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
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

        if plt is not None and len(t_buf) > 10:
            try:
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                t_arr = np.asarray(t_buf, dtype=float)
                refs = np.asarray(ref_buf, dtype=float)
                acts = np.asarray(act_buf, dtype=float)
                errs = acts - refs

                fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                                         gridspec_kw={"hspace": 0.08})

                plt.rcParams.update(
                    {
                        "font.family": "sans-serif",
                        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                        "font.size": 10,
                        "axes.titlesize": 11,
                        "axes.titleweight": "bold",
                    }
                )

                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

                for i in range(refs.shape[1]):
                    lbl = f"J{i+1}" if args.mode == "joint" else ["X", "Y", "Z"][i]
                    axes[0].plot(t_arr, refs[:, i], "--", color=colors[i], alpha=0.5,
                                 lw=1.2, label=f"{lbl} ref")
                    axes[0].plot(t_arr, acts[:, i], "-", color=colors[i],
                                 lw=1.3, label=f"{lbl} act")
                axes[0].set_ylabel("Joint Angle (rad)" if args.mode == "joint" else "TCP Position (m)")
                axes[0].set_title("(a)  Impedance Replay: Reference vs. Actual", loc="left")
                axes[0].legend(ncol=6, fontsize=8, framealpha=0.85, edgecolor="none")
                axes[0].grid(True, ls="--", alpha=0.25)

                for i in range(errs.shape[1]):
                    lbl = f"J{i+1}" if args.mode == "joint" else ["X", "Y", "Z"][i]
                    scale = 57.3 if args.mode == "joint" else 1000.0
                    axes[1].plot(t_arr, errs[:, i] * scale, color=colors[i], lw=1.0, label=lbl)
                unit = "deg" if args.mode == "joint" else "mm"
                axes[1].set_ylabel(f"Tracking Error ({unit})")
                axes[1].set_title("(b)  Per-Joint Tracking Error", loc="left")
                axes[1].legend(ncol=6, fontsize=8, framealpha=0.85, edgecolor="none")
                axes[1].grid(True, ls="--", alpha=0.25)

                rms_scale = 57.3 if args.mode == "joint" else 1000.0
                rms = np.sqrt(np.mean(errs**2, axis=1)) * rms_scale
                axes[2].plot(t_arr, rms, "k-", lw=1.5)
                axes[2].set_ylabel(f"RMS Error ({unit})")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_title(f"(c)  RMS Tracking Error  (mean = {np.mean(rms):.2f}{unit})", loc="left")
                axes[2].grid(True, ls="--", alpha=0.25)

                fig.tight_layout()
                fname = f"trajectory_impedance_{args.mode}_{ts_str}.svg"
                fig.savefig(fname, format="svg", bbox_inches="tight")
                print(f"\nSaved plot -> {fname}")
                plt.close(fig)
            except Exception as e:
                print(f"Plot failed: {e}")

        if plt is not None:
            if plot_state is not None:
                plt.close(plot_state["fig"])
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
