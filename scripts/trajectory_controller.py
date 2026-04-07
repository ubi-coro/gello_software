#!/usr/bin/env python3
"""Impedance Controller - Record & Replay on GELLO hardware.

Records a human-guided trajectory in gravity compensation,
then replays it using joint-space or task-space impedance control.

Phases:
  1. PREP    - Settle in gravity comp (current control)
  2. RECORD  - User moves arm, trajectory recorded (current control)
  3. HOME    - Return to trajectory start (position control)
  4. SETTLE  - Hold at start in position control
  5. SWITCH  - Switch back to current control
  6. REPLAY  - Impedance tracking (current control)
  7. HOLD    - Hold final position until Ctrl+C
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


THESIS_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.3,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "none",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "mathtext.default": "regular",
}

JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SCRIPTS_DIR = Path(__file__).resolve().parent
IMPEDANCE_PLOTS_DIR = SCRIPTS_DIR / "impedancePlots"


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


def save_thesis_plots(
    *,
    t: np.ndarray,
    q_ref: np.ndarray,
    q_act: np.ndarray,
    tau_ff: np.ndarray,
    tau_pd: np.ndarray,
    n_joints: int,
    mode: str,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_max: np.ndarray,
    ts_str: str,
) -> list:
    """Generate two focused thesis figures for joint-space replay."""
    if plt is None:
        print("matplotlib not available - skipping thesis plots.")
        return []

    if len(t) < 10:
        print("Not enough REPLAY samples for thesis plots.")
        return []

    plt.rcParams.update(THESIS_RCPARAMS)
    saved: list = []
    colors = JOINT_COLORS[:n_joints]
    IMPEDANCE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    errs_deg = (q_act - q_ref) * 57.3
    rms_per_t = np.sqrt(np.mean(errs_deg ** 2, axis=1))
    mean_rms = float(np.mean(rms_per_t))
    max_err = float(np.max(np.abs(errs_deg)))

    fig1, (ax1a, ax1b) = plt.subplots(
        2, 1, figsize=(7.0, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.10},
    )

    for i in range(n_joints):
        ax1a.plot(t, q_ref[:, i], "--", color=colors[i], alpha=0.45, lw=1.1, label=f"J{i+1} ref")
        ax1a.plot(t, q_act[:, i], "-", color=colors[i], lw=1.3, label=f"J{i+1} act")
    ax1a.set_ylabel("Joint Angle (rad)")
    ax1a.set_title("(a) Joint-space Impedance Replay: Reference vs Actual", loc="left")
    ax1a.legend(ncol=6, loc="upper right")
    ax1a.grid(True, ls="--")

    for i in range(n_joints):
        ax1b.plot(t, errs_deg[:, i], color=colors[i], lw=1.0, label=f"J{i+1}")
    ax1b.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)
    ax1b.axhline(+mean_rms, color="#E53935", ls="--", lw=0.9, alpha=0.6)
    ax1b.axhline(-mean_rms, color="#E53935", ls="--", lw=0.9, alpha=0.6, label=f"+/-Mean RMS = {mean_rms:.2f} deg")
    ax1b.set_ylabel("Tracking Error (deg)")
    ax1b.set_xlabel("Time (s)")
    ax1b.set_title(
        f"(b) Per-joint Tracking Error (mean RMS = {mean_rms:.2f} deg, max = {max_err:.1f} deg)",
        loc="left",
    )
    ax1b.legend(ncol=4, loc="upper right")
    ax1b.grid(True, ls="--")

    fig1.align_ylabels([ax1a, ax1b])
    fig1.tight_layout()
    fname1 = IMPEDANCE_PLOTS_DIR / f"impedance_tracking_{mode}_{ts_str}.svg"
    fig1.savefig(fname1, format="svg", bbox_inches="tight")
    plt.close(fig1)
    saved.append(str(fname1))
    print(f"  Saved -> {fname1}")

    fig2, (ax2a, ax2b) = plt.subplots(
        2, 1, figsize=(7.0, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.10},
    )

    for i in range(n_joints):
        ax2a.plot(t, tau_ff[:, i], color=colors[i], lw=1.0, label=f"J{i+1}")
    ax2a.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)
    ax2a.set_ylabel("Torque (Nm)")
    ax2a.set_title("(a) Feedforward Torques: tau_ff = tau_grav + tau_fric + tau_damp", loc="left")
    ax2a.legend(ncol=6, loc="upper right")
    ax2a.grid(True, ls="--")

    for i in range(n_joints):
        ax2b.plot(t, tau_pd[:, i], color=colors[i], lw=1.0, label=f"J{i+1}")
    ax2b.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)
    for lim in sorted(set(tau_max.tolist())):
        ax2b.axhline(+lim, color="#999", ls=":", lw=0.7, alpha=0.5)
        ax2b.axhline(-lim, color="#999", ls=":", lw=0.7, alpha=0.5)
    ax2b.set_ylabel("Torque (Nm)")
    ax2b.set_xlabel("Time (s)")
    ax2b.set_title("(b) PD Correction Torques: tau_pd = Kp*(q_ref-q) - Kd*dq", loc="left")
    ax2b.legend(ncol=6, loc="upper right")
    ax2b.grid(True, ls="--")

    rms_ff = float(np.sqrt(np.mean(tau_ff ** 2)))
    rms_pd = float(np.sqrt(np.mean(tau_pd ** 2)))

    fig2.align_ylabels([ax2a, ax2b])
    fig2.tight_layout()
    fname2 = IMPEDANCE_PLOTS_DIR / f"impedance_torques_{mode}_{ts_str}.svg"
    fig2.savefig(fname2, format="svg", bbox_inches="tight")
    plt.close(fig2)
    saved.append(str(fname2))
    print(f"  Saved -> {fname2}")

    return saved


def _init_plot(mode: str):
    """Create lightweight live plot."""
    if plt is None:
        return None
    plt.rcParams.update(THESIS_RCPARAMS)
    plt.ion()
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    n_dims = 6 if mode == "joint" else 3
    colors = JOINT_COLORS[:n_dims]
    lines_ref, lines_act, lines_err = [], [], []
    for i in range(n_dims):
        lbl = f"J{i+1}" if mode == "joint" else ["X", "Y", "Z"][i]
        lr, = axes[0].plot([], [], "--", lw=1.2, alpha=0.5, color=colors[i], label=f"{lbl} ref")
        la, = axes[0].plot([], [], "-", lw=1.3, color=colors[i], label=f"{lbl} act")
        le, = axes[1].plot([], [], "-", lw=1.0, color=colors[i], label=lbl)
        lines_ref.append(lr)
        lines_act.append(la)
        lines_err.append(le)
    axes[0].set_ylabel("Joint Angle (rad)" if mode == "joint" else "Position (m)")
    axes[0].set_title("Impedance Replay (Live)", loc="left")
    axes[0].legend(ncol=6, loc="upper right")
    axes[0].grid(True, ls="--")
    axes[1].set_ylabel("Error (rad)" if mode == "joint" else "Error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(ncol=6, loc="upper right")
    axes[1].grid(True, ls="--")
    fig.tight_layout()
    return {
        "fig": fig,
        "axes": axes,
        "lines_ref": lines_ref,
        "lines_act": lines_act,
        "lines_err": lines_err,
    }


def _update_plot(ps, t, refs, acts):
    if ps is None or t.size == 0:
        return
    tx = t - t[0]
    errs = acts - refs
    for i in range(refs.shape[1]):
        ps["lines_ref"][i].set_data(tx, refs[:, i])
        ps["lines_act"][i].set_data(tx, acts[:, i])
        ps["lines_err"][i].set_data(tx, errs[:, i])
    for ax in ps["axes"]:
        ax.set_xlim(tx[0], max(tx[-1], 0.1))
        ax.relim()
        ax.autoscale_view(scaley=True)
    ps["fig"].canvas.draw_idle()
    ps["fig"].canvas.flush_events()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record & Replay via Impedance Control")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--mode", choices=["joint", "task"], default="joint")

    p.add_argument("--prep-time", type=float, default=3.0)
    p.add_argument("--record-time", type=float, default=5.0)
    p.add_argument("--home-time", type=float, default=5.0)

    p.add_argument("--kp-j", type=float, default=0.0,
                   help="Joint Kp base (Nm/rad). 0 = pure gravity comp.")
    p.add_argument("--kd-j", type=float, default=0.0,
                   help="Joint Kd base (Nm*s/rad). 0 = no extra damping.")

    p.add_argument("--kp-t", type=float, default=150.0)
    p.add_argument("--kd-t", type=float, default=10.0)
    p.add_argument("--kd-null", type=float, default=0.5)

    p.add_argument("--plot-rate", type=float, default=10.0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-thesis-plots", action="store_true", help="Disable thesis figure generation")
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
    tau_ff_buf: deque = deque()
    tau_pd_buf: deque = deque()

    kp = np.zeros(6)
    kd = np.zeros(6)
    tau_max = np.zeros(6)

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

        kp_scale = np.array([0.8, 1.0, 1.0, 0.5, 0.5, 0.3], dtype=float)[:n]
        kd_scale = np.array([0.8, 1.0, 1.0, 0.8, 0.8, 0.5], dtype=float)[:n]

        kp = args.kp_j * kp_scale
        kd = args.kd_j * kd_scale
        tau_max = np.array([1.2, 2.0, 2.0, 0.8, 0.8, 0.5], dtype=float)[:n]

        buf_size = int(max(args.record_time, 10.0) * 500)
        t_buf = deque(maxlen=buf_size)
        ref_buf = deque(maxlen=buf_size)
        act_buf = deque(maxlen=buf_size)
        tau_ff_buf = deque(maxlen=buf_size)
        tau_pd_buf = deque(maxlen=buf_size)

        print(f"Replay Kp: {[f'{x:.2f}' for x in kp]}")
        print(f"Replay Kd: {[f'{x:.2f}' for x in kd]}")
        print(f"Torque limits: {[f'{x:.1f}' for x in tau_max]} Nm")
        if args.kp_j == 0.0 and args.kd_j == 0.0:
            print("  WARNING: Kp=0, Kd=0: REPLAY is pure gravity comp (no tracking)")

        t_start = time.time()
        phase = "PREP"
        trajectory_q: List[Tuple[float, np.ndarray, np.ndarray]] = []

        record_start_t = 0.0
        home_start_t = 0.0
        settle_start_t = 0.0
        replay_start_t = 0.0
        target_q = np.zeros(n)
        hold_q = np.zeros(n)
        debug_counter = 0

        home_start_raw = np.zeros(system.num_motors)
        target_raw = np.zeros(system.num_motors)

        plot_state = None if args.no_plot else _init_plot(args.mode)
        plot_period = 1.0 / max(args.plot_rate, 1e-3)
        last_plot_t = time.time()

        print(f"[PREP] Settle in gravity comp for {args.prep_time}s ...")

        while running:
            now_perf = time.perf_counter()
            t_now = time.time() - t_start

            q, dq, _, _ = system.get_leader_joint_states()

            if phase == "PREP":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)

                if t_now >= args.prep_time:
                    phase = "RECORD"
                    record_start_t = t_now
                    print(f"[RECORD] Recording for {args.record_time}s - MOVE THE ARM NOW.")

            elif phase == "RECORD":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)

                t_rel = t_now - record_start_t
                trajectory_q.append((t_rel, q.copy(), dq.copy()))

                if t_rel >= args.record_time:
                    phase = "HOME"
                    print(f"[HOME] Switching to position mode & returning to start ({args.home_time}s)")

                    raw_pos_now = np.asarray(system.driver.get_joints(), dtype=float)

                    system.driver.set_torque_mode(False)
                    time.sleep(0.02)
                    system.driver.set_operating_mode(3)
                    time.sleep(0.02)
                    system.driver.set_torque_mode(True)
                    system.driver.set_joints(raw_pos_now.tolist())
                    time.sleep(0.02)

                    home_start_t = t_now
                    home_start_raw = raw_pos_now.copy()
                    target_q = trajectory_q[0][1].copy()

                    target_raw = np.zeros(system.num_motors)
                    target_raw[:n] = target_q * system.joint_signs[:n] + system.joint_offsets[:n]
                    if system.num_motors > n:
                        target_raw[-1] = raw_pos_now[-1]

                    for i in range(system.num_motors):
                        delta = target_raw[i] - home_start_raw[i]
                        while delta > np.pi:
                            target_raw[i] -= 2.0 * np.pi
                            delta -= 2.0 * np.pi
                        while delta < -np.pi:
                            target_raw[i] += 2.0 * np.pi
                            delta += 2.0 * np.pi

            elif phase == "HOME":
                t_rel = t_now - home_start_t
                alpha_t = np.clip(t_rel / args.home_time, 0.0, 1.0)
                s = 10 * alpha_t**3 - 15 * alpha_t**4 + 6 * alpha_t**5
                des_raw = home_start_raw + s * (target_raw - home_start_raw)
                system.driver.set_joints(des_raw.tolist())

                if t_rel >= args.home_time:
                    phase = "SETTLE"
                    settle_start_t = t_now
                    print("[SETTLE] Holding start position (1.5s) ...")

            elif phase == "SETTLE":
                system.driver.set_joints(target_raw.tolist())

                if (t_now - settle_start_t) >= 1.5:
                    print("[SWITCH] Switching to current control for impedance replay...")
                    system.driver.set_torque_mode(False)
                    time.sleep(0.02)
                    system.driver.set_operating_mode(0)
                    time.sleep(0.02)
                    system.driver.set_torque_mode(True)
                    time.sleep(0.05)

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
                _, q_ref, _dq_ref = trajectory_q[idx]

                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_ff = tau_grav + tau_fric + tau_damp

                if args.mode == "joint":
                    q_ref_near = q_ref.copy()
                    for i in range(n):
                        while q_ref_near[i] - q[i] > np.pi:
                            q_ref_near[i] -= 2.0 * np.pi
                        while q_ref_near[i] - q[i] < -np.pi:
                            q_ref_near[i] += 2.0 * np.pi

                    tau_pd = kp * (q_ref_near - q) - kd * dq
                    tau_pd_clipped = np.clip(tau_pd, -tau_max, tau_max)
                    system.set_leader_joint_torque(tau_ff + tau_pd_clipped, 0.0)

                    t_buf.append(t_rel)
                    ref_buf.append(q_ref_near.copy())
                    act_buf.append(q.copy())
                    tau_ff_buf.append(tau_ff.copy())
                    tau_pd_buf.append(tau_pd_clipped.copy())

                else:
                    x_ref, _ = compute_task_kinematics(system, q_ref)
                    x_act, j_act = compute_task_kinematics(system, q)
                    dx_act = j_act @ dq

                    f_task = args.kp_t * (x_ref - x_act) - args.kd_t * dx_act
                    tau_task = j_act.T @ f_task
                    null_proj = np.eye(n) - np.linalg.pinv(j_act) @ j_act
                    tau_null = null_proj @ (-args.kd_null * dq)
                    tau_imp = np.clip(tau_task + tau_null, -tau_max, tau_max)
                    system.set_leader_joint_torque(tau_ff + tau_imp, 0.0)

                    t_buf.append(t_rel)
                    ref_buf.append(x_ref.copy())
                    act_buf.append(x_act.copy())
                    tau_ff_buf.append(tau_ff.copy())
                    tau_pd_buf.append(tau_imp.copy())

                debug_counter += 1
                if debug_counter % 30 == 1 and args.mode == "joint":
                    q_ref_dbg = q_ref.copy()
                    for i in range(n):
                        while q_ref_dbg[i] - q[i] > np.pi:
                            q_ref_dbg[i] -= 2.0 * np.pi
                        while q_ref_dbg[i] - q[i] < -np.pi:
                            q_ref_dbg[i] += 2.0 * np.pi

                    tau_dbg = np.clip(kp * (q_ref_dbg - q) - kd * dq, -tau_max, tau_max)
                    err_terms = q - q_ref_dbg
                    err_str = " ".join(f"{e*57.3:+5.1f}deg" for e in err_terms)
                    tau_str = " ".join(f"{t:+.2f}" for t in tau_dbg)
                    ff_str = " ".join(f"{t:+.2f}" for t in tau_ff)
                    print(f"  [t={t_rel:5.2f}] err=[{err_str}]  tau_pd=[{tau_str}]  tau_ff=[{ff_str}]")

                wall_now = time.time()
                if (not args.no_plot and plot_state is not None and wall_now - last_plot_t >= plot_period):
                    _update_plot(
                        plot_state,
                        np.asarray(t_buf, dtype=float),
                        np.asarray(ref_buf, dtype=float),
                        np.asarray(act_buf, dtype=float),
                    )
                    last_plot_t = wall_now

            elif phase == "HOLD":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_pd = np.clip(kp * (hold_q - q) - kd * dq, -tau_max, tau_max)
                system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp + tau_pd, 0.0)

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

        if (
            args.mode == "joint"
            and not args.no_thesis_plots
            and plt is not None
            and len(t_buf) > 10
            and len(tau_ff_buf) > 10
        ):
            try:
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                print(f"\nGenerating thesis plots ({len(t_buf)} samples)...")

                files = save_thesis_plots(
                    t=np.asarray(t_buf, dtype=float),
                    q_ref=np.asarray(ref_buf, dtype=float),
                    q_act=np.asarray(act_buf, dtype=float),
                    tau_ff=np.asarray(tau_ff_buf, dtype=float),
                    tau_pd=np.asarray(tau_pd_buf, dtype=float),
                    n_joints=n,
                    mode=args.mode,
                    kp=kp,
                    kd=kd,
                    tau_max=tau_max,
                    ts_str=ts_str,
                )
                print(f"Generated {len(files)} thesis plots.")
            except Exception as e:
                print(f"Thesis plot generation failed: {e}")
                import traceback
                traceback.print_exc()

        if plt is not None:
            if plot_state is not None:
                plt.close(plot_state["fig"])
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
