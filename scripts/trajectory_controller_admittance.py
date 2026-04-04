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
    3. HOME   — Position interpolation moves arm to trajectory start
    4. TARE   — Observer bias/deadband calibration at rest
    5. WARMUP — Observer settles while holding start position
    6. REPLAY — Admittance control tracks the recorded trajectory

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
    colors = np.array([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    ])[:n_dims]
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


# ── THESIS PLOT GENERATION ───────────────────────────────────────────

def save_thesis_plots(
    *,
    t: np.ndarray,
    tau_raw: np.ndarray,
    tau_comp: np.ndarray,
    d_prev: np.ndarray,
    q_ref: np.ndarray,
    q_act: np.ndarray,
    q_c: np.ndarray,
    currents: np.ndarray,
    tare_offset: np.ndarray,
    tare_std: np.ndarray,
    deadband: np.ndarray,
    tau_clamp: np.ndarray,
    tare_samples: Optional[np.ndarray],
    motor_kt: np.ndarray,
    motor_gear_ratio: np.ndarray,
    motor_eta: np.ndarray,
    n_joints: int,
    ts_str: str,
) -> list:
    """Generate publication-quality SVG figures for thesis.

    Produces:
      1. Observer failure during replay (2-panel: τ_ext raw + Δq)
      2. Tare offset bar chart
      3. η-mechanism detail for J3 (2-panel: current + τ_ext)
    """
    del tau_comp, q_act, tau_clamp, tare_samples

    if plt is None:
        print("matplotlib not available - skipping thesis plots.")
        return []

    saved: list = []

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "mathtext.default": "regular",
    })

    colors = {
        "J1": "#1f77b4", "J2": "#ff7f0e", "J3": "#2ca02c",
        "J4": "#d62728", "J5": "#9467bd", "J6": "#8c564b",
    }
    joint_labels = [f"J{i+1}" for i in range(n_joints)]
    joint_labels_full = [
        "J1 (Base)", "J2 (Shoulder)", "J3 (Elbow)",
        "J4 (Wrist 1)", "J5 (Wrist 2)", "J6 (Wrist 3)",
    ][:n_joints]

    n_samples = len(t)
    if n_samples < 5:
        print("Not enough REPLAY samples for thesis plots.")
        return []

    c_list = [colors[f"J{i+1}"] for i in range(n_joints)]

    # FIGURE 1: Observer failure analysis (2 panels)
    fig1, (ax1a, ax1b) = plt.subplots(
        2, 1, figsize=(7.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.12},
    )

    for i in range(n_joints):
        lw = 1.8 if i in (1, 2) else 0.9
        alpha = 1.0 if i in (1, 2) else 0.35
        ax1a.plot(t, tau_raw[:, i], color=c_list[i], lw=lw, alpha=alpha,
                  label=joint_labels[i])

    ax1a.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)

    for j_idx, j_name in [(1, "J2"), (2, "J3")]:
        if j_idx >= n_joints:
            continue
        peak_idx = int(np.argmax(np.abs(tau_raw[:, j_idx])))
        peak_val = float(tau_raw[peak_idx, j_idx])
        peak_t = float(t[peak_idx])
        ax1a.annotate(
            f"{j_name}: {peak_val:+.1f} Nm",
            xy=(peak_t, peak_val),
            xytext=(peak_t + 0.3, peak_val + 0.3 * np.sign(peak_val)),
            fontsize=8.5,
            color=c_list[j_idx],
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": c_list[j_idx], "lw": 1.2},
        )

    ax1a.set_ylabel(r"$\hat{\tau}_{\mathrm{ext}}$  (Nm)")
    ax1a.set_title("(a)  Raw Shi Observer Output During Trajectory Replay", loc="left")
    ax1a.legend(ncol=6, loc="upper right", framealpha=0.9, edgecolor="none", fontsize=8)
    ax1a.grid(True, ls="--")

    n_min = min(q_c.shape[0], q_ref.shape[0], n_samples)
    dev_deg = (q_c[:n_min] - q_ref[:n_min]) * (180.0 / np.pi)
    t_dev = t[:n_min]

    for i in range(n_joints):
        lw = 1.8 if i in (1, 2) else 0.9
        alpha = 1.0 if i in (1, 2) else 0.35
        ax1b.plot(t_dev, dev_deg[:, i], color=c_list[i], lw=lw, alpha=alpha,
                  label=joint_labels[i])

    ax1b.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)
    ax1b.set_ylabel(r"$q_c - q_{\mathrm{ref}}$  (deg)")
    ax1b.set_xlabel("Time  (s)")
    ax1b.set_title("(b)  Resulting Admittance Output Deviation", loc="left")
    ax1b.legend(ncol=6, loc="upper right", framealpha=0.9, edgecolor="none", fontsize=8)
    ax1b.grid(True, ls="--")

    fig1.align_ylabels([ax1a, ax1b])
    fig1.tight_layout()
    fname1 = f"shi_observer_replay_failure_{ts_str}.svg"
    fig1.savefig(fname1, format="svg", bbox_inches="tight")
    plt.close(fig1)
    saved.append(fname1)
    print(f"  Saved -> {fname1}")

    # FIGURE 2: Tare offset bar chart
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.0))
    x_pos = np.arange(n_joints)
    bar_w = 0.55

    ax2.bar(
        x_pos, tare_offset, bar_w,
        yerr=3.0 * tare_std,
        capsize=4,
        ecolor="#555555",
        color=c_list,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    for i in range(n_joints):
        ax2.plot(
            [i - bar_w / 2 - 0.08, i + bar_w / 2 + 0.08],
            [+deadband[i]] * 2,
            color="#E53935", ls="--", lw=1.1, zorder=4,
        )
        ax2.plot(
            [i - bar_w / 2 - 0.08, i + bar_w / 2 + 0.08],
            [-deadband[i]] * 2,
            color="#E53935", ls="--", lw=1.1, zorder=4,
        )

    proxy = ax2.plot([], [], color="#E53935", ls="--", lw=1.2,
                     label="Applied Deadband")[0]
    ax2.legend(handles=[proxy], loc="upper left", framealpha=0.9)

    ax2.axhline(0, color="k", ls="-", lw=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(joint_labels_full, fontsize=9)
    ax2.set_ylabel("Torque  (Nm)")
    ax2.set_title("Observer Tare Offset at Rest  (error bars: 3σ)", fontweight="bold")
    ax2.grid(True, axis="y", ls="--")
    fig2.tight_layout()
    fname2 = f"shi_observer_tare_offsets_{ts_str}.svg"
    fig2.savefig(fname2, format="svg", bbox_inches="tight")
    plt.close(fig2)
    saved.append(fname2)
    print(f"  Saved -> {fname2}")

    # FIGURE 3: eta-mechanism detail for J3 (2 panels)
    j = 2
    if n_joints > j and currents.shape[0] == n_samples:
        fig3, (ax3a, ax3b) = plt.subplots(
            2, 1, figsize=(7.0, 5.0), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.2], "hspace": 0.12},
        )

        c3 = c_list[j]
        eta_j = float(motor_eta[j])
        r_j = float(motor_gear_ratio[j])
        kt_j = float(motor_kt[j])
        ratio_str = f"{(1.0 / eta_j) / eta_j:.1f}"

        i_ma = currents[:, j]
        ax3a.plot(t, i_ma, color=c3, lw=1.2, label="Motor Current")
        ax3a.set_ylabel("Current  (mA)")
        ax3a.set_title(
            f"(a)  {joint_labels_full[j]}  -  Motor Current (r = {r_j:.0f}:1, eta = {eta_j:.2f})",
            loc="left",
        )
        ax3a.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)

        d_j = d_prev[:, j]
        for k in range(n_samples - 1):
            c_bg = "#E3F2FD" if d_j[k] > 0 else "#FBE9E7"
            ax3a.axvspan(t[k], t[k + 1], alpha=0.4, color=c_bg, linewidth=0)

        p_fwd = ax3a.plot([], [], color="#E3F2FD", lw=8,
                          label=f"d = +1 (eta = {eta_j:.2f})")[0]
        p_bwd = ax3a.plot([], [], color="#FBE9E7", lw=8,
                          label=f"d = -1 (1/eta = {1 / eta_j:.2f})")[0]
        ax3a.legend(
            handles=[ax3a.get_lines()[0], p_fwd, p_bwd],
            loc="upper right", framealpha=0.9, edgecolor="none", fontsize=8,
        )
        ax3a.grid(True, ls="--")

        ax3b.plot(t, tau_raw[:, j], color=c3, lw=1.4,
                  label=r"$\hat{\tau}_{\mathrm{ext}}$  (raw)")
        ax3b.axhline(
            tare_offset[j], color="#757575", ls=":", lw=1.2,
            label=f"Tare offset ({tare_offset[j]:+.3f} Nm)",
        )
        ax3b.axhspan(
            tare_offset[j] - deadband[j], tare_offset[j] + deadband[j],
            color="#4CAF50", alpha=0.12, zorder=0,
            label=f"Deadband (+/-{deadband[j]:.2f} Nm)",
        )
        ax3b.axhline(0, color="k", ls="-", lw=0.4, alpha=0.4)

        for k in range(n_samples - 1):
            c_bg = "#E3F2FD" if d_j[k] > 0 else "#FBE9E7"
            ax3b.axvspan(t[k], t[k + 1], alpha=0.3, color=c_bg, linewidth=0)

        ax3b.text(
            0.02, 0.95,
            f"eta ratio: {ratio_str}:1 -> torque jumps up to {r_j * abs(1 / eta_j - eta_j) * kt_j * 0.5:.1f} Nm",
            transform=ax3b.transAxes,
            fontsize=8.5,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#999", "alpha": 0.9},
        )

        ax3b.set_ylabel(r"$\hat{\tau}_{\mathrm{ext}}$  (Nm)")
        ax3b.set_xlabel("Time  (s)")
        ax3b.set_title(
            r"(b)  Estimated External Torque  $\hat{\tau}_{\mathrm{ext}} = -(\tau_{\mathrm{out}} - \tau_{\mathrm{grav}})$",
            loc="left",
        )
        ax3b.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=8)
        ax3b.grid(True, ls="--")

        fig3.align_ylabels([ax3a, ax3b])
        fig3.tight_layout()
        fname3 = f"shi_observer_eta_detail_J3_{ts_str}.svg"
        fig3.savefig(fname3, format="svg", bbox_inches="tight")
        plt.close(fig3)
        saved.append(fname3)
        print(f"  Saved -> {fname3}")

    return saved


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record & Replay via Admittance Control")
    p.add_argument("--config", type=str,
                   default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--mode", choices=["joint", "task"], default="joint")

    p.add_argument("--prep-time",   type=float, default=3.0)
    p.add_argument("--record-time", type=float, default=5.0)
    p.add_argument("--home-time",   type=float, default=4.0)

    # Kept for compatibility with existing launch scripts.
    p.add_argument("--kp-low", type=float, default=10.0)
    p.add_argument("--kd-low", type=float, default=0.5)

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
    p.add_argument("--thesis-plots", dest="thesis_plots", action="store_true",
                   default=True,
                   help="Enable thesis plot generation at shutdown (default: enabled)")
    p.add_argument("--no-thesis-plots", dest="thesis_plots", action="store_false",
                   help="Disable thesis plot generation at shutdown")
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

    plot_state = None
    tau_raw_buf: deque = deque()
    tau_comp_buf: deque = deque()
    d_prev_buf: deque = deque()
    qc_buf: deque = deque()
    curr_buf: deque = deque()
    t_buf: deque = deque()
    ref_buf: deque = deque()
    act_buf: deque = deque()
    tare_samples_for_plot: Optional[np.ndarray] = None
    observer_tare = np.zeros(n)
    observer_deadband = np.zeros(n)
    tau_ext_max = np.zeros(n)
    shi: Optional[MinimalistTorqueEstimator] = None

    try:
        system = FACTRGravityCompensation(str(config_path),
                                          enable_visualization=False)
        if system.driver is None:
            raise RuntimeError("FACTR driver is not initialized")
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

        # ── THESIS PLOT BUFFERS (Replay phase) ───────────────────
        tau_raw_buf = deque(maxlen=int(args.record_time * 500))
        tau_comp_buf = deque(maxlen=int(args.record_time * 500))
        d_prev_buf = deque(maxlen=int(args.record_time * 500))
        qc_buf = deque(maxlen=int(args.record_time * 500))
        curr_buf = deque(maxlen=int(args.record_time * 500))

        # Tare analysis storage (filled during TARE phase)
        tare_samples_for_plot = None

        # Variables set during phase transitions
        record_start_t = 0.0
        home_start_t   = 0.0
        tare_start_t   = 0.0
        warmup_start_t = 0.0
        home_start_q   = np.zeros(n)
        target_q       = np.zeros(n)
        replay_start_t = 0.0
        home_start_raw = np.zeros(system.num_motors)
        target_raw = np.zeros(system.num_motors)
        last_sent_raw = np.zeros(system.num_motors)
        q_ref_init_check: Optional[np.ndarray] = None
        tare_samples: List[np.ndarray] = []
        observer_tare = np.zeros(n)
        observer_deadband = np.zeros(n)
        tau_ext_prev = np.zeros(n)
        tau_ext_rate_limit = 0.05
        tau_ext_max = np.array([0.15, 0.15, 0.15, 0.10, 0.10, 0.05],
                               dtype=float)
        if n < len(tau_ext_max):
            tau_ext_max = tau_ext_max[:n]
        elif n > len(tau_ext_max):
            tau_ext_max = np.pad(tau_ext_max, (0, n - len(tau_ext_max)),
                                 mode="edge")
        leak = 0.995
        debug_counter = 0

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
                    print(f"[{phase}] Switching to position mode & returning "
                          f"to start ({args.home_time}s) …")

                    # Read raw positions before mode switch to avoid a jump.
                    raw_pos_now = np.asarray(system.driver.get_joints(),
                                             dtype=float)

                    system.driver.set_torque_mode(False)
                    time.sleep(0.02)
                    system.driver.set_operating_mode(3)
                    time.sleep(0.02)
                    system.driver.set_torque_mode(True)

                    # Immediately hold current raw position after switching.
                    system.driver.set_joints(raw_pos_now.tolist())
                    last_sent_raw = raw_pos_now.copy()
                    time.sleep(0.02)

                    home_start_t = t_now
                    home_start_q = q.copy()
                    home_start_raw = raw_pos_now.copy()
                    target_q = trajectory_q[0][1]

                    target_raw = np.zeros(system.num_motors)
                    target_raw[:n] = (
                        target_q * system.joint_signs[:n]
                        + system.joint_offsets[:n]
                    )
                    if system.num_motors > n:
                        target_raw[-1] = raw_pos_now[-1]

                    # Ensure HOME interpolation takes the shortest angular path.
                    for i in range(n):
                        delta = target_raw[i] - home_start_raw[i]
                        while delta > np.pi:
                            target_raw[i] -= 2.0 * np.pi
                            delta -= 2.0 * np.pi
                        while delta < -np.pi:
                            target_raw[i] += 2.0 * np.pi
                            delta += 2.0 * np.pi

            # ── PHASE: HOME ──────────────────────────────────────────
            elif phase == "HOME":
                t_rel = t_now - home_start_t
                alpha_t = np.clip(t_rel / args.home_time, 0.0, 1.0)
                # Minimum-jerk profile
                s = 10*alpha_t**3 - 15*alpha_t**4 + 6*alpha_t**5
                des_raw = home_start_raw + s * (target_raw - home_start_raw)
                system.driver.set_joints(des_raw.tolist())
                last_sent_raw = des_raw.copy()

                if t_rel >= args.home_time:
                    phase = "TARE"
                    tare_start_t = t_now
                    tare_samples = []
                    print("[TARE] Holding position, calibrating observer (1s)...")

            # ── PHASE: TARE ──────────────────────────────────────────
            elif phase == "TARE":
                t_rel = t_now - tare_start_t

                system.driver.set_joints(target_raw.tolist())
                last_sent_raw = target_raw.copy()
                tare_samples.append(tau_ext.copy())

                if t_rel >= 1.0:
                    tare_array = np.asarray(tare_samples, dtype=float)
                    observer_tare = np.mean(tare_array, axis=0)
                    tare_std = np.std(tare_array, axis=0)
                    min_deadband = np.array([0.05, 0.30, 0.30, 0.03, 0.03, 0.02],
                                            dtype=float)
                    if n < len(min_deadband):
                        min_deadband = min_deadband[:n]
                    elif n > len(min_deadband):
                        min_deadband = np.pad(
                            min_deadband,
                            (0, n - len(min_deadband)),
                            mode="edge",
                        )
                    observer_deadband = np.maximum(3.0 * tare_std, min_deadband)

                    # ── THESIS: store full tare array for plotting ─
                    tare_samples_for_plot = tare_array.copy()

                    print(f"  Tare offset: {[f'{x:+.3f}' for x in observer_tare]} Nm")
                    print(f"  Deadband 3σ+min: {[f'{x:.3f}' for x in observer_deadband]} Nm")

                    phase = "WARMUP"
                    warmup_start_t = t_now
                    print("[WARMUP] Observer stabilisiert (0.5s)...")

            # ── PHASE: WARMUP ────────────────────────────────────────
            elif phase == "WARMUP":
                t_rel = t_now - warmup_start_t
                system.driver.set_joints(target_raw.tolist())
                last_sent_raw = target_raw.copy()

                if t_rel >= 0.5:
                    phase = "REPLAY"
                    replay_start_t = time.time() - t_start
                    tau_ext_prev = np.zeros(n)
                    debug_counter = 0

                    q_c = trajectory_q[0][1].copy()
                    dq_c = np.zeros(n)
                    q_ref_init_check = trajectory_q[0][1]

                    # Initialization verification for q_c/q_ref consistency.
                    init_dev = (q_c - q_ref_init_check) * 57.3
                    print("  [INIT CHECK] q_c == q_ref[0]: "
                          f"{np.allclose(q_c, q_ref_init_check)}")
                    print("  [INIT CHECK] q_c (deg): "
                          f"[{' '.join(f'{x*57.3:+.1f}' for x in q_c)}]")
                    print("  [INIT CHECK] q_ref[0] (deg): "
                          f"[{' '.join(f'{x*57.3:+.1f}' for x in q_ref_init_check)}]")
                    print("  [INIT CHECK] deviation (deg): "
                          f"[{' '.join(f'{x:+.1f}' for x in init_dev)}]")
                    print("  [INIT CHECK] traj len: "
                          f"{len(trajectory_q)}, "
                          f"t_range: [{trajectory_q[0][0]:.4f}, "
                          f"{trajectory_q[-1][0]:.4f}]s")

                    print("  [LIMITS] arm_joint_limits_min (deg): "
                        f"[{' '.join(f'{x*57.3:+.1f}' for x in system.arm_joint_limits_min)}]")
                    print("  [LIMITS] arm_joint_limits_max (deg): "
                        f"[{' '.join(f'{x*57.3:+.1f}' for x in system.arm_joint_limits_max)}]")
                    print("  [LIMITS] q_c at start (deg):         "
                        f"[{' '.join(f'{x*57.3:+.1f}' for x in q_c)}]")
                    below_min = q_c < system.arm_joint_limits_min
                    above_max = q_c > system.arm_joint_limits_max
                    if np.any(below_min) or np.any(above_max):
                        print("  [LIMITS] WARNING: q_c OUTSIDE joint limits")
                        print(f"  [LIMITS] Below min indices: {np.where(below_min)[0].tolist()}")
                        print(f"  [LIMITS] Above max indices: {np.where(above_max)[0].tolist()}")

                    if args.mode == "task":
                        x_c, _ = compute_task_kinematics(system, q_c)
                        dx_c = np.zeros(3)

                    print("[REPLAY] Admittance tracking active.")

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

                # First-step verification (runs once at REPLAY entry).
                if debug_counter == 0:
                    print(f"  [REPLAY STEP 0] idx={idx}, t_rel={t_rel:.6f}s")
                    print("  [REPLAY STEP 0] q_c (deg):   "
                          f"[{' '.join(f'{x*57.3:+.1f}' for x in q_c)}]")
                    print("  [REPLAY STEP 0] q_ref (deg): "
                          f"[{' '.join(f'{x*57.3:+.1f}' for x in q_ref)}]")
                    print("  [REPLAY STEP 0] q_actual (deg): "
                          f"[{' '.join(f'{x*57.3:+.1f}' for x in q)}]")
                    print("  [REPLAY STEP 0] q_c is q_ref: "
                          f"{q_c is q_ref_init_check}, "
                          "q_ref is traj[0][1]: "
                          f"{q_ref is trajectory_q[0][1]}")
                    print("  [REPLAY STEP 0] measured_dt: "
                          f"{measured_dt*1000:.2f}ms")

                tau_ext_comp = tau_ext - observer_tare

                tau_ext_comp = np.where(
                    np.abs(tau_ext_comp) < observer_deadband,
                    0.0,
                    tau_ext_comp
                )

                delta_tau = np.clip(
                    tau_ext_comp - tau_ext_prev,
                    -tau_ext_rate_limit,
                    tau_ext_rate_limit,
                )
                tau_ext_comp = tau_ext_prev + delta_tau
                tau_ext_prev = tau_ext_comp.copy()

                tau_ext_comp = np.clip(tau_ext_comp, -tau_ext_max, tau_ext_max)

                # ── THESIS LOGGING ───────────────────────────────
                tau_raw_buf.append(tau_ext.copy())
                tau_comp_buf.append(tau_ext_comp.copy())
                d_prev_buf.append(shi.d_prev.copy())
                curr_buf.append(currents_arm.copy())
                # q_c is logged after admittance dynamics.

                # --- Joint-Space Admittance ---
                if args.mode == "joint":
                    ddq_c = (
                        tau_ext_comp
                        - args.damp_j  * dq_c
                        - args.stiff_j * (q_c - q_ref)

                    ) / args.mass_j

                    dq_c = leak * dq_c + ddq_c * measured_dt
                    q_c  = q_c  + dq_c  * measured_dt

                    # ── THESIS: log q_c after dynamics ───────────
                    qc_buf.append(q_c.copy())

                    t_buf.append(t_rel)
                    ref_buf.append(q_ref.copy())
                    act_buf.append(q.copy())

                # --- Task-Space Admittance ---
                elif args.mode == "task":
                    x_ref, _     = compute_task_kinematics(system, q_ref)
                    x_act, J_act = compute_task_kinematics(system, q)

                    # τ_ext → F_ext  via  (J^T)^+
                    F_ext = np.linalg.pinv(J_act.T) @ tau_ext_comp

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

                    # ── THESIS: log q_c after dynamics ───────────
                    qc_buf.append(q_c.copy())

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

                # Prevent 360° jumps by wrapping target to the nearest turn.
                for i in range(n):
                    while target_hw[i] - last_sent_raw[i] > np.pi:
                        target_hw[i] -= 2.0 * np.pi
                    while target_hw[i] - last_sent_raw[i] < -np.pi:
                        target_hw[i] += 2.0 * np.pi

                system.driver.set_joints(target_hw.tolist())
                last_sent_raw = target_hw.copy()

                debug_counter += 1
                if debug_counter % 30 == 1:
                    q_ref_arr = np.asarray(q_ref, dtype=float)
                    q_c_arr = np.asarray(q_c, dtype=float)
                    same_joint_dim = q_ref_arr.shape == q_c_arr.shape
                    if same_joint_dim:
                        # Paranoid check to verify the exact q_c-q_ref state at print time.
                        _check_dev = (q_c_arr - q_ref_arr) * 57.3
                        _check_max = float(np.max(np.abs(_check_dev)))
                        if _check_max > 1.0 and debug_counter <= 2:
                            print(f"  [BUG?] q_c     = {q_c_arr}")
                            print(f"  [BUG?] q_ref   = {q_ref_arr}")
                            print(f"  [BUG?] q_c-ref = {_check_dev}")
                            print(f"  [BUG?] q_ref is traj[{idx}][1]: "
                                  f"{q_ref is trajectory_q[idx][1]}")
                            print(f"  [BUG?] id(q_c)={id(q_c)}, id(q_ref)={id(q_ref)}")
                        dq_deg_str = ' '.join(f'{x:+5.1f}' for x in _check_dev)
                    else:
                        dq_deg_str = "n/a(mode=task)"

                    print(
                        f"  [t={t_rel:5.2f}] "
                        f"d={[f'{int(x):+d}' for x in shi.d_prev]}  "
                        f"tau_raw=[{' '.join(f'{x:+.3f}' for x in tau_ext)}]  "
                        f"tau_comp=[{' '.join(f'{x:+.3f}' for x in tau_ext_comp)}]  "
                        f"dq_deg=[{dq_deg_str}]"
                    )

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
                if system.driver is not None:
                    system.driver.set_operating_mode(3)
                    time.sleep(0.05)
                system.shutdown()
            except Exception:
                pass

        safe_t_buf = t_buf
        safe_ref_buf = ref_buf
        safe_act_buf = act_buf
        safe_shi = shi

        # ── THESIS PLOT GENERATION ───────────────────────────
        if args.thesis_plots and plt is not None and len(tau_raw_buf) > 10:
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                print(f"\nGenerating thesis plots ({len(tau_raw_buf)} samples)...")

                t_arr = np.asarray(safe_t_buf, dtype=float)
                tau_raw_arr = np.asarray(tau_raw_buf, dtype=float)
                tau_comp_arr = np.asarray(tau_comp_buf, dtype=float)
                d_prev_arr = np.asarray(d_prev_buf, dtype=float)
                q_ref_arr = np.asarray(safe_ref_buf, dtype=float)
                q_act_arr = np.asarray(safe_act_buf, dtype=float)
                q_c_arr = np.asarray(qc_buf, dtype=float)
                curr_arr = np.asarray(curr_buf, dtype=float)

                if safe_shi is None:
                    raise RuntimeError("Shi observer not initialized")
                m_kt = safe_shi.kt[:n].copy()
                m_gr = safe_shi.gear_ratio[:n].copy()
                m_eta = safe_shi.eta[:n].copy()

                files = save_thesis_plots(
                    t=t_arr,
                    tau_raw=tau_raw_arr,
                    tau_comp=tau_comp_arr,
                    d_prev=d_prev_arr,
                    q_ref=q_ref_arr,
                    q_act=q_act_arr,
                    q_c=q_c_arr,
                    currents=curr_arr,
                    tare_offset=observer_tare,
                    tare_std=(
                        np.std(tare_samples_for_plot, axis=0)
                        if tare_samples_for_plot is not None
                        else np.zeros(n)
                    ),
                    deadband=observer_deadband,
                    tau_clamp=tau_ext_max,
                    tare_samples=tare_samples_for_plot,
                    motor_kt=m_kt,
                    motor_gear_ratio=m_gr,
                    motor_eta=m_eta,
                    n_joints=n,
                    ts_str=ts,
                )
                print(f"Generated {len(files)} thesis plots.")

            except Exception as e:
                print(f"Thesis plot generation failed: {e}")
                import traceback
                traceback.print_exc()

        # ── Legacy live-plot save (if enabled) ───────────────
        if plt is not None:
            if "plot_state" in dir() and plot_state is not None:
                ts_legacy = time.strftime("%Y%m%d_%H%M%S")
                path = f"trajectory_admittance_{args.mode}_{ts_legacy}.svg"
                plot_state["fig"].savefig(path, format="svg",
                                          bbox_inches="tight")
                print(f"Saved live plot -> {path}")
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())