#!/usr/bin/env python3
"""Impedance and Admittance Controller Testing Script on real GELLO hardware.

Records a human-guided trajectory in gravity compensation,
then replays it using either joint-space or task-space impedance control.
Plots the reference vs actual trajectory error using academic styling.

Phases:
1. Settle (Grav comp)
2. Record (User moves arm)
3. Homing (Moves slowly to trajectory start)
4. Replay (Impedance control: joint or task space)
5. Plot resulting trajectory matching.
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

# Optional, fallback robust try-except for pinocchio
try:
    import pinocchio as pin
except ImportError:
    pin = None

from gello.factr.gravity_compensation import FACTRGravityCompensation  # noqa: E402


def _init_plot(mode: str, window_s: float):
    if plt is None:
        return None

    # Set acadamic typesetting (Arial/sans-serif)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2

    plt.ion()
    
    # 1x1 Matrix to satisfy Master Thesis layout explicitly requested
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Trajectory Replay Error ({mode.capitalize()} Space)", fontweight="bold")

    lines_ref = []
    lines_act = []
    
    n_dims = 6 if mode == "joint" else 3
    colors = plt.cm.tab10(np.linspace(0, 1, n_dims))
    
    for i in range(n_dims):
        lbl = f"Joint {i+1}" if mode == "joint" else ["X", "Y", "Z"][i]
        (lr,) = ax.plot([], [], linestyle="--", linewidth=1.5, alpha=0.5, color=colors[i], label=f"{lbl} Ref")
        (la,) = ax.plot([], [], linestyle="-", linewidth=1.5, color=colors[i], label=f"{lbl} Act")
        lines_ref.append(lr)
        lines_act.append(la)

    ylab = "Joint Angle (rad)" if mode == "joint" else "TCP Position (m)"
    ax.set_ylabel(ylab)
    ax.set_xlabel(f"Time (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Put legend outside to avoid cluttering 1x1 plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    fig.tight_layout()
    return {
        "fig": fig,
        "ax": ax,
        "lines_ref": lines_ref,
        "lines_act": lines_act,
    }


def _update_plot(
    plot_state,
    t_hist: np.ndarray,
    ref_hist: np.ndarray,
    act_hist: np.ndarray,
):
    if plot_state is None or t_hist.size == 0:
        return

    t0 = t_hist[0]
    tx = t_hist - t0

    for i in range(ref_hist.shape[1]):
        plot_state["lines_ref"][i].set_data(tx, ref_hist[:, i])
        plot_state["lines_act"][i].set_data(tx, act_hist[:, i])

    plot_state["ax"].set_xlim(tx[0], tx[-1] if tx[-1] > 1e-3 else 1.0)
    plot_state["ax"].relim()
    plot_state["ax"].autoscale_view(scaley=True)

    plot_state["fig"].canvas.draw_idle()
    plot_state["fig"].canvas.flush_events()


def compute_task_kinematics(system: FACTRGravityCompensation, q: np.ndarray):
    """Computes EE position and spatial velocity Jacobian."""
    if pin is None:
        raise RuntimeError("pinocchio not installed but required for task-space.")
    
    n = system.num_arm_joints
    q_full = np.zeros(system._pin_nq)
    q_full[:min(len(q), system._pin_nq)] = q[:min(len(q), system._pin_nq)]
    
    pin.forwardKinematics(system.pin_model, system.pin_data, q_full)
    pin.computeJointJacobians(system.pin_model, system.pin_data, q_full)
    pin.updateFramePlacements(system.pin_model, system.pin_data)
    
    ee_frame_id = system.pin_model.nframes - 1
    tcp_pose = system.pin_data.oMf[ee_frame_id]
    x = np.array(tcp_pose.translation)
    
    J_full = pin.getFrameJacobian(system.pin_model, system.pin_data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_v = J_full[:3, :n]
    
    return x, J_v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and Replay Trajectory via Impedance Control")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ur5e_gello_factr_hw_V2.yaml",
        help="Path to FACTR YAML config",
    )
    parser.add_argument("--mode", type=str, choices=["joint", "task"], default="joint", help="Impedance control space")
    
    # Timing args
    parser.add_argument("--prep-time", type=float, default=3.0, help="Time to settle before record (s)")
    parser.add_argument("--record-time", type=float, default=5.0, help="Duration to record trajectory (s)")
    parser.add_argument("--home-time", type=float, default=4.0, help="Time allocated to home to start pos (s)")
    
    # Gains (Joint-Space)
    parser.add_argument("--kp-j", type=float, default=30.0, help="Joint space proportional gain")
    parser.add_argument("--kd-j", type=float, default=2.0, help="Joint space derivative gain")
    
    # Gains (Task-Space)
    parser.add_argument("--kp-t", type=float, default=150.0, help="Task space proportional gain (N/m)")
    parser.add_argument("--kd-t", type=float, default=10.0, help="Task space derivative gain")
    parser.add_argument("--kd-null", type=float, default=0.5, help="Null-space joint damping for task control")
    
    parser.add_argument("--plot-rate", type=float, default=10.0, help="Live-plot refresh rate [Hz]")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    print("=" * 70)
    print(f"Trajectory Record & Replay Impedance Control [{args.mode.upper()} SPACE]")
    print(f"Config: {config_path}")
    print("=" * 70)

    system: Optional[FACTRGravityCompensation] = None
    running = True

    def _signal_handler(signum, frame):
        del signum, frame
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        n = int(system.num_arm_joints)

        t_start = time.time()
        last_step_t = time.perf_counter()
        
        phase = "PREP"
        trajectory_q: List[Tuple[float, np.ndarray, np.ndarray]] = []  # time_offset, q, dq
        
        print(f"[{phase}] Settle robot in gravity comp for {args.prep_time}s...")

        # Plotting states (used during replay)
        plot_state = None if args.no_plot else _init_plot(args.mode, args.record_time)
        plot_period = 1.0 / max(args.plot_rate, 1e-3)
        last_plot_t = time.time()
        
        t_buf = deque(maxlen=int(args.record_time * 500))
        ref_buf = deque(maxlen=int(args.record_time * 500))
        act_buf = deque(maxlen=int(args.record_time * 500))

        # Replay variables
        home_start_q = None
        target_q = None
        replay_start_t = 0.0

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf
            t_now = time.time() - t_start

            q, dq, _, _ = system.get_leader_joint_states()

            # Base torques
            tau_grav = system.gravity_compensation(q, dq)
            tau_fric = system.friction_compensation(dq)
            tau_cmd = np.zeros(n)
            
            # --- PHASE LOGIC ---
            if phase == "PREP":
                # Pure gravity compensation with slight damping
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_cmd = tau_grav + tau_fric + tau_damp
                
                if t_now >= args.prep_time:
                    phase = "RECORD"
                    print(f"[{phase}] Recording for {args.record_time}s! MOVE THE ARM NOW.")
                    record_start_t = t_now

            elif phase == "RECORD":
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                tau_cmd = tau_grav + tau_fric + tau_damp
                
                # Store relative time, q, dq
                t_rel = t_now - record_start_t
                trajectory_q.append((t_rel, q.copy(), dq.copy()))
                
                if t_rel >= args.record_time:
                    phase = "HOME"
                    print(f"[{phase}] Returning to start position for {args.home_time}s...")
                    home_start_t = t_now
                    home_start_q = q.copy()
                    target_q = trajectory_q[0][1]

            elif phase == "HOME":
                # Interpolate smoothly back to start position
                t_rel = t_now - home_start_t
                alpha = np.clip(t_rel / args.home_time, 0, 1)
                
                # Minimum jerk scaling
                alpha_smooth = 10 * (alpha**3) - 15 * (alpha**4) + 6 * (alpha**5)
                
                des_q = home_start_q + alpha_smooth * (target_q - home_start_q)
                
                # Joint space PD to move home robustly
                tau_pd = 20.0 * (des_q - q) - 2.0 * dq
                tau_cmd = tau_grav + tau_fric + tau_pd
                
                if t_rel >= args.home_time + 1.0: # add 1s to settle
                    phase = "REPLAY"
                    print(f"[{phase}] Replaying trajectory in [{args.mode.upper()}] space!")
                    replay_start_t = t_now

            elif phase == "REPLAY":
                t_rel = t_now - replay_start_t
                
                if t_rel > trajectory_q[-1][0]:
                    print("Replay finished! Holding at last position. Press Ctrl+C to exit.")
                    running = False
                    continue
                
                # Find the closest reference point by time
                # In a real setup, you'd interpolate, but with high sampling rate this is fine.
                idx = np.searchsorted([p[0] for p in trajectory_q], t_rel)
                idx = min(idx, len(trajectory_q) - 1)
                _, q_ref, dq_ref = trajectory_q[idx]

                if args.mode == "joint":
                    # Joint Space Impedance Control
                    tau_pd = args.kp_j * (q_ref - q) + args.kd_j * (dq_ref - dq)
                    tau_cmd = tau_grav + tau_fric + tau_pd
                    
                    t_buf.append(t_rel)
                    ref_buf.append(q_ref.copy())
                    act_buf.append(q.copy())

                elif args.mode == "task":
                    # Task Space Impedance Control
                    x_ref, J_v_ref = compute_task_kinematics(system, q_ref)
                    x_act, J_v_act = compute_task_kinematics(system, q)
                    
                    # Assume dx_ref is negligible or compute properly, simpler is dx = J dq
                    dx_ref = J_v_ref @ dq_ref
                    dx_act = J_v_act @ dq
                    
                    F_ts = args.kp_t * (x_ref - x_act) + args.kd_t * (dx_ref - dx_act)
                    tau_ts = J_v_act.T @ F_ts
                    
                    # Null space damping to keep arm stable
                    tau_null = -args.kd_null * dq
                    
                    tau_cmd = tau_grav + tau_fric + tau_ts + tau_null
                    
                    t_buf.append(t_rel)
                    ref_buf.append(x_ref.copy())
                    act_buf.append(x_act.copy())

                # Live Plotting
                wall_now = time.time()
                if (not args.no_plot) and (plot_state is not None) and (wall_now - last_plot_t >= plot_period):
                    _update_plot(
                        plot_state,
                        np.asarray(t_buf, dtype=float),
                        np.asarray(ref_buf, dtype=float),
                        np.asarray(act_buf, dtype=float),
                    )
                    last_plot_t = wall_now

            # Send Torques
            system.set_leader_joint_torque(tau_cmd, 0.0)

            # Sleep to maintain loop rate
            sleep_s = max(0.0, float(system.dt) - (time.perf_counter() - now_perf))
            if sleep_s > 0:
                time.sleep(sleep_s)

        print("Finished.")
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Test failed: {exc}")
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
            if 'plot_state' in locals() and plot_state is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"trajectory_replay_{args.mode}_{timestamp}.svg"
                plot_state["fig"].savefig(save_path, format="svg", bbox_inches="tight")
                print(f"\nSaved final 1x1 plot to {save_path}")
            plt.ioff()
            plt.close('all')

if __name__ == "__main__":
    raise SystemExit(main())
