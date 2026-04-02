#!/usr/bin/env python3
"""Stage-1 observer comparison on real GELLO hardware.

Runs GELLO in gravity-comp hold mode, reads both observers in parallel:
- Shi minimalist estimator (needs Present Current)
- Yamane observer (needs tau_cmd + q)

Use case:
1) Start script
2) Let arm settle in gravity compensation
3) Hang known weight (e.g. 0.5 kg) at EE
4) Compare tau_ext traces / norms live
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import multiprocessing as mp
import queue

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gello.bilat_4ch.gello_ur5e_observer_shi import (  # noqa: E402
    InterventionDetector,
    MinimalistEstimatorConfig,
    MinimalistTorqueEstimator,
    MotorParams,
    MotorType,
)
from gello.bilat_4ch.gello_ur5e_observer_yamane import LeaderObserver  # noqa: E402
from gello.factr.gravity_compensation import FACTRGravityCompensation  # noqa: E402


def _resolve_leader_urdf(config_path: Path, leader_urdf: str) -> Path:
    candidate_1 = (config_path.parent / leader_urdf).resolve()
    if candidate_1.exists():
        return candidate_1

    candidate_2 = (REPO_ROOT / "gello" / "factr" / "urdf" / Path(leader_urdf).name).resolve()
    if candidate_2.exists():
        return candidate_2

    candidate_3 = (REPO_ROOT / leader_urdf).resolve()
    if candidate_3.exists():
        return candidate_3

    raise FileNotFoundError(f"URDF not found for leader_urdf='{leader_urdf}'")


def _infer_motor_params(servo_types: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    # Values mapping: (gear_ratio, Kt_motor)
    # kt_motor ≈ Kt_effective / gear_ratio, where Kt_effective = stall_torque / stall_current
    params_by_servo = {
        "XC330_T288_T": (288.35, 1.136 / 288.35),
        "XM430_W210_T": (212.6, 1.304 / 212.6),
        "XM430_W350_T": (353.5, 1.783 / 353.5),
    }
    gear_ratios = []
    kts = []
    for s in servo_types[:n]:
        gr, kt = params_by_servo.get(s, (1.0, 0.00504))
        gear_ratios.append(float(gr))
        kts.append(float(kt))
    if len(gear_ratios) < n:
        gear_ratios.extend([1.0] * (n - len(gear_ratios)))
        kts.extend([0.00504] * (n - len(kts)))
    return np.asarray(gear_ratios, dtype=float), np.asarray(kts, dtype=float)


def _init_plot(n_joints: int, window_s: float, observer_mode: str, plt):

    if plt is None:
        return None

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2

    plt.ion()
    n_subplots = 2 if observer_mode == "both" else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 6), sharey=True)
    if n_subplots == 1:
        axes = [axes]
        
    fig.suptitle(f"Observer Comparison ({observer_mode})", fontweight="bold")

    lines_shi = [] if observer_mode in ["both", "shi"] else None
    lines_yam = [] if observer_mode in ["both", "yamane"] else None
    
    ax_idx = 0
    if observer_mode in ["both", "yamane"]:
        ax_yam = axes[ax_idx]
        for j in range(n_joints):
            (line_yam,) = ax_yam.plot([], [], linewidth=1.5, label=f"Joint {j+1}")
            lines_yam.append(line_yam)
        ax_yam.set_title("Yamane Observer")
        ax_yam.set_ylabel(r"External Torque $	tau_{\text{ext}}$ (Nm)")
        ax_yam.set_xlabel(f"Time window [{window_s:.0f}s]")
        ax_yam.grid(True, linestyle="--", alpha=0.5)
        ax_yam.legend(loc="upper left", fontsize=10)
        ax_idx += 1

    if observer_mode in ["both", "shi"]:
        ax_shi = axes[ax_idx]
        for j in range(n_joints):
            (line_shi,) = ax_shi.plot([], [], linewidth=1.5, label=f"Joint {j+1}")
            lines_shi.append(line_shi)
        ax_shi.set_title("Minimalist Observer (Shi)")
        if ax_idx == 0:
            ax_shi.set_ylabel(r"External Torque $	tau_{\text{ext}}$ (Nm)")
        ax_shi.set_xlabel(f"Time window [{window_s:.0f}s]")
        ax_shi.grid(True, linestyle="--", alpha=0.5)
        ax_shi.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    return {
        "fig": fig,
        "axes": axes,
        "lines_shi": lines_shi,
        "lines_yam": lines_yam,
    }


def _update_plot(
    plot_state,
    n_joints: int,
    t_hist: np.ndarray,
    shi_hist,
    yam_hist,
):
    if plot_state is None or t_hist.size == 0:
        return

    t0 = t_hist[0]
    tx = t_hist - t0

    for j in range(n_joints):
        if plot_state["lines_shi"] is not None and shi_hist is not None:
            plot_state["lines_shi"][j].set_data(tx, shi_hist[:, j])
        if plot_state["lines_yam"] is not None and yam_hist is not None:
            plot_state["lines_yam"][j].set_data(tx, yam_hist[:, j])

    for ax in plot_state["axes"]:
        ax.set_xlim(tx[0], tx[-1] if tx[-1] > 1e-3 else 1.0)
        ax.relim()
        ax.autoscale_view(scaley=True)

    plot_state["fig"].canvas.draw_idle()
    plot_state["fig"].canvas.flush_events()


def _visualization_worker(queue, n_joints, window_s, observer_mode):
    import matplotlib.pyplot as plt
    try:
        plot_state = _init_plot(n_joints, window_s, observer_mode, plt)
        while True:
            data = None
            try:
                while True:
                    data = queue.get_nowait()
            except Exception: # queue.Empty
                pass
            
            if data == "QUIT":
                break
                
            if data is not None:
                t_hist, shi_hist, yam_hist = data
                _update_plot(plot_state, n_joints, t_hist, shi_hist, yam_hist)
                
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Visualization error: {e}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observer comparison with known payload")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ur5e_gello_factr_hw_V2.yaml",
        help="Path to FACTR YAML config",
    )
    parser.add_argument("--weight-kg", type=float, default=0.5, help="Known payload for experiment notes")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = until Ctrl+C)")
    parser.add_argument("--observer", choices=["both", "shi", "yamane"], default="both", help="Which observer(s) to run and plot")

    parser.add_argument("--shi-kt", type=float, default=0.00504, help="Shi kt default for all joints [Nm/A motor-side]")
    parser.add_argument("--shi-eta", type=float, default=0.65, help="Shi eta default for all joints")
    parser.add_argument("--shi-alpha", type=float, default=0.5, help="Shi EMA alpha")
    parser.add_argument("--shi-vel-threshold", type=float, default=0.05, help="Shi vel threshold [rad/s]")

    parser.add_argument("--yamane-omega-c", type=float, default=25.0, help="Yamane observer cutoff [rad/s]")
    parser.add_argument("--yamane-zeta", type=float, default=1.0, help="Yamane damping ratio")

    parser.add_argument("--plot-window", type=float, default=15.0, help="Live-plot window size [s]")
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
    print("Stage-1 observer comparison")
    print(f"Config: {config_path}")
    print(f"Known payload note: {args.weight_kg:.3f} kg")
    print("Running in gravity-comp hold with both observers in parallel")
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
        servo_types = list(system.config["dynamixel"]["servo_types"])
        gear_ratio, kt = _infer_motor_params(servo_types, n)

        leader_urdf = str(system.config["arm_teleop"]["leader_urdf"])
        urdf_path = _resolve_leader_urdf(config_path, leader_urdf)

        motor_params = MotorParams(
            kt=kt,
            gear_ratio=gear_ratio,
            eta=np.full(n, float(args.shi_eta), dtype=float),
            motor_type=MotorType.CURRENT,
        )

        shi = None
        detector = None
        if args.observer in ["both", "shi"]:
            shi = MinimalistTorqueEstimator(
                MinimalistEstimatorConfig(
                    urdf_path=str(urdf_path),
                    motor_params=motor_params,
                    alpha_ema=float(args.shi_alpha),
                    vel_threshold=float(args.shi_vel_threshold),
                )
            )
            detector = InterventionDetector(
                n_joints=n,
                torque_threshold=0.3,
                activation_count=3,
                deactivation_count=5,
            )

        yamane = None
        if args.observer in ["both", "yamane"]:
            yamane = LeaderObserver(
                urdf_path=str(urdf_path),
                omega_c=float(args.yamane_omega_c),
                dt=float(system.dt),
                q0=system.calibration_joint_pos[:n],
                zeta=float(args.yamane_zeta),
            )

        maxlen = max(100, int(args.plot_window / max(system.dt, 1e-4)))
        t_buf = deque(maxlen=maxlen)
        shi_buf = deque(maxlen=maxlen)
        yam_buf = deque(maxlen=maxlen)
        norm_shi_buf = deque(maxlen=maxlen)
        norm_yam_buf = deque(maxlen=maxlen)
        det_buf = deque(maxlen=maxlen)

        viz_queue = None
        viz_process = None
        if not args.no_plot:
            viz_queue = mp.Queue()
            viz_process = mp.Process(target=_visualization_worker, args=(viz_queue, n, args.plot_window, args.observer), daemon=True)
            viz_process.start()
        plot_period = 1.0 / max(args.plot_rate, 1e-3)

        t_start = time.time()
        last_step_t = time.perf_counter()
        last_plot_t = t_start
        loop_counter = 0

        print("Observer loop started. Press Ctrl+C to stop.")

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf

            q, dq, _, _ = system.get_leader_joint_states()

            tau_grav = system.gravity_compensation(q, dq)
            tau_fric = system.friction_compensation(dq)
            tau_damp = -float(system.gravity_comp_velocity_damping) * dq
            tau_cmd = tau_grav + tau_fric + tau_damp
            system.set_leader_joint_torque(tau_cmd, 0.0)

            currents_all = system.driver.get_currents() if system.driver is not None else np.zeros(n)
            currents_arm = currents_all[:n] * system.joint_signs[:n]

            tau_ext_shi = np.zeros(n)
            tau_ext_yam = np.zeros(n)
            intervention_active = False

            if shi is not None:
                tau_ext_shi = shi.update(q, dq, currents_arm)
                intervention_active = detector.update(tau_ext_shi)
            
            if yamane is not None:
                _, tau_ext_yam = yamane.update(q_measured=q, tau_cmd=tau_cmd, dt=measured_dt)

            t_now = time.time() - t_start
            t_buf.append(t_now)
            if shi is not None:
                shi_buf.append(tau_ext_shi.copy())
                norm_shi_buf.append(float(np.linalg.norm(tau_ext_shi)))
                det_buf.append(1.0 if intervention_active else 0.0)
            if yamane is not None:
                yam_buf.append(tau_ext_yam.copy())
                norm_yam_buf.append(float(np.linalg.norm(tau_ext_yam)))

            loop_counter += 1
            if loop_counter % 100 == 0:
                msg = f"t={t_now:7.2f}s | "
                if shi is not None:
                    msg += f"||shi||={norm_shi_buf[-1]:6.3f} Nm | intervention={int(intervention_active)} | "
                if yamane is not None:
                    msg += f"||yam||={norm_yam_buf[-1]:6.3f} Nm"
                print(msg.strip(" | "))

            wall_now = time.time()
            if (not args.no_plot) and (viz_queue is not None) and (wall_now - last_plot_t >= plot_period):
                t_hist = np.asarray(t_buf, dtype=float)
                shi_hist = np.asarray(shi_buf, dtype=float) if shi else None
                yam_hist = np.asarray(yam_buf, dtype=float) if yamane else None
                try:
                    viz_queue.put_nowait((t_hist, shi_hist, yam_hist))
                except Exception:
                    pass
                last_plot_t = wall_now

            if args.duration > 0.0 and t_now >= args.duration:
                running = False

            sleep_s = max(0.0, float(system.dt) - (time.perf_counter() - now_perf))
            if sleep_s > 0:
                time.sleep(sleep_s)

        print("Stopping observer loop...")
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Observer test failed: {exc}")
        return 1
    finally:
        if system is not None:
            try:
                system.shutdown()
            except Exception as exc:
                print(f"Shutdown warning: {exc}")
        if not args.no_plot and 'viz_queue' in locals() and viz_queue is not None:
            try:
                viz_queue.put_nowait("QUIT")
            except Exception:
                pass
        if not args.no_plot and 'viz_process' in locals() and viz_process is not None:
            try:
                viz_process.join(timeout=1.0)
                if viz_process.is_alive():
                    viz_process.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
