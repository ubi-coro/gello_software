from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    plt = None

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
from gello.cr_dagger.core.intervention_detector import (
    DeltaPositionParams,
    EnergyInjectionParams,
    FusedDetectorParams,
    FusedInterventionDetector,
    TorqueThresholdParams,
    WrenchChangeParams,
)
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CR-DAgger intervention detection test")
    p.add_argument("--config", type=str, default="configs/ur5e_gello_factr_hw_V2.yaml")
    p.add_argument("--prep-time", type=float, default=3.0)
    p.add_argument("--record-time", type=float, default=5.0)
    p.add_argument("--home-time", type=float, default=4.0)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=20.0)
    p.add_argument("--torque-threshold", type=float, default=0.3)
    p.add_argument("--delta-threshold", type=float, default=0.02)
    p.add_argument("--energy-threshold", type=float, default=0.005)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _intervention_spans(t: np.ndarray, flags: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if len(t) == 0 or len(flags) == 0:
        return spans
    start = None
    for i, active in enumerate(flags.astype(bool)):
        if active and start is None:
            start = float(t[i])
        elif (not active) and start is not None:
            spans.append((start, float(t[i])))
            start = None
    if start is not None:
        spans.append((start, float(t[-1])))
    return spans


def _save_plot(
    t_hist: np.ndarray,
    q_ref_hist: np.ndarray,
    q_act_hist: np.ndarray,
    q_c_hist: np.ndarray,
    is_corr_hist: np.ndarray,
) -> None:
    if plt is None or t_hist.size == 0:
        return

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    spans = _intervention_spans(t_hist, is_corr_hist)

    for j in range(min(6, q_ref_hist.shape[1])):
        ax = axes[j]
        for t0, t1 in spans:
            ax.axvspan(t0, t1, color="red", alpha=0.12)
        ax.plot(t_hist, q_ref_hist[:, j], "--", lw=1.2, label="q_ref")
        ax.plot(t_hist, q_act_hist[:, j], "-", lw=1.2, label="q_actual")
        ax.plot(t_hist, q_c_hist[:, j], "-", lw=1.2, label="q_compliant")
        ax.set_ylabel(f"J{j + 1} [rad]")
        ax.grid(True, ls="--", alpha=0.4)
        if j == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Replay Time [s]")
    fig.tight_layout()

    out = Path.cwd() / f"intervention_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
    fig.savefig(str(out), format="svg", bbox_inches="tight")
    print(f"Saved plot: {out}")
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    system: Optional[FACTRGravityCompensation] = None
    running = True
    n = 6

    def _sig(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    t_buf = deque(maxlen=50000)
    q_ref_buf = deque(maxlen=50000)
    q_act_buf = deque(maxlen=50000)
    q_c_buf = deque(maxlen=50000)
    corr_buf = deque(maxlen=50000)

    try:
        system = FACTRGravityCompensation(str(config_path), enable_visualization=False)
        n = int(system.num_arm_joints)
        if system.driver is None:
            raise RuntimeError("Dynamixel driver is not available")

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

        detector = FusedInterventionDetector(
            params=FusedDetectorParams(
                torque=TorqueThresholdParams(
                    threshold_high=float(args.torque_threshold),
                    threshold_low=float(args.torque_threshold) * 0.5,
                ),
                delta=DeltaPositionParams(position_threshold=float(args.delta_threshold)),
                energy=EnergyInjectionParams(energy_threshold=float(args.energy_threshold)),
                wrench=WrenchChangeParams(force_threshold=3.0),
                min_votes=int(args.min_votes),
            ),
            n_joints=n,
            dt=float(system.dt),
        )

        params = AdmittanceParams(
            mass=float(args.mass),
            damping=float(args.damping),
            stiffness=float(args.stiffness),
        )

        phase = "PREP"
        phase_start = time.perf_counter()
        last_step_t = time.perf_counter()
        step_count = 0

        trajectory_q: list[tuple[float, np.ndarray, np.ndarray]] = []
        home_start_q = np.zeros(n)
        target_q = np.zeros(n)
        admittance: JointSpaceAdmittanceController | None = None

        print(f"[PREP] settle for {args.prep_time:.1f}s")

        while running:
            now_perf = time.perf_counter()
            measured_dt = max(now_perf - last_step_t, 1e-4)
            last_step_t = now_perf

            q, dq, _, _ = system.get_leader_joint_states()
            currents_all = system.driver.get_currents()
            currents_arm = currents_all[:n] * system.joint_signs[:n]
            tau_ext = shi.update(q, dq, currents_arm)

            if phase == "PREP":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)

                if now_perf - phase_start >= float(args.prep_time):
                    phase = "RECORD"
                    phase_start = now_perf
                    print(f"[RECORD] moving for {args.record_time:.1f}s")

            elif phase == "RECORD":
                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_damp = -float(system.gravity_comp_velocity_damping) * dq
                system.set_leader_joint_torque(tau_grav + tau_fric + tau_damp, 0.0)

                t_rel = now_perf - phase_start
                trajectory_q.append((t_rel, q.copy(), dq.copy()))

                if t_rel >= float(args.record_time):
                    if len(trajectory_q) == 0:
                        print("No trajectory recorded; aborting.")
                        return 1
                    phase = "HOME"
                    phase_start = now_perf
                    home_start_q = q.copy()
                    target_q = trajectory_q[0][1].copy()
                    print(f"[HOME] returning to start for {args.home_time:.1f}s + settle")

            elif phase == "HOME":
                t_rel = now_perf - phase_start
                alpha = np.clip(t_rel / float(args.home_time), 0.0, 1.0)
                s = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
                des_q = home_start_q + s * (target_q - home_start_q)

                tau_grav = system.gravity_compensation(q, dq)
                tau_fric = system.friction_compensation(dq)
                tau_cmd = tau_grav + tau_fric + 20.0 * (des_q - q) - 2.0 * dq
                system.set_leader_joint_torque(tau_cmd, 0.0)

                if t_rel >= float(args.home_time) + 1.0:
                    print("[REPLAY] switching to position mode")
                    system.driver.set_torque_mode(False)
                    time.sleep(0.05)
                    system.driver.set_operating_mode(3)
                    time.sleep(0.05)
                    system.driver.set_torque_mode(True)
                    time.sleep(0.05)
                    shi.reset()
                    detector.reset()

                    admittance = JointSpaceAdmittanceController(
                        params=params,
                        n_joints=n,
                        q_init=trajectory_q[0][1].copy(),
                        q_min=system.arm_joint_limits_min,
                        q_max=system.arm_joint_limits_max,
                    )
                    phase = "REPLAY"
                    phase_start = now_perf
                    step_count = 0

            elif phase == "REPLAY":
                if admittance is None:
                    raise RuntimeError("Admittance controller not initialized")

                t_rel = now_perf - phase_start
                if t_rel > trajectory_q[-1][0]:
                    print("Replay complete.")
                    break

                times = [p[0] for p in trajectory_q]
                idx = int(np.searchsorted(times, t_rel))
                idx = min(idx, len(trajectory_q) - 1)
                _, q_ref, _ = trajectory_q[idx]

                q_c = admittance.step(q_ref=q_ref, tau_ext_human=tau_ext, dt=measured_dt)

                is_intervening = detector.update(
                    tau_ext=tau_ext,
                    q_c=admittance.q_c,
                    q_ref=q_ref,
                    dq_c=admittance.dq_c,
                    wrench=None,
                )
                diag = detector.get_diagnostics()
                votes = diag.get("votes", {})
                vote_delta = votes.get("delta_q", votes.get("delta", False))

                target_hw = np.zeros(system.num_motors)
                target_hw[:n] = q_c * system.joint_signs[:n] + system.joint_offsets[:n]
                if system.num_motors > n:
                    target_hw[-1] = system.leader_gripper_raw_rad
                system.driver.set_joints(target_hw.tolist())

                t_buf.append(t_rel)
                q_ref_buf.append(q_ref.copy())
                q_act_buf.append(q.copy())
                q_c_buf.append(q_c.copy())
                corr_buf.append(bool(is_intervening))

                step_count += 1
                if step_count % 100 == 0:
                    print(
                        f"[{t_rel:6.2f}s] "
                        f"INT={'YES' if is_intervening else 'no ':3s} "
                        f"conf={diag.get('confidence', 0.0):.2f} "
                        f"votes: T={votes.get('torque', False)} "
                        f"D={vote_delta} "
                        f"E={votes.get('energy', False)} "
                        f"W={votes.get('wrench', False)} "
                        f"energy={diag.get('energy_level', 0.0):.4f} "
                        f"|tau_ext|={np.linalg.norm(tau_ext):.3f} "
                        f"|delta_q|={np.linalg.norm(admittance.get_delta_q()):.4f}"
                    )

            loop_elapsed = time.perf_counter() - now_perf
            sleep_s = max(0.0, float(system.dt) - loop_elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        if not args.no_plot:
            _save_plot(
                np.asarray(t_buf, dtype=float),
                np.asarray(q_ref_buf, dtype=float),
                np.asarray(q_act_buf, dtype=float),
                np.asarray(q_c_buf, dtype=float),
                np.asarray(corr_buf, dtype=bool),
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
    raise SystemExit(main())
