#!/usr/bin/env python3
"""Manual BotaSys MiniOne readout smoke run.

This script intentionally avoids pytest naming conventions. It is meant for
bench bring-up of a USB-connected MiniOne without IMU usage before the sensor is
mounted on GELLO.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

_RUNNING = True


@dataclass(frozen=True)
class WrenchSample:
    host_t: float
    sensor_timestamp_us: int
    force: tuple[float, float, float]
    torque: tuple[float, float, float]
    temperature_c: float
    status_val: int
    throttled: bool
    overrange: bool
    invalid: bool
    raw: bool


def _handle_signal(signum: int, frame: object) -> None:
    del signum, frame
    global _RUNNING
    _RUNNING = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "bota_binary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual MiniOne wrench readout and live plotting utility."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to a bota-driver JSON config file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Run duration in seconds. Use <=0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=10.0,
        help="Time span shown in the live plot.",
    )
    parser.add_argument(
        "--nominal-rate",
        type=float,
        default=500.0,
        help="Only used to size local plotting/rate buffers.",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=2.0,
        help="Console status print frequency in Hz.",
    )
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=20.0,
        help="Maximum live plot refresh rate in Hz.",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=500,
        help="Number of recent inter-arrival samples used for rate statistics.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Use non-blocking read_frame() instead of read_frame_blocking().",
    )
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=500.0,
        help="Loop rate for --poll mode in Hz.",
    )
    parser.add_argument(
        "--no-tare",
        action="store_true",
        help="Skip tare in INACTIVE state before activation.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib live plot. Best for clean frequency readings.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help=(
            "Enable EMA wrench filtering with fixed alpha in (0, 1]. "
            "Set 0 to disable. Ignored when --ema-cutoff > 0."
        ),
    )
    parser.add_argument(
        "--ema-cutoff",
        type=float,
        default=0.0,
        help=(
            "Enable EMA wrench filtering using a cutoff frequency in Hz. "
            "Alpha is recomputed from measured dt. Overrides --ema-alpha."
        ),
    )
    parser.add_argument(
        "--ema-print-only",
        action="store_true",
        help="Print EMA values but keep the live plot on raw wrench values.",
    )
    return parser.parse_args()


def _frame_to_sample(frame: object, host_t: float) -> WrenchSample:
    status = frame.status
    return WrenchSample(
        host_t=host_t,
        sensor_timestamp_us=int(frame.timestamp),
        force=tuple(float(v) for v in frame.force[:3]),
        torque=tuple(float(v) for v in frame.torque[:3]),
        temperature_c=float(frame.temperature),
        status_val=int(getattr(status, "val", 0)),
        throttled=bool(status.throttled),
        overrange=bool(status.overrange),
        invalid=bool(status.invalid),
        raw=bool(status.raw),
    )


def _frequency_hz(intervals_s: Iterable[float]) -> float:
    values = [v for v in intervals_s if v > 0.0]
    if not values:
        return 0.0
    return len(values) / sum(values)


def _format_vec(values: tuple[float, float, float], unit: str) -> str:
    return f"[{values[0]: .4f}, {values[1]: .4f}, {values[2]: .4f}] {unit}"


def _print_sample(sample: WrenchSample, avg_hz: float, expected_hz: float | None) -> None:
    expected = f", expected={expected_hz:.2f} Hz" if expected_hz else ""
    print(
        " | ".join(
            [
                f"rate={avg_hz:.2f} Hz{expected}",
                f"F={_format_vec(sample.force, 'N')}",
                f"T={_format_vec(sample.torque, 'Nm')}",
                f"temp={sample.temperature_c:.2f} C",
                (
                    "status="
                    f"val={sample.status_val} "
                    f"thr={sample.throttled} over={sample.overrange} "
                    f"invalid={sample.invalid} raw={sample.raw}"
                ),
                f"sensor_ts={sample.sensor_timestamp_us} us",
            ]
        ),
        flush=True,
    )


def _print_filtered_sample(
    raw_sample: WrenchSample,
    filtered_sample: WrenchSample,
    avg_hz: float,
    expected_hz: float | None,
) -> None:
    expected = f", expected={expected_hz:.2f} Hz" if expected_hz else ""
    print(
        " | ".join(
            [
                f"rate={avg_hz:.2f} Hz{expected}",
                f"F={_format_vec(raw_sample.force, 'N')}",
                f"T={_format_vec(raw_sample.torque, 'Nm')}",
                f"F_ema={_format_vec(filtered_sample.force, 'N')}",
                f"T_ema={_format_vec(filtered_sample.torque, 'Nm')}",
                f"temp={raw_sample.temperature_c:.2f} C",
                (
                    "status="
                    f"val={raw_sample.status_val} "
                    f"thr={raw_sample.throttled} over={raw_sample.overrange} "
                    f"invalid={raw_sample.invalid} raw={raw_sample.raw}"
                ),
                f"sensor_ts={raw_sample.sensor_timestamp_us} us",
            ]
        ),
        flush=True,
    )


def _ema_alpha_from_cutoff(cutoff_hz: float, dt_s: float) -> float:
    if cutoff_hz <= 0.0 or dt_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)


def _filtered_sample(
    sample: WrenchSample,
    previous: WrenchSample | None,
    alpha: float,
) -> WrenchSample:
    if previous is None:
        return sample

    a = min(max(float(alpha), 0.0), 1.0)
    force = tuple(
        previous.force[idx] + a * (sample.force[idx] - previous.force[idx])
        for idx in range(3)
    )
    torque = tuple(
        previous.torque[idx] + a * (sample.torque[idx] - previous.torque[idx])
        for idx in range(3)
    )
    return WrenchSample(
        host_t=sample.host_t,
        sensor_timestamp_us=sample.sensor_timestamp_us,
        force=force,
        torque=torque,
        temperature_c=sample.temperature_c,
        status_val=sample.status_val,
        throttled=sample.throttled,
        overrange=sample.overrange,
        invalid=sample.invalid,
        raw=sample.raw,
    )


class LiveWrenchPlot:
    def __init__(self, buffer_seconds: float) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._buffer_seconds = float(buffer_seconds)
        self._fig, (self._force_ax, self._torque_ax) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7)
        )
        self._fig.canvas.manager.set_window_title("BotaSys MiniOne wrench")
        self._force_lines = tuple(
            self._force_ax.plot([], [], label=label)[0] for label in ("Fx", "Fy", "Fz")
        )
        self._torque_lines = tuple(
            self._torque_ax.plot([], [], label=label)[0] for label in ("Tx", "Ty", "Tz")
        )
        self._force_ax.set_ylabel("Force [N]")
        self._torque_ax.set_ylabel("Torque [Nm]")
        self._torque_ax.set_xlabel("Time [s]")
        self._force_ax.grid(True)
        self._torque_ax.grid(True)
        self._force_ax.legend(loc="upper right")
        self._torque_ax.legend(loc="upper right")
        self._plt.ion()
        self._plt.show(block=False)

    def update(self, samples: Deque[WrenchSample], start_t: float) -> None:
        if not samples:
            return

        t = [sample.host_t - start_t for sample in samples]
        force = list(zip(*(sample.force for sample in samples)))
        torque = list(zip(*(sample.torque for sample in samples)))

        for idx, line in enumerate(self._force_lines):
            line.set_data(t, force[idx])
        for idx, line in enumerate(self._torque_lines):
            line.set_data(t, torque[idx])

        x_max = max(t[-1], self._buffer_seconds)
        x_min = max(0.0, x_max - self._buffer_seconds)
        self._force_ax.set_xlim(x_min, x_max)
        self._torque_ax.set_xlim(x_min, x_max)
        self._force_ax.relim()
        self._force_ax.autoscale_view(scalex=False, scaley=True)
        self._torque_ax.relim()
        self._torque_ax.autoscale_view(scalex=False, scaley=True)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)


def _expected_rate(driver: object) -> float | None:
    try:
        timestep = driver.get_expected_timestep()
    except Exception:
        return None
    seconds = float(timestep.total_seconds())
    if seconds <= 0.0:
        return None
    return 1.0 / seconds


def _load_bota_driver() -> object:
    try:
        import bota_driver
    except ImportError as exc:  # pragma: no cover - manual hardware utility
        raise RuntimeError(
            "Could not import bota_driver. Install/activate the environment that "
            "contains the Bota Systems Python driver."
        ) from exc
    return bota_driver


def _open_driver(config_path: Path, tare: bool) -> object:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass --config /path/to/bota.json."
        )

    bota_driver = _load_bota_driver()
    driver = bota_driver.BotaDriver(str(config_path))
    print(f"BotaDriver version: {driver.get_driver_version_string()}")
    print(f"Initial state: {driver.get_driver_state()}")

    if not driver.configure():
        raise RuntimeError("Failed to configure Bota driver")
    print(f"Configured state: {driver.get_driver_state()}")

    if tare:
        print("Taring sensor in INACTIVE state...")
        if not driver.tare():
            raise RuntimeError("Failed to tare MiniOne")

    if not driver.activate():
        raise RuntimeError("Failed to activate Bota driver")
    print(f"Active state: {driver.get_driver_state()}")
    return driver


def _state_name(state: object) -> str:
    return str(getattr(state, "name", state))


def _close_driver(driver: object | None) -> None:
    if driver is None:
        return

    try:
        state = _state_name(driver.get_driver_state())
        if state.endswith("ACTIVE") and not state.endswith("INACTIVE"):
            if not driver.deactivate():
                print("WARNING: Bota driver deactivate() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver deactivate failed: {exc}", file=sys.stderr)

    try:
        state = _state_name(driver.get_driver_state())
        if state.endswith("INACTIVE"):
            if not driver.cleanup():
                print("WARNING: Bota driver cleanup() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver cleanup failed: {exc}", file=sys.stderr)

    try:
        if not driver.shutdown():
            print("WARNING: Bota driver shutdown() returned false", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Bota driver shutdown failed: {exc}", file=sys.stderr)


def _read_once(driver: object, poll: bool) -> object:
    if poll:
        return driver.read_frame()
    return driver.read_frame_blocking()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    max_samples = max(10, int(args.buffer_seconds * args.nominal_rate))
    samples: Deque[WrenchSample] = deque(maxlen=max_samples)
    plot_samples: Deque[WrenchSample] = deque(maxlen=max_samples)
    intervals_s: Deque[float] = deque(maxlen=max(2, int(args.rate_window)))
    ema_enabled = args.ema_cutoff > 0.0 or args.ema_alpha > 0.0
    filtered_sample: WrenchSample | None = None
    last_filter_sensor_ts: int | None = None

    plot = None
    if not args.no_plot:
        try:
            plot = LiveWrenchPlot(args.buffer_seconds)
        except Exception as exc:
            print(f"Live plot disabled: {exc}", file=sys.stderr)

    driver = None
    start_t = time.perf_counter()
    last_read_t: float | None = None
    last_print_t = start_t
    last_plot_t = start_t
    next_poll_t = start_t

    exit_code = 0
    try:
        driver = _open_driver(args.config.expanduser().resolve(), tare=not args.no_tare)
        expected_hz = _expected_rate(driver)
        if expected_hz:
            print(f"Driver expected timestep: {1000.0 / expected_hz:.3f} ms ({expected_hz:.2f} Hz)")

        while _RUNNING:
            now = time.perf_counter()
            if args.duration > 0.0 and now - start_t >= args.duration:
                break

            if args.poll:
                next_poll_t += 1.0 / max(args.poll_rate, 1e-6)

            frame = _read_once(driver, args.poll)
            read_t = time.perf_counter()
            sample = _frame_to_sample(frame, read_t)
            samples.append(sample)

            if last_read_t is not None:
                intervals_s.append(read_t - last_read_t)
            dt_s = 0.0 if last_read_t is None else read_t - last_read_t
            last_read_t = read_t

            if ema_enabled:
                is_new_sensor_frame = sample.sensor_timestamp_us != last_filter_sensor_ts
                if is_new_sensor_frame:
                    if last_filter_sensor_ts is None:
                        filter_dt_s = 0.0
                    else:
                        filter_dt_s = max(
                            0.0,
                            (sample.sensor_timestamp_us - last_filter_sensor_ts) * 1e-6,
                        )
                    if args.ema_cutoff > 0.0:
                        alpha = _ema_alpha_from_cutoff(args.ema_cutoff, filter_dt_s)
                    else:
                        alpha = args.ema_alpha
                    filtered_sample = _filtered_sample(sample, filtered_sample, alpha)
                    last_filter_sensor_ts = sample.sensor_timestamp_us
                plot_sample = sample if args.ema_print_only else filtered_sample or sample
            else:
                plot_sample = sample
            plot_samples.append(plot_sample)

            if read_t - last_print_t >= 1.0 / max(args.print_rate, 1e-6):
                if ema_enabled and filtered_sample is not None:
                    _print_filtered_sample(
                        sample,
                        filtered_sample,
                        _frequency_hz(intervals_s),
                        expected_hz,
                    )
                else:
                    _print_sample(sample, _frequency_hz(intervals_s), expected_hz)
                last_print_t = read_t

            if plot and read_t - last_plot_t >= 1.0 / max(args.plot_rate, 1e-6):
                plot.update(plot_samples, start_t)
                last_plot_t = read_t

            if args.poll:
                sleep_s = max(0.0, next_poll_t - time.perf_counter())
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    except Exception as exc:
        exit_code = 1
        print(f"FATAL: {exc}", file=sys.stderr)
    finally:
        total_s = max(time.perf_counter() - start_t, 1e-9)
        _close_driver(driver)
        print(
            f"Finished: samples={len(samples)}, elapsed={total_s:.3f} s, "
            f"recent_rate={_frequency_hz(intervals_s):.2f} Hz"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
