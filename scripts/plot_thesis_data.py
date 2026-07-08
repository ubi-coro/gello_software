#!/usr/bin/env python3
"""Generate thesis plots from CR-DAgger/BOTA NPZ logs.

The script is intentionally defensive: it scans the exported data folders,
selects representative files by filename patterns, and skips plots whose
required signals are not present. This makes it usable across intermediate
experiment dumps without hand-editing paths for every run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "Arial",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_LABELS = [f"J{i + 1}" for i in range(6)]
WRENCH_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
EXCLUDE_NAME_RE = re.compile(r"(mapped_axis_check|raw_no_gello|\btest_)", re.IGNORECASE)
GROUP_BY_PLOT = True
SHOW_FIGURES = False
NO_SAVE = False


@dataclass(frozen=True)
class LoadedNpz:
    path: Path
    data: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create prioritized thesis figures from 20_Daten NPZ logs."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "20_Daten")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "20_Daten" / "thesis_plots")
    parser.add_argument(
        "--plots",
        choices=[
            "core",
            "priority2",
            "all",
            "thesis",
            "minimum",
            "optional",
            "impedance",
            "impedance_appendix",
            "observer_noise",
            "observer_debug",
            "observer_admittance",
            "observer_attempts",
            "bota_validation",
            "bota_admittance",
            "phaseb_timing",
            "phaseb_tracking_histogram",
            "phaseb_timing_tracking_histogram",
            "correction_signal",
            "intervention_detection",
            "whiteboard_task",
            "whiteboard_policy_debug",
            "whiteboard_j3_debug",
            "cable_routing_takeover",
        ],
        default="thesis",
        help=(
            "thesis=final A-D + 1-8 list, minimum=A-D + 1/4/6/7, "
            "optional=optional 9-10, impedance=single J1 impedance tracking plot, "
            "impedance_appendix=stacked J1/J2/J3 impedance plot, "
            "observer_noise=sensorless static-hold observer residual plot, "
            "observer_debug=per-file observer residual and leader joint debug plots, "
            "observer_admittance=observer-driven admittance static-hold comparison, "
            "observer_attempts=compact sensorless observer failure-mode plot, "
            "bota_validation=compact BOTA nominal-vs-push validation plot, "
            "bota_admittance=BOTA admittance response and tracking plot, "
            "phaseb_timing=Phase-B loop and policy-action timing histogram, "
            "phaseb_tracking_histogram=pooled UR5e tracking-error histogram for cable-routing runs, "
            "phaseb_timing_tracking_histogram=Phase-B timing plot with pooled tracking-error histogram, "
            "correction_signal=Phase-B static-hold correction signal with/without intervention, "
            "intervention_detection=Phase-B intervention detection pipeline and release close-up, "
            "whiteboard_task=representative whiteboard-wiping Phase-B task plot, "
            "whiteboard_policy_debug=six-axis policy/leader/follower debug plot for whiteboard wiping, "
            "whiteboard_j3_debug=zoomed J3 policy/leader/follower debug plot for whiteboard wiping, "
            "cable_routing_takeover=appendix and focused cable-routing smooth-takeover plots, "
            "core/priority2/all keep the earlier list."
        ),
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--timeline-pattern",
        default="phaseB_calibrated_bota_static_hold_interventions",
        help="Filename/folder substring used for the Phase-B timeline plot.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap per aggregate plot. 0 means use all matching files.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write files directly into --out-dir instead of one subfolder per plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open plot windows for visual inspection.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write figure files. Intended for use with --show.",
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        nargs="+",
        default=[],
        help=(
            "Explicit NPZ file(s) or directory/directories for single-file plot modes "
            "such as --plots impedance. Directories are scanned recursively."
        ),
    )
    parser.add_argument(
        "--aggregate-inputs",
        action="store_true",
        help="Aggregate multiple --input-npz files into one mean/std plot when supported.",
    )
    parser.add_argument(
        "--plot-title",
        default="",
        help="Override the figure title for single-file plot modes.",
    )
    parser.add_argument(
        "--debug-joint",
        type=int,
        default=3,
        help="Joint number for zoom debug plots, 1-based. Default: 3.",
    )
    parser.add_argument(
        "--debug-t-start",
        type=float,
        default=0.0,
        help="Start time [s] for zoom debug plots.",
    )
    parser.add_argument(
        "--debug-t-end",
        type=float,
        default=20.0,
        help="End time [s] for zoom debug plots.",
    )
    parser.add_argument(
        "--debug-all-joints",
        action="store_true",
        help="Show all six joints in the policy debug plot instead of a single zoomed joint.",
    )
    parser.add_argument(
        "--invert-leader",
        action="store_true",
        help="Invert leader command/actual traces before start-value normalisation in policy debug plots.",
    )
    return parser.parse_args()


def find_npz(root: Path, include: Sequence[str] = (), exclude_debug: bool = True) -> list[Path]:
    if not root.exists():
        return []
    files = sorted(p for p in root.rglob("*.npz") if p.is_file())
    out: list[Path] = []
    inc = [s.lower() for s in include if s]
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/").lower()
        if exclude_debug and EXCLUDE_NAME_RE.search(rel):
            continue
        if inc and not all(s in rel for s in inc):
            continue
        out.append(path)
    return out


def load_npz(path: Path) -> LoadedNpz | None:
    try:
        return LoadedNpz(path=path, data=np.load(path, allow_pickle=True))
    except Exception as exc:
        print(f"[skip] cannot load {path}: {exc}", file=sys.stderr)
        return None


def keys(npz: object) -> set[str]:
    return set(getattr(npz, "files", []))


def get_array(npz: object, candidates: Sequence[str]) -> np.ndarray | None:
    available = keys(npz)
    for key in candidates:
        if key in available:
            try:
                arr = np.asarray(npz[key], dtype=float)
            except Exception:
                continue
            if arr.size > 0:
                return arr
    return None


def get_metadata(npz: object) -> dict:
    for key in ("metadata_json", "metadata"):
        if key not in keys(npz):
            continue
        try:
            value = npz[key]
            if key == "metadata_json":
                text = str(value.item() if np.asarray(value).shape == () else value[0])
                return json.loads(text)
            obj = value[0]
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "item"):
                item = obj.item()
                return item if isinstance(item, dict) else {}
        except Exception:
            return {}
    return {}


def as_2d(arr: np.ndarray | None, width: int | None = None) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if width is not None:
        if a.shape[1] < width:
            pad = np.full((a.shape[0], width - a.shape[1]), np.nan)
            a = np.concatenate([a, pad], axis=1)
        a = a[:, :width]
    keep = np.all(np.isfinite(a), axis=1)
    if not np.any(keep):
        return None
    return a[keep]


def time_axis(npz: object, n: int) -> np.ndarray:
    for key in ("t_rel", "timestamps", "t_mono", "host_t"):
        arr = get_array(npz, [key])
        if arr is not None and arr.size >= n:
            t = np.asarray(arr[:n], dtype=float).reshape(-1)
            return t - t[0]
    sensor_ts = get_array(npz, ["sensor_timestamp_us"])
    if sensor_ts is not None and sensor_ts.size >= n:
        t = np.asarray(sensor_ts[:n], dtype=float).reshape(-1) * 1e-6
        return t - t[0]
    meta = get_metadata(npz)
    hz = float(meta.get("control_loop_hz", 0.0) or meta.get("expected_hz", 0.0) or 0.0)
    if hz <= 0.0:
        hz = 30.0
    return np.arange(n, dtype=float) / hz


def norm_rows(arr: np.ndarray | None) -> np.ndarray | None:
    a = as_2d(arr)
    if a is None:
        return None
    return np.linalg.norm(a, axis=1)


def robust_noise(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(max(sigma, np.std(x[: max(5, x.size // 5)]), 1e-12))


def snr_db(signal_level: float, noise_level: float) -> float:
    signal = max(float(signal_level), 1e-12)
    noise = max(float(noise_level), 1e-12)
    return 20.0 * math.log10(signal / noise)


def rms_axis(arr: np.ndarray) -> np.ndarray:
    a = as_2d(arr)
    if a is None:
        return np.array([])
    return np.sqrt(np.nanmean(a * a, axis=0))


def p95_abs_axis(arr: np.ndarray) -> np.ndarray:
    a = as_2d(arr)
    if a is None:
        return np.array([])
    return np.nanpercentile(np.abs(a), 95, axis=0)


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    target_dir = out_dir / stem if GROUP_BY_PLOT else out_dir
    written = []
    if not NO_SAVE:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = target_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
        print("[plot]", ", ".join(str(p) for p in written))
    if SHOW_FIGURES:
        try:
            fig.canvas.manager.set_window_title(stem)
        except Exception:
            pass
        plt.show(block=True)
    plt.close(fig)
    if NO_SAVE and not SHOW_FIGURES:
        print(f"[plot] {stem} built but not saved (--no-save)")
    return written


def short_run_id(path: Path) -> str:
    match = re.search(r"_(\d+)_ep(\d+)$", path.stem)
    if match:
        return f"{match.group(1)}_ep{match.group(2)}"
    return path.stem[-28:]


def save_figs_together(
    figures: Sequence[tuple[plt.Figure, str]],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    written = []
    if not NO_SAVE:
        for fig, stem in figures:
            target_dir = out_dir / stem if GROUP_BY_PLOT else out_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                path = target_dir / f"{stem}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                written.append(path)
        print("[plot]", ", ".join(str(p) for p in written))
    if SHOW_FIGURES:
        for fig, stem in figures:
            try:
                fig.canvas.manager.set_window_title(stem)
            except Exception:
                pass
        plt.show(block=True)
    for fig, stem in figures:
        plt.close(fig)
        if NO_SAVE and not SHOW_FIGURES:
            print(f"[plot] {stem} built but not saved (--no-save)")
    return written


def limit_files(files: list[Path], max_files: int) -> list[Path]:
    if max_files and max_files > 0:
        return files[: int(max_files)]
    return files


def first_usable(paths: Sequence[Path], required: Sequence[str]) -> LoadedNpz | None:
    for path in paths:
        loaded = load_npz(path)
        if loaded is None:
            continue
        if all(get_array(loaded.data, [key]) is not None for key in required):
            return loaded
    return None


def expand_input_npz(paths: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else (Path.cwd() / raw_path)
        path = path.resolve()
        if path.is_dir():
            expanded.extend(sorted(p for p in path.rglob("*.npz") if p.is_file()))
        elif path.is_file():
            expanded.append(path)
        else:
            print(f"[skip] input path not found: {path}", file=sys.stderr)
    return list(dict.fromkeys(expanded))


def display_joint_label(path: Path, joint_index: int) -> str:
    for part in [path.parent.name, path.stem]:
        match = re.search(r"(?:^|_)sine_j(\d+)(?:_|$)", part.lower())
        if match:
            return f"J{int(match.group(1))}"
    return f"J{joint_index + 1}"


def first_rising_midline_crossing(t: np.ndarray, y: np.ndarray) -> float | None:
    if t.size < 3 or y.size < 3:
        return None
    center = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
    rel = np.asarray(y, dtype=float) - center
    idx = np.flatnonzero((rel[:-1] < 0.0) & (rel[1:] >= 0.0))
    if idx.size == 0:
        return None
    i = int(idx[0])
    denom = rel[i + 1] - rel[i]
    if abs(float(denom)) < 1e-12:
        return float(t[i])
    alpha = float(np.clip(-rel[i] / denom, 0.0, 1.0))
    return float(t[i] + alpha * (t[i + 1] - t[i]))


def final_phase_b_files(data_root: Path, max_files: int = 0) -> list[Path]:
    files = []
    files += find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold"])
    files += find_npz(data_root / "cr_dagger_npz", include=["phaseb_cable_routing_w_intervention"])
    files += find_npz(data_root / "cr_dagger_npz", include=["phaseb_cable_routing_wo_intervention"])
    # Preserve order while removing duplicates.
    unique = list(dict.fromkeys(files))
    return limit_files(unique, max_files)


def plot_impedance_baseline(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    paths = find_npz(data_root / "epsilon_measurements", include=["e1_impedance", "static_hold"])
    if not paths:
        paths = find_npz(data_root / "epsilon_measurements", include=["2026-05-18_baseline_impedance", "naive_residual", "static_hold"])
    if not paths:
        paths = find_npz(data_root / "epsilon_measurements", include=["2026-05-18_baseline_impedance", "static_hold"])
    loaded = None
    for path in paths:
        candidate = load_npz(path)
        if candidate is None:
            continue
        tau_candidate = norm_rows(get_array(candidate.data, ["tau_residual", "tau_ext_shi"]))
        if tau_candidate is not None and float(np.nanpercentile(tau_candidate, 95)) > 1e-6:
            loaded = candidate
            break
    if loaded is None:
        loaded = first_usable(paths, ["epsilon"])
    if loaded is None:
        print("[skip] plot A: no impedance static-hold residual file found")
        return

    npz = loaded.data
    tau_res = as_2d(get_array(npz, ["tau_residual", "tau_ext_shi"]), 6)
    tau_cmd = as_2d(get_array(npz, ["tau_cmd"]), 6)
    tau_ext = as_2d(get_array(npz, ["tau_ext_shi", "tau_residual"]), 6)
    eps = as_2d(get_array(npz, ["epsilon", "epsilon_leader"]), 6)
    if tau_res is None and eps is None:
        print(f"[skip] plot A: missing torque residual and epsilon in {loaded.path}")
        return
    n = tau_res.shape[0] if tau_res is not None else eps.shape[0]
    t = time_axis(npz, n)
    tau_norm = np.linalg.norm(tau_res[:, :6], axis=1) if tau_res is not None else np.full(n, np.nan)
    eps_norm = np.linalg.norm(eps[:n], axis=1) if eps is not None else np.full(n, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.2), sharex="col")
    for joint in range(2):
        ax = axes[joint, 0]
        if tau_cmd is not None:
            ax.plot(t, tau_cmd[:n, joint], label=f"tau_cmd J{joint + 1}", linewidth=0.9)
        if tau_ext is not None:
            ax.plot(t, tau_ext[:n, joint], label=f"tau_ext J{joint + 1}", linewidth=0.9)
        ax.set_ylabel("[Nm]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0, 0].set_title("Current-mode impedance static hold")
    axes[1, 0].set_xlabel("time [s]")
    axes[0, 1].hist(tau_norm, bins=60, color="tab:orange", alpha=0.85)
    axes[0, 1].axvline(np.nanpercentile(tau_norm, 95), color="black", linestyle="--", linewidth=0.8, label="p95")
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].set_xlabel("||tau_residual|| [Nm]")
    axes[0, 1].set_ylabel("samples")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].legend()
    axes[1, 1].plot(t, eps_norm, color="tab:blue", linewidth=0.9)
    axes[1, 1].set_title("Tracking error")
    axes[1, 1].set_ylabel("||epsilon|| [rad]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle("Impedance baseline: stiction-sized residuals")
    fig.tight_layout()
    save_fig(fig, out_dir, "A_impedance_baseline_static_hold", formats, dpi)


def plot_impedance_sine_tracking_j1(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    aggregate_inputs: bool = False,
    plot_title: str = "",
) -> None:
    if input_npz:
        loaded_items = []
        for path in expand_input_npz(input_npz):
            loaded = load_npz(path)
            if loaded is not None:
                loaded_items.append(loaded)
    else:
        paths = find_npz(
            data_root / "epsilon_measurements",
            include=["2026-05-18_baseline_impedance", "sine_j1"],
        )
        if not paths:
            paths = find_npz(data_root / "epsilon_measurements", include=["baseline_impedance", "sine_j1"])
        if not paths:
            paths = find_npz(data_root / "epsilon_measurements", include=["sine_j1"])

        loaded = select_longest(paths, ["q_ref", "q_leader"])
        if loaded is None:
            loaded = select_longest(paths, ["q_cmd_leader", "q_leader"])
        loaded_items = [loaded] if loaded is not None else []

    if not loaded_items:
        print("[skip] impedance sine tracking: no usable impedance file found")
        return

    if aggregate_inputs and len(loaded_items) > 1:
        plot_impedance_sine_tracking_aggregate(loaded_items, out_dir, formats, dpi, plot_title)
        return

    multi_file = len(loaded_items) > 1
    for loaded in loaded_items:
        _plot_impedance_sine_tracking_loaded(
            loaded,
            out_dir,
            formats,
            dpi,
            stem=(
                f"A_impedance_j1_sine_tracking_{loaded.path.stem}"
                if multi_file
                else "A_impedance_j1_sine_tracking"
            ),
            plot_title=plot_title,
        )


def _impedance_tracking_series(
    loaded: LoadedNpz,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict] | None:
    npz = loaded.data
    q_ref = as_2d(get_array(npz, ["q_ref", "q_cmd_leader", "q_c_leader"]), 6)
    q_actual = as_2d(get_array(npz, ["q_leader", "q_actual", "q_follower"]), 6)
    if q_ref is None or q_actual is None:
        print(f"[skip] impedance sine tracking: missing q_ref/q_actual in {loaded.path}")
        return None

    n = min(q_ref.shape[0], q_actual.shape[0])
    meta = get_metadata(npz)
    joint = int(meta.get("joint_index", 0))
    if joint < 0 or joint >= min(q_ref.shape[1], q_actual.shape[1]):
        joint = 0
    t = time_axis(npz, n)
    return t, q_ref[:n, joint], q_actual[:n, joint], joint, meta


def plot_impedance_sine_tracking_aggregate(
    loaded_items: Sequence[LoadedNpz],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    plot_title: str = "",
) -> None:
    rows = []
    for loaded in loaded_items:
        series = _impedance_tracking_series(loaded)
        if series is not None:
            rows.append((loaded, *series))
    if len(rows) < 2:
        print("[skip] aggregate impedance tracking: fewer than two usable files")
        return

    dt_candidates = []
    phase_starts = []
    phase_ends = []
    for _loaded, t, *_rest in rows:
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            dt_candidates.append(float(np.nanmedian(diffs)))
    for _loaded, t, q_ref_j, *_rest in rows:
        crossing = first_rising_midline_crossing(t, q_ref_j)
        if crossing is None:
            crossing = 0.0
        phase_starts.append(float(crossing))
        phase_ends.append(float(t[-1] - crossing))
    dt = float(np.nanmedian(dt_candidates)) if dt_candidates else 1.0 / 30.0
    t_end = min(phase_ends)
    n_common = max(2, int(np.floor(t_end / dt)) + 1)
    t_common = np.linspace(0.0, dt * (n_common - 1), n_common)

    ref_interp = []
    actual_interp = []
    source_paths = []
    kp_values = []
    kd_values = []
    joints = []
    for (loaded, t, q_ref_j, q_actual_j, joint, meta), phase_start in zip(rows, phase_starts):
        t_sample = t_common + phase_start
        if t.size < 2 or t[-1] < t_sample[-1]:
            continue
        ref_interp.append(np.interp(t_sample, t, q_ref_j))
        actual_interp.append(np.interp(t_sample, t, q_actual_j))
        source_paths.append(loaded.path)
        kp_values.append(meta.get("kp", "n/a"))
        kd_values.append(meta.get("kd", "n/a"))
        joints.append(joint)

    if len(actual_interp) < 2:
        print("[skip] aggregate impedance tracking: fewer than two files cover common time axis")
        return

    ref_arr = np.vstack(ref_interp)
    actual_arr = np.vstack(actual_interp)
    ref_mean = np.nanmean(ref_arr, axis=0)
    actual_mean = np.nanmean(actual_arr, axis=0)
    actual_std = np.nanstd(actual_arr, axis=0)
    eps_arr = actual_arr - ref_arr
    rms = float(np.sqrt(np.nanmean(eps_arr * eps_arr)))
    p95 = float(np.nanpercentile(np.abs(eps_arr), 95))
    joint = int(round(float(np.nanmedian(joints)))) if joints else 0
    joint_label = display_joint_label(source_paths[0], joint) if source_paths else f"J{joint + 1}"

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(
        t_common,
        ref_mean,
        label="Reference position (mean)",
        color="#1f3a5f",
        linewidth=1.3,
        linestyle="--",
    )
    ax.plot(
        t_common,
        actual_mean,
        label="Measured position (mean)",
        color="#d55e00",
        linewidth=1.15,
    )
    ax.fill_between(
        t_common,
        actual_mean - actual_std,
        actual_mean + actual_std,
        color="#d55e00",
        alpha=0.18,
        linewidth=0.0,
        label="Measured position +/- 1 std",
    )
    title = plot_title or f"Current-mode impedance tracking: {joint_label} sine"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.04,
        f"n={len(actual_interp)} runs\nRMS error: {rms:.4f} rad\np95 error: {p95:.4f} rad",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
    )
    fig.tight_layout()

    print("[sources]")
    for path in source_paths:
        print(f"  {path}")
    print(f"[impedance] kp={kp_values[0] if kp_values else 'n/a'} kd={kd_values[0] if kd_values else 'n/a'}")
    save_fig(fig, out_dir, "A_impedance_j1_sine_tracking_aggregate", formats, dpi)


def _aligned_impedance_arrays(
    loaded_items: Sequence[LoadedNpz],
) -> dict | None:
    rows = []
    for loaded in loaded_items:
        series = _impedance_tracking_series(loaded)
        if series is not None:
            rows.append((loaded, *series))
    if len(rows) < 1:
        return None

    dt_candidates = []
    phase_starts = []
    phase_ends = []
    for _loaded, t, *_rest in rows:
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            dt_candidates.append(float(np.nanmedian(diffs)))
    for _loaded, t, q_ref_j, *_rest in rows:
        crossing = first_rising_midline_crossing(t, q_ref_j)
        if crossing is None:
            crossing = 0.0
        phase_starts.append(float(crossing))
        phase_ends.append(float(t[-1] - crossing))

    dt = float(np.nanmedian(dt_candidates)) if dt_candidates else 1.0 / 30.0
    t_end = min(phase_ends)
    n_common = max(2, int(np.floor(t_end / dt)) + 1)
    t_common = np.linspace(0.0, dt * (n_common - 1), n_common)

    ref_interp = []
    actual_interp = []
    source_paths = []
    kp_values = []
    kd_values = []
    joints = []
    for (loaded, t, q_ref_j, q_actual_j, joint, meta), phase_start in zip(rows, phase_starts):
        t_sample = t_common + phase_start
        if t.size < 2 or t[-1] < t_sample[-1]:
            continue
        ref_interp.append(np.interp(t_sample, t, q_ref_j))
        actual_interp.append(np.interp(t_sample, t, q_actual_j))
        source_paths.append(loaded.path)
        kp_values.append(meta.get("kp", "n/a"))
        kd_values.append(meta.get("kd", "n/a"))
        joints.append(joint)

    if len(actual_interp) < 1:
        return None

    ref_arr = np.vstack(ref_interp)
    actual_arr = np.vstack(actual_interp)
    joint = int(round(float(np.nanmedian(joints)))) if joints else 0
    joint_label = display_joint_label(source_paths[0], joint) if source_paths else f"J{joint + 1}"
    eps_arr = actual_arr - ref_arr
    return {
        "t": t_common,
        "ref": ref_arr,
        "actual": actual_arr,
        "joint_label": joint_label,
        "source_paths": source_paths,
        "kp": kp_values[0] if kp_values else "n/a",
        "kd": kd_values[0] if kd_values else "n/a",
        "rms": float(np.sqrt(np.nanmean(eps_arr * eps_arr))),
        "p95": float(np.nanpercentile(np.abs(eps_arr), 95)),
    }


def plot_impedance_appendix(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    plot_title: str = "",
) -> None:
    groups: dict[str, list[Path]] = {}
    if input_npz:
        for path in expand_input_npz(input_npz):
            label = display_joint_label(path, 0)
            groups.setdefault(label, []).append(path)
    else:
        base = data_root / "epsilon_measurements" / "2026-05-18_baseline_impedance"
        for label, folder in [("J1", "sine_j1"), ("J2", "sine_j2"), ("J3", "sine_j3")]:
            groups[label] = find_npz(base / folder)

    ordered_labels = [label for label in ["J1", "J2", "J3"] if groups.get(label)]
    if not ordered_labels:
        ordered_labels = sorted(groups.keys())[:3]
    if not ordered_labels:
        print("[skip] impedance appendix: no input files found")
        return

    panel_data = []
    for label in ordered_labels[:3]:
        loaded_items = [loaded for p in groups[label] if (loaded := load_npz(p)) is not None]
        aligned = _aligned_impedance_arrays(loaded_items)
        if aligned is None:
            print(f"[skip] impedance appendix: no usable files for {label}")
            continue
        aligned["joint_label"] = label
        panel_data.append(aligned)

    if not panel_data:
        print("[skip] impedance appendix: no usable panels")
        return

    fig, axes = plt.subplots(len(panel_data), 1, figsize=(7.0, 2.45 * len(panel_data)), sharex=True)
    if len(panel_data) == 1:
        axes = [axes]

    for ax, data in zip(axes, panel_data):
        t = data["t"]
        ref_arr = data["ref"]
        actual_arr = data["actual"]
        ref_mean = np.nanmean(ref_arr, axis=0)
        actual_mean = np.nanmean(actual_arr, axis=0)
        actual_std = np.nanstd(actual_arr, axis=0)

        for actual in actual_arr:
            ax.plot(t, actual, color="#d55e00", alpha=0.18, linewidth=0.65)
        ax.plot(t, ref_mean, color="#1f3a5f", linewidth=1.2, linestyle="--", label="Reference mean")
        ax.fill_between(
            t,
            actual_mean - actual_std,
            actual_mean + actual_std,
            color="#d55e00",
            alpha=0.16,
            linewidth=0.0,
            label="Measured +/- 1 std",
        )
        ax.plot(t, actual_mean, color="#d55e00", linewidth=1.15, label="Measured mean")
        ax.set_ylabel(f"{data['joint_label']}\nposition [rad]")
        ax.grid(alpha=0.3)
        ax.text(
            0.015,
            0.05,
            f"n={actual_arr.shape[0]}, RMS={data['rms']:.3f} rad, p95={data['p95']:.3f} rad",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )

    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time [s]")
    title = plot_title or "Current-mode impedance tracking across joints"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    print("[appendix sources]")
    for data in panel_data:
        print(f"  {data['joint_label']}: {len(data['source_paths'])} files")
        for path in data["source_paths"]:
            print(f"    {path}")
    print(f"[impedance] kp={panel_data[0]['kp']} kd={panel_data[0]['kd']}")
    save_fig(fig, out_dir, "A_impedance_j1_j2_j3_appendix", formats, dpi)


def _plot_impedance_sine_tracking_loaded(
    loaded: LoadedNpz,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    stem: str,
    plot_title: str = "",
) -> None:
    series = _impedance_tracking_series(loaded)
    if series is None:
        return
    t, q_ref_j, q_actual_j, joint, meta = series
    kp = meta.get("kp", "n/a")
    kd = meta.get("kd", "n/a")
    joint_label = display_joint_label(loaded.path, joint)
    eps_j = q_actual_j - q_ref_j
    rms = float(np.sqrt(np.nanmean(eps_j * eps_j)))
    p95 = float(np.nanpercentile(np.abs(eps_j), 95))

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(
        t,
        q_ref_j,
        label="Reference position",
        color="#1f3a5f",
        linewidth=1.3,
        linestyle="--",
    )
    ax.plot(
        t,
        q_actual_j,
        label="Measured position",
        color="#d55e00",
        linewidth=1.15,
    )
    title = plot_title or f"Current-mode impedance tracking: {joint_label} sine"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.04,
        f"RMS error: {rms:.4f} rad\np95 error: {p95:.4f} rad",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
    )
    fig.tight_layout()
    print(f"[source] {loaded.path}")
    print(f"[impedance] kp={kp} kd={kd}")
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_observer_vs_bota_qualitative(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    observer_static = first_usable(
        find_npz(data_root / "epsilon_measurements", include=["e3_observer_yamane", "static_hold"]),
        ["tau_ext_shi"],
    )
    observer_active = first_usable(
        find_npz(data_root / "epsilon_measurements", include=["e3_observer_yamane", "sine"]),
        ["tau_ext_shi"],
    )
    bota_static = first_usable(
        find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_no_interventions"]),
        ["wrench_base"],
    )
    bota_active = first_usable(
        find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"]),
        ["wrench_base"],
    )
    if None in (observer_static, observer_active, bota_static, bota_active):
        print("[skip] plot B: missing observer/BOTA qualitative files")
        return

    series = []
    for label, loaded, names, cols, color in [
        ("Observer no-contact", observer_static, ["tau_ext_shi", "tau_residual"], 6, "tab:blue"),
        ("Observer active", observer_active, ["tau_ext_shi", "tau_residual"], 6, "tab:cyan"),
        ("BOTA no-contact", bota_static, ["wrench_base", "wrench"], 3, "tab:orange"),
        ("BOTA intervention", bota_active, ["wrench_base", "wrench"], 3, "tab:red"),
    ]:
        sig = as_2d(get_array(loaded.data, names), max(cols, 6))
        if sig is None:
            continue
        y = np.linalg.norm(sig[:, :cols], axis=1)
        t = time_axis(loaded.data, y.size)
        keep = t <= min(20.0, t[-1])
        series.append((label, t[keep], y[keep], color, "Nm" if cols == 6 else "N"))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=False)
    for label, t, y, color, unit in series[:2]:
        axes[0].plot(t, y, label=f"{label} [{unit}]", color=color, linewidth=0.9)
    for label, t, y, color, unit in series[2:]:
        axes[1].plot(t, y, label=f"{label} [{unit}]", color=color, linewidth=0.9)
    axes[0].set_title("Sensorless observer residual")
    axes[1].set_title("Calibrated BOTA wrench")
    for ax in axes:
        ax.set_ylabel("p-norm signal")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle("Qualitative observer/BOTA separation")
    fig.tight_layout()
    save_fig(fig, out_dir, "B_observer_vs_bota_qualitative", formats, dpi)


def _sensorless_static_rows(
    data_root: Path,
) -> list[tuple[str, str, Path, np.ndarray, np.ndarray, float, str]]:
    specs = [
        (
            "Current-based residual",
            "Shi / current",
            data_root
            / "epsilon_measurements"
            / "2026-05-18_baseline_impedance"
            / "naive_residual",
            ["tau_residual", "tau_ext_shi"],
            "#1f77b4",
        ),
        (
            "Momentum observer",
            "Yamane",
            data_root / "epsilon_measurements" / "E3_observer_yamane",
            ["tau_ext_shi", "tau_residual"],
            "#d55e00",
        ),
    ]
    rows = []
    for label, short_label, folder, candidates, color in specs:
        for path in find_npz(folder, include=["static_hold"]):
            loaded = load_npz(path)
            if loaded is None:
                continue
            tau = as_2d(get_array(loaded.data, candidates), 6)
            if tau is None:
                continue
            t = time_axis(loaded.data, tau.shape[0])
            residual = np.nanmax(np.abs(tau[:, :6]), axis=1)
            residual = residual[np.isfinite(residual)]
            if residual.size < 2:
                continue
            t = t[: residual.size]
            p95 = float(np.nanpercentile(residual, 95))
            rows.append((label, short_label, path, t, residual, p95, color))
    return rows


def plot_sensorless_observer_noise(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    plot_title: str = "",
) -> None:
    rows = _sensorless_static_rows(data_root)
    if not rows:
        print("[skip] observer noise: no static-hold sensorless residual files found")
        return

    by_label: dict[str, list[tuple[str, str, Path, np.ndarray, np.ndarray, float, str]]] = {}
    for row in rows:
        by_label.setdefault(row[0], []).append(row)
    required = ["Current-based residual", "Momentum observer"]
    if not all(label in by_label for label in required):
        print("[skip] observer noise: missing Shi/current or Yamane static-hold group")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), gridspec_kw={"width_ratios": [1.65, 1.0]})
    ax_ts, ax_box = axes

    for label in required:
        group = by_label[label]
        representative = sorted(group, key=lambda row: row[5])[len(group) // 2]
        _full_label, short_label, _path, t, residual, p95, color = representative
        keep = t <= min(20.0, float(t[-1]))
        ax_ts.plot(
            t[keep],
            residual[keep],
            label=f"{short_label} (p95={p95:.2f} Nm)",
            color=color,
            linewidth=1.1,
        )

    ax_ts.axhspan(0.5, 1.0, color="0.6", alpha=0.15, linewidth=0.0, label="small interaction scale")
    ax_ts.set_title("No-contact static hold")
    ax_ts.set_xlabel("Time [s]")
    ax_ts.set_ylabel("max |residual torque| [Nm]")
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(loc="upper right")

    box_values = [np.asarray([row[5] for row in by_label[label]], dtype=float) for label in required]
    positions = np.arange(1, len(required) + 1)
    bp = ax_box.boxplot(
        box_values,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        tick_labels=["Shi /\ncurrent", "Yamane"],
    )
    colors = ["#1f77b4", "#d55e00"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.1)
    for pos, values, color in zip(positions, box_values, colors):
        ax_box.scatter(
            np.full(values.shape, pos, dtype=float),
            values,
            color=color,
            s=20,
            zorder=3,
            alpha=0.9,
        )
        ax_box.text(
            pos,
            float(np.nanmax(values)) + 0.025,
            f"median\n{np.nanmedian(values):.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax_box.axhspan(0.5, 1.0, color="0.6", alpha=0.15, linewidth=0.0)
    ax_box.set_title("p95 across runs")
    ax_box.set_ylabel("p95 max |residual torque| [Nm]")
    ax_box.grid(axis="y", alpha=0.3)

    title = plot_title or "Sensorless Observer Noise During Static Hold"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    print("[observer noise sources]")
    for label in required:
        print(f"  {label}:")
        for _full_label, _short_label, path, _t, _residual, p95, _color in by_label[label]:
            print(f"    p95={p95:.3f} Nm  {path}")
    save_fig(fig, out_dir, "B_sensorless_observer_static_hold_noise", formats, dpi)


def _observer_debug_paths(data_root: Path, input_npz: Sequence[Path]) -> list[Path]:
    if input_npz:
        return expand_input_npz(input_npz)
    paths = []
    paths += find_npz(
        data_root
        / "epsilon_measurements"
        / "2026-05-18_baseline_impedance"
        / "naive_residual",
        include=["static_hold"],
    )
    paths += find_npz(data_root / "epsilon_measurements" / "E3_observer_yamane", include=["static_hold"])
    return list(dict.fromkeys(paths))


def _observer_signal(npz: object) -> tuple[np.ndarray | None, str]:
    for key in ("tau_residual", "tau_ext_shi", "tau_ext"):
        arr = as_2d(get_array(npz, [key]), 6)
        if arr is not None:
            return arr, key
    return None, "n/a"


def plot_observer_debug(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
) -> None:
    paths = _observer_debug_paths(data_root, input_npz)
    if not paths:
        print("[skip] observer debug: no input files found")
        return

    for path in paths:
        loaded = load_npz(path)
        if loaded is None:
            continue
        npz = loaded.data
        tau, tau_key = _observer_signal(npz)
        q = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
        if tau is None or q is None:
            print(f"[skip] observer debug: missing residual or q_leader in {path}")
            continue

        n = min(tau.shape[0], q.shape[0])
        t = time_axis(npz, n)
        tau = tau[:n, :6]
        q = q[:n, :6]
        residual_max = np.nanmax(np.abs(tau), axis=1)
        residual_norm = np.linalg.norm(tau, axis=1)
        p95 = float(np.nanpercentile(residual_max, 95))
        q_ptp = np.ptp(q, axis=0)
        meta = get_metadata(npz)
        contact_state = get_array(npz, ["contact_state"])
        contact_summary = ""
        if contact_state is not None and contact_state.size:
            vals, counts = np.unique(np.asarray(contact_state[:n], dtype=int), return_counts=True)
            contact_summary = " contact_state=" + str(dict(zip(vals.tolist(), counts.tolist())))

        print(f"[source] {path}")
        print(
            "[debug] "
            f"mode={meta.get('mode', 'n/a')} test={meta.get('test_mode', 'n/a')} "
            f"observer={meta.get('admittance_observer', 'n/a')} signal={tau_key} "
            f"p95 max|tau|={p95:.3f} Nm q_ptp={np.array2string(q_ptp, precision=4)}"
            f"{contact_summary}"
        )

        stem_base = f"observer_debug_{path.stem}"
        fig_res, ax = plt.subplots(figsize=(8.2, 3.4))
        ax.plot(t, residual_max, color="#d55e00", linewidth=1.05, label="max joint residual")
        ax.plot(t, residual_norm, color="#1f3a5f", linewidth=0.9, alpha=0.8, label="residual norm")
        ax.axhline(p95, color="black", linestyle="--", linewidth=0.8, label=f"p95 max = {p95:.3f} Nm")
        ax.set_title(f"Observer residual debug: {path.name}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Residual torque [Nm]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig_res.tight_layout()

        fig_q, ax = plt.subplots(figsize=(8.2, 3.8))
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, 6))
        for j in range(6):
            ax.plot(t, q[:, j], label=JOINT_LABELS[j], color=colors[j], linewidth=0.95)
        ax.set_title(f"Leader joint positions: {path.name}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Joint position [rad]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3)
        fig_q.tight_layout()
        save_figs_together(
            [
                (fig_res, f"{stem_base}_residual"),
                (fig_q, f"{stem_base}_leader_positions"),
            ],
            out_dir,
            formats,
            dpi,
        )


def _default_observer_admittance_cases(data_root: Path) -> list[tuple[str, Path]]:
    return [
        (
            "Shi/current input",
            data_root
            / "epsilon_measurements"
            / "2026-05-19_admittance_position_shi"
            / "epsilon_leader_admittance_position_static_hold_1058589.npz",
        ),
        (
            "Yamane input",
            data_root
            / "epsilon_measurements"
            / "E4_admittance_impedance"
            / "epsilon_leader_admittance_impedance_static_hold_700631.npz",
        ),
    ]


def _admittance_case_series(path: Path, label: str) -> dict | None:
    loaded = load_npz(path)
    if loaded is None:
        return None
    npz = loaded.data
    tau, tau_key = _observer_signal(npz)
    q_ref = as_2d(get_array(npz, ["q_ref"]), 6)
    q_leader = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
    q_c = as_2d(get_array(npz, ["q_c_leader", "q_cmd_leader"]), 6)
    delta = as_2d(get_array(npz, ["delta_corr", "delta_corr_raw"]), 6)
    if tau is None or q_ref is None or q_leader is None:
        print(f"[skip] observer admittance: missing tau/q streams in {path}")
        return None

    n = min(tau.shape[0], q_ref.shape[0], q_leader.shape[0])
    if q_c is not None:
        n = min(n, q_c.shape[0])
    if delta is not None:
        n = min(n, delta.shape[0])
    t = time_axis(npz, n)
    tau = tau[:n, :6]
    q_ref = q_ref[:n, :6]
    q_leader = q_leader[:n, :6]
    if delta is not None:
        delta_used = delta[:n, :6]
    elif q_c is not None:
        delta_used = q_c[:n, :6] - q_ref
    else:
        delta_used = np.full_like(q_ref, np.nan)

    contact_probability = get_array(npz, ["contact_probability"])
    if contact_probability is not None and contact_probability.size >= n:
        contact_probability = np.asarray(contact_probability[:n], dtype=float).reshape(-1)
    else:
        contact_probability = np.full(n, np.nan)

    residual = np.nanmax(np.abs(tau), axis=1)
    delta_norm = np.linalg.norm(delta_used, axis=1)
    leader_dev = np.nanmax(np.abs(q_leader - q_leader[0]), axis=1)
    meta = get_metadata(npz)
    return {
        "label": label,
        "path": path,
        "t": t,
        "residual": residual,
        "delta_norm": delta_norm,
        "leader_dev": leader_dev,
        "contact_probability": contact_probability,
        "tau_key": tau_key,
        "meta": meta,
        "p95_residual": float(np.nanpercentile(residual, 95)),
        "p95_delta": float(np.nanpercentile(delta_norm, 95)),
        "max_delta": float(np.nanmax(delta_norm)),
        "max_leader_dev": float(np.nanmax(leader_dev)),
    }


def plot_observer_admittance_static_comparison(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    plot_title: str = "",
) -> None:
    if input_npz:
        paths = expand_input_npz(input_npz)
        labels = [f"Case {idx + 1}" for idx in range(len(paths))]
        if len(paths) >= 2:
            labels[0] = "Shi/current input"
            labels[1] = "Yamane input"
        cases = list(zip(labels, paths))[:2]
    else:
        cases = _default_observer_admittance_cases(data_root)

    series = []
    for label, path in cases:
        item = _admittance_case_series(path, label)
        if item is not None:
            series.append(item)
    if len(series) < 2:
        print("[skip] observer admittance: need two usable cases")
        return

    fig, axes = plt.subplots(3, 2, figsize=(9.4, 6.5), sharex="col")
    colors = ["#1f77b4", "#d55e00"]
    row_specs = [
        ("residual", "max |observer residual| [Nm]"),
        ("delta_norm", "||admittance offset|| [rad]"),
        ("leader_dev", "max |leader deviation| [rad]"),
    ]
    for col, (item, color) in enumerate(zip(series, colors)):
        t = item["t"]
        t_plot = t - t[0]
        keep = t_plot <= min(20.0, float(t_plot[-1]))
        meta = item["meta"]
        subtitle = (
            f"{item['label']}\n"
            f"M={meta.get('adm_mass', 'n/a')}, D={meta.get('adm_damp', 'n/a')}, "
            f"K={meta.get('adm_stiff', 'n/a')}, limit={meta.get('adm_delta_max', 'n/a')}"
        )
        axes[0, col].set_title(subtitle, fontsize=10.5, fontweight="bold")
        for row, (key, ylabel) in enumerate(row_specs):
            ax = axes[row, col]
            ax.plot(t_plot[keep], item[key][keep], color=color, linewidth=1.1)
            if key == "residual":
                ax.axhline(
                    item["p95_residual"],
                    color="black",
                    linestyle="--",
                    linewidth=0.75,
                    label=f"p95={item['p95_residual']:.2f}",
                )
                ax.legend(loc="upper right")
            if key == "delta_norm":
                ax.axhline(
                    item["p95_delta"],
                    color="black",
                    linestyle="--",
                    linewidth=0.75,
                    label=f"p95={item['p95_delta']:.3f}",
                )
                ax.legend(loc="upper right")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes[-1, col].set_xlabel("Time [s]")

        text = (
            f"mode={meta.get('mode', 'n/a')}\n"
            f"observer={meta.get('admittance_observer', 'n/a')}\n"
            f"signal={item['tau_key']}"
        )
        axes[2, col].text(
            0.02,
            0.06,
            text,
            transform=axes[2, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
        )

    for row, (key, _ylabel) in enumerate(row_specs):
        row_max = 0.0
        for item in series:
            t_plot = item["t"] - item["t"][0]
            keep = t_plot <= min(20.0, float(t_plot[-1]))
            row_max = max(row_max, float(np.nanmax(item[key][keep])))
            if key == "residual":
                row_max = max(row_max, item["p95_residual"])
            if key == "delta_norm":
                row_max = max(row_max, item["p95_delta"])
        if row_max > 0.0 and np.isfinite(row_max):
            for col in range(2):
                axes[row, col].set_ylim(-0.02 * row_max, 1.08 * row_max)

    title = plot_title or "Observer-Driven Admittance During Commanded Static Hold"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    print("[observer admittance sources]")
    for item in series:
        print(
            f"  {item['label']}: p95_residual={item['p95_residual']:.3f} Nm "
            f"p95_delta={item['p95_delta']:.4f} rad max_leader_dev={item['max_leader_dev']:.4f} rad"
        )
        print(f"    {item['path']}")
    save_fig(fig, out_dir, "B_observer_driven_admittance_static_hold", formats, dpi)


def plot_sensorless_observer_attempts(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    plot_title: str = "",
) -> None:
    if input_npz:
        paths = expand_input_npz(input_npz)
        if len(paths) < 2:
            print("[skip] observer attempts: pass two NPZ files or use defaults")
            return
        shi_path, yamane_path = paths[:2]
    else:
        shi_path = (
            data_root
            / "epsilon_measurements"
            / "smooth_friction_error_j1"
            / "epsilon_leader_tracking_sine_540492.npz"
        )
        yamane_path = (
            data_root
            / "epsilon_measurements"
            / "E3_observer_yamane"
            / "epsilon_leader_observer_sine_699453.npz"
        )

    cases = [
        {
            "label": "Current-based Shi et al. observer output during sine motion",
            "path": shi_path,
            "response": "j2_position",
            "response_label": "J2 position [rad]",
            "residual_key_order": ["tau_residual", "tau_ext_shi", "tau_ext"],
            "color": "#1f77b4",
        },
        {
            "label": "Momentum-based Yamane et al. observer output during sine motion",
            "path": yamane_path,
            "response": "j2_position",
            "response_label": "J2 position [rad]",
            "residual_key_order": ["tau_ext_shi", "tau_residual", "tau_ext"],
            "color": "#d55e00",
        },
    ]

    prepared = []
    for case in cases:
        loaded = load_npz(case["path"])
        if loaded is None:
            return
        npz = loaded.data
        tau = None
        tau_key = "n/a"
        for key in case["residual_key_order"]:
            tau = as_2d(get_array(npz, [key]), 6)
            if tau is not None:
                tau_key = key
                break
        if tau is None:
            print(f"[skip] observer attempts: missing residual signal in {case['path']}")
            return

        n = tau.shape[0]
        response = None
        if case["response"] == "j2_position":
            q_leader = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
            if q_leader is not None and q_leader.shape[1] >= 2:
                response = q_leader[:, 1]
        else:
            delta = as_2d(get_array(npz, ["delta_corr", "delta_corr_raw"]), 6)
            q_ref = as_2d(get_array(npz, ["q_ref"]), 6)
            q_c = as_2d(get_array(npz, ["q_c_leader", "q_cmd_leader"]), 6)
            if delta is not None:
                response = np.linalg.norm(delta, axis=1)
            elif q_ref is not None and q_c is not None:
                m = min(q_ref.shape[0], q_c.shape[0])
                response = np.linalg.norm(q_c[:m, :6] - q_ref[:m, :6], axis=1)
        if response is None:
            print(f"[skip] observer attempts: missing response signal in {case['path']}")
            return

        q_leader = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
        if q_leader is not None:
            n = min(n, q_leader.shape[0])
        n = min(n, response.size)
        t = time_axis(npz, n)
        residual = np.nanmax(np.abs(tau[:n, :6]), axis=1)
        response = response[:n]
        q_dev_max = float("nan")
        if q_leader is not None:
            q_dev_max = float(np.nanmax(np.nanmax(np.abs(q_leader[:n, :6] - q_leader[0, :6]), axis=1)))
        meta = get_metadata(npz)
        prepared.append(
            {
                **case,
                "t": t - t[0],
                "residual": residual,
                "response_data": response,
                "tau_key": tau_key,
                "meta": meta,
                "p95_residual": float(np.nanpercentile(residual, 95)),
                "p95_response": float(np.nanpercentile(response, 95)),
                "q_dev_max": q_dev_max,
            }
        )

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.35), sharex=False)
    for ax, item in zip(axes, prepared):
        t = item["t"]
        keep = t <= min(20.0, float(t[-1]))
        color = item["color"]
        ax.plot(t[keep], item["residual"][keep], color=color, linewidth=1.05, label="Observer torque")
        ax.set_ylabel("Torque [Nm]")
        ax.grid(alpha=0.3)
        ax.set_title(item["label"], fontsize=10.8, fontweight="bold")

        ax_r = ax.twinx()
        ax_r.plot(
            t[keep],
            item["response_data"][keep],
            color="black",
            linewidth=1.0,
            alpha=0.78,
            label=item["response_label"],
        )
        ax_r.set_ylabel(item["response_label"])

        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(
            lines + lines_r,
            labels + labels_r,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.72",
            fontsize=8.5,
        )

    axes[-1].set_xlabel("Time [s]", labelpad=30)
    title = plot_title or "Comparison: Shi et al. and Yamane et al. Observer Output at Sine Tracking"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.19, hspace=0.62, left=0.09, right=0.91)

    print("[observer attempts sources]")
    for item in prepared:
        print(
            f"  {item['label']}: p95_residual={item['p95_residual']:.3f} "
            f"p95_response={item['p95_response']:.3f} q_dev_max={item['q_dev_max']:.3f}"
        )
        print(f"    {item['path']}")
    save_fig(fig, out_dir, "B_sensorless_observer_failure_modes", formats, dpi)


def plot_observer_vs_bota_snr(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    max_files: int,
    stem: str = "01_observer_vs_bota_snr",
) -> None:
    observer_noise: list[float] = []
    observer_signal: list[float] = []
    for path in find_npz(data_root / "epsilon_measurements", include=["e3_observer_yamane"]):
        loaded = load_npz(path)
        if loaded is None:
            continue
        tau = norm_rows(get_array(loaded.data, ["tau_ext_shi", "tau_residual", "tau_ext"]))
        if tau is None:
            continue
        value = float(np.nanpercentile(tau, 95))
        name = path.name.lower()
        if "static_hold" in name:
            observer_noise.append(value)
        else:
            observer_signal.append(value)

    bota_noise: list[float] = []
    bota_signal: list[float] = []
    for path in find_npz(data_root / "bota_minione_npz", include=["identity", "calibrated", "static_check"]):
        loaded = load_npz(path)
        if loaded is None:
            continue
        wrench = as_2d(get_array(loaded.data, ["wrench_base_calibrated", "wrench_base_calibrated_conditioned"]), 6)
        if wrench is not None:
            bota_noise.append(float(np.nanpercentile(np.linalg.norm(wrench[:, :3], axis=1), 95)))
    for path in find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"]):
        loaded = load_npz(path)
        if loaded is None:
            continue
        wrench = as_2d(get_array(loaded.data, ["wrench_base", "bota_wrench_conditioned", "wrench"]), 6)
        if wrench is not None:
            bota_signal.append(float(np.nanpercentile(np.linalg.norm(wrench[:, :3], axis=1), 95)))

    if not observer_noise or not observer_signal or not bota_noise or not bota_signal:
        print("[skip] plot 1: missing selected observer/BOTA noise or signal rows")
        return

    obs_noise = float(np.nanmedian(observer_noise))
    obs_signal = float(np.nanmedian(observer_signal))
    bot_noise = float(np.nanmedian(bota_noise))
    bot_signal = float(np.nanmedian(bota_signal))
    obs_snr = snr_db(obs_signal, obs_noise)
    bot_snr = snr_db(bot_signal, bot_noise)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, [obs_noise, bot_noise], width=width, label="no-contact/noise")
    axes[0].bar(x + width / 2, [obs_signal, bot_signal], width=width, label="active signal")
    axes[0].set_xticks(x, ["Observer\n[Nm]", "BOTA\n[N]"])
    axes[0].set_title("Noise floor vs. intervention signal")
    axes[0].set_ylabel("p95 norm")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(["Observer", "BOTA"], [obs_snr, bot_snr], color=["tab:blue", "tab:orange"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Separation proxy")
    axes[1].set_ylabel("20 log10(signal / noise) [dB]")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Observer vs. BOTA signal quality")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_phase_b_timeline(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    pattern: str,
    stem: str = "02_phase_b_episode_timeline",
) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=[pattern])
    if not candidates:
        candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb", "interventions"])
    if not candidates:
        print("[skip] plot 2: no Phase-B episode found")
        return

    loaded = None
    for path in candidates:
        candidate = load_npz(path)
        if candidate is not None and get_array(candidate.data, ["q_ref", "q_ref_follower"]) is not None:
            loaded = candidate
            break
    if loaded is None:
        print("[skip] plot 2: no usable Phase-B episode found")
        return

    npz = loaded.data
    q_ref = as_2d(get_array(npz, ["q_ref_follower", "q_ref"]), 6)
    q_cmd = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower", "q_compliant"]), 6)
    q_actual = as_2d(get_array(npz, ["q_follower", "q_actual"]), 6)
    delta = as_2d(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
    wrench = as_2d(get_array(npz, ["wrench_base", "bota_wrench_conditioned", "wrench", "bota_wrench_base"]), 6)
    epsilon = norm_rows(get_array(npz, ["epsilon_ur5e", "epsilon_follower"]))
    flag = get_array(npz, ["intervention_active", "is_correction"])
    if q_ref is None or q_cmd is None:
        print(f"[skip] plot 2: missing q_ref/q_cmd in {loaded.path}")
        return
    n = min(q_ref.shape[0], q_cmd.shape[0])
    t = time_axis(npz, n)
    q_ref = q_ref[:n]
    q_cmd = q_cmd[:n]
    delta_norm = norm_rows(delta[:n] if delta is not None else q_cmd - q_ref)
    wrench_force = np.linalg.norm(wrench[:n, :3], axis=1) if wrench is not None else np.full(n, np.nan)
    flag_arr = np.asarray(flag[:n], dtype=float).reshape(-1) if flag is not None and flag.size >= n else np.zeros(n)

    fig, axes = plt.subplots(5, 1, figsize=(10.0, 8.2), sharex=True)
    j = 0
    axes[0].plot(t, q_ref[:, j], label="q_ref J1", linewidth=1.3)
    axes[0].plot(t, q_cmd[:, j], label="q_cmd_ur5e J1", linewidth=1.1)
    if q_actual is not None:
        axes[0].plot(t, q_actual[:n, j], label="q_actual J1", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("joint [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, delta_norm, color="tab:green")
    axes[1].set_ylabel("||delta_q|| [rad]")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, wrench_force, color="tab:red")
    axes[2].set_ylabel("||F_BOTA|| [N]")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, epsilon[:n] if epsilon is not None else np.full(n, np.nan), color="tab:blue")
    axes[3].set_ylabel("||epsilon_ur5e|| [rad]")
    axes[3].grid(alpha=0.3)
    
    axes[4].plot(t, flag_arr, color="black", drawstyle="steps-post")
    axes[4].set_ylabel("intervention")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylim(-0.05, max(1.05, float(np.nanmax(flag_arr)) + 0.1))
    axes[4].grid(alpha=0.3)
    fig.suptitle(f"Phase-B episode timeline: {loaded.path.parent.name}")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_bota_raw_vs_calibrated(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    stem: str = "03_bota_raw_vs_calibrated",
) -> None:
    root = data_root / "bota_minione_npz"
    loaded = None
    fit_files = find_npz(root, include=["calibration_fit"])
    for path in fit_files:
        cand = load_npz(path)
        if cand is not None and get_array(cand.data, ["val_y", "calib_y"]) is not None:
            loaded = cand
            break
    candidates = find_npz(root, include=["calibrated"])
    for path in candidates:
        if loaded is not None:
            break
        cand = load_npz(path)
        if cand is None:
            continue
        if get_array(cand.data, ["wrench_base_static_ema", "wrench_base_static"]) is not None and get_array(
            cand.data, ["wrench_base_calibrated"]
        ) is not None:
            loaded = cand
            break
    if loaded is None:
        print("[skip] plot 3: no raw+calibrated BOTA file found")
        return

    npz = loaded.data
    raw = as_2d(get_array(npz, ["wrench_base_static_ema", "wrench_base_static", "val_y", "calib_y"]), 6)
    cal = as_2d(get_array(npz, ["wrench_base_calibrated", "val_residual", "calib_residual"]), 6)
    if raw is None or cal is None:
        print(f"[skip] plot 3: missing arrays in {loaded.path}")
        return
    n = min(raw.shape[0], cal.shape[0])
    raw = raw[:n]
    cal = cal[:n]
    t = time_axis(npz, n)
    raw_f = np.linalg.norm(raw[:, :3], axis=1)
    cal_f = np.linalg.norm(cal[:, :3], axis=1)
    raw_t = np.linalg.norm(raw[:, 3:], axis=1)
    cal_t = np.linalg.norm(cal[:, 3:], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(t, raw_f, label="raw/static", alpha=0.8)
    axes[0].plot(t, cal_f, label="calibrated", alpha=0.8)
    axes[0].set_title("Force residual magnitude")
    axes[0].set_ylabel("|F| [N]")
    axes[0].set_xlabel("time [s]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].boxplot(
        [raw_f, cal_f, raw_t, cal_t],
        tick_labels=["F raw", "F cal", "T raw", "T cal"],
        showfliers=False,
    )
    axes[1].set_title("Residual distribution")
    axes[1].set_ylabel("[N] / [Nm]")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("BOTA calibration: raw vs. calibrated")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_bota_ft_sensor_validation(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    plot_title: str = "",
) -> None:
    calibrated_path = (
        data_root
        / "bota_minione_npz"
        / "20260529_bota_identity_calibrated_static_check_20260529_130048.npz"
    )
    raw_path = (
        data_root
        / "bota_minione_npz"
        / "20260529_bota_identity_raw_static_check_20260529_130232.npz"
    )
    calibrated_loaded = load_npz(calibrated_path)
    raw_loaded = load_npz(raw_path)
    if calibrated_loaded is None or raw_loaded is None:
        return

    raw_npz = raw_loaded.data
    calibrated_npz = calibrated_loaded.data
    raw = as_2d(get_array(raw_npz, ["wrench_base_static_ema", "wrench_base_static", "wrench_base_raw"]), 6)
    calibrated = as_2d(get_array(calibrated_npz, ["wrench_base_calibrated"]), 6)
    q_raw = as_2d(get_array(raw_npz, ["q_leader"]), 6)
    q_calibrated = as_2d(get_array(calibrated_npz, ["q_leader"]), 6)
    if raw is None or calibrated is None or q_raw is None or q_calibrated is None:
        print("[skip] BOTA validation: missing static calibration arrays or q_leader")
        return

    n_raw = min(raw.shape[0], q_raw.shape[0])
    n_calibrated = min(calibrated.shape[0], q_calibrated.shape[0])
    t_raw = time_axis(raw_npz, n_raw)
    t_calibrated = time_axis(calibrated_npz, n_calibrated)
    raw_norm = np.linalg.norm(raw[:n_raw, :3], axis=1)
    cal_norm = np.linalg.norm(calibrated[:n_calibrated, :3], axis=1)
    raw_p95 = float(np.nanpercentile(raw_norm, 95))
    cal_p95 = float(np.nanpercentile(cal_norm, 95))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.25))

    keep_raw = t_raw <= min(30.0, float(t_raw[-1]))
    keep_calibrated = t_calibrated <= min(30.0, float(t_calibrated[-1]))
    axes[0].plot(t_raw[keep_raw], raw_norm[keep_raw], color="0.55", linewidth=1.0, label="Raw wrench norm")
    axes[0].plot(
        t_calibrated[keep_calibrated],
        cal_norm[keep_calibrated],
        color="#0072b2",
        linewidth=1.0,
        label="Calibrated wrench norm",
    )
    axes[0].set_title("No-contact static validation (calibrated vs. raw)", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Force norm [N]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.72",
        fontsize=8.0,
    )

    axes[1].plot(
        t_raw[keep_raw],
        (q_raw[:n_raw, 3] - 2.0 * np.pi)[keep_raw],
        color="#d55e00",
        linestyle="--",
        linewidth=1.0,
        label="J4 raw run",
    )
    axes[1].plot(
        t_raw[keep_raw],
        q_raw[:n_raw, 4][keep_raw],
        color="#009e73",
        linestyle="--",
        linewidth=1.0,
        label="J5 raw run",
    )
    axes[1].plot(
        t_calibrated[keep_calibrated],
        q_calibrated[:n_calibrated, 3][keep_calibrated],
        color="#d55e00",
        linewidth=1.0,
        alpha=0.86,
        label="J4 calibrated run",
    )
    axes[1].plot(
        t_calibrated[keep_calibrated],
        q_calibrated[:n_calibrated, 4][keep_calibrated],
        color="#009e73",
        linewidth=1.0,
        alpha=0.86,
        label="J5 calibrated run",
    )
    axes[1].set_title("Leader motion in both validation runs", fontsize=10.8, fontweight="bold")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Joint position [rad]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.72",
        fontsize=8.0,
    )

    title = plot_title or "BOTA F/T Sensor Calibration Validation"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.895)
    fig.subplots_adjust(left=0.075, right=0.935, top=0.80, bottom=0.27, wspace=0.34)

    print("[BOTA validation sources]")
    print(
        f"  static check: raw_p95={raw_p95:.3f} N "
        f"calibrated_p95={cal_p95:.3f} N"
    )
    print(f"  raw q_ptp J4={np.ptp(q_raw[:n_raw, 3]):.3f} rad J5={np.ptp(q_raw[:n_raw, 4]):.3f} rad")
    print(
        f"  calibrated q_ptp J4={np.ptp(q_calibrated[:n_calibrated, 3]):.3f} rad "
        f"J5={np.ptp(q_calibrated[:n_calibrated, 4]):.3f} rad"
    )
    print(f"    {raw_path}")
    print(f"    {calibrated_path}")
    save_fig(fig, out_dir, "K4_bota_ft_sensor_calibration", formats, dpi)


def plot_bota_admittance_response(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    plot_title: str = "",
) -> None:
    response_path = (
        data_root
        / "epsilon_measurements_calibrated_bota"
        / "se3_trans_stronger"
        / "epsilon_leader_admittance_position_static_hold_1908620.npz"
    )
    response_loaded = load_npz(response_path)
    if response_loaded is None:
        return

    response_npz = response_loaded.data
    response_meta = get_metadata(response_npz)
    wrench = as_2d(get_array(response_npz, ["bota_wrench_conditioned", "wrench"]), 6)
    delta = as_2d(get_array(response_npz, ["delta_corr", "delta_corr_raw"]), 6)
    if wrench is None or delta is None:
        print(f"[skip] BOTA admittance: missing wrench/delta arrays in {response_path}")
        return

    n_response = min(wrench.shape[0], delta.shape[0])
    t_response = time_axis(response_npz, n_response)
    keep_response = t_response <= min(30.0, float(t_response[-1]))
    force_norm = np.linalg.norm(wrench[:n_response, :3], axis=1)
    delta_max_abs = np.nanmax(np.abs(delta[:n_response, :6]), axis=1)
    delta_limit = float(response_meta.get("adm_delta_max", np.nan))
    intervention = get_array(response_npz, ["intervention_active"])

    tracking_paths = sorted(
        (data_root / "epsilon_measurements" / "2026-05-22_thesis_sine_nominal").glob("*.npz")
    )
    tracking_rows = []
    for path in tracking_paths:
        loaded = load_npz(path)
        if loaded is None:
            continue
        npz = loaded.data
        meta = get_metadata(npz)
        q_ref = as_2d(get_array(npz, ["q_ref"]), 6)
        q_cmd = as_2d(get_array(npz, ["q_c_leader", "q_cmd_leader"]), 6)
        q_leader = as_2d(get_array(npz, ["q_leader"]), 6)
        if q_ref is None or q_cmd is None or q_leader is None:
            continue
        joint = int(meta.get("joint_index", 1))
        if joint < 0 or joint >= 6:
            joint = 1
        n = min(q_ref.shape[0], q_cmd.shape[0], q_leader.shape[0])
        t = time_axis(npz, n)
        crossing = first_rising_midline_crossing(t, q_ref[:n, joint])
        if crossing is None:
            crossing = 0.0
        tracking_rows.append(
            {
                "path": path,
                "joint": joint,
                "t": t - crossing,
                "ref": q_ref[:n, joint],
                "cmd": q_cmd[:n, joint],
                "leader": q_leader[:n, joint],
            }
        )

    if len(tracking_rows) < 2:
        print("[skip] BOTA admittance: need at least two sine-push tracking runs")
        return

    joint = tracking_rows[0]["joint"]
    dt_candidates = []
    t_end_candidates = []
    for row in tracking_rows:
        valid = row["t"] >= 0.0
        diffs = np.diff(row["t"][valid])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            dt_candidates.append(float(np.nanmedian(diffs)))
        if np.any(valid):
            t_end_candidates.append(float(np.nanmax(row["t"][valid])))
    dt = float(np.nanmedian(dt_candidates)) if dt_candidates else 1.0 / 240.0
    t_end = min(30.0, min(t_end_candidates))
    n_common = max(2, int(np.floor(t_end / dt)) + 1)
    t_common = np.linspace(0.0, dt * (n_common - 1), n_common)

    ref_stack = []
    leader_stack = []
    for row in tracking_rows:
        order = np.argsort(row["t"])
        t_row = row["t"][order]
        valid = np.isfinite(t_row)
        t_row = t_row[valid]
        if t_row.size < 2:
            continue
        ref_stack.append(np.interp(t_common, t_row, row["ref"][order][valid]))
        leader_stack.append(np.interp(t_common, t_row, row["leader"][order][valid]))

    ref_arr = np.asarray(ref_stack)
    leader_arr = np.asarray(leader_stack)
    if ref_arr.shape[0] < 2 or leader_arr.shape[0] < 2:
        print("[skip] BOTA admittance: fewer than two interpolated tracking runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.25))

    if intervention is not None:
        flag = np.asarray(intervention[:n_response], dtype=float).reshape(-1) > 0.5
        transitions = np.diff(np.r_[False, flag[keep_response], False].astype(int))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        t_keep = t_response[keep_response]
        for start, stop in zip(starts, stops):
            if start < t_keep.size and stop > 0:
                axes[0].axvspan(t_keep[start], t_keep[min(stop - 1, t_keep.size - 1)], color="0.86", alpha=0.45, linewidth=0)

    axes[0].plot(t_response[keep_response], force_norm[keep_response], color="#0072b2", linewidth=1.0, label="Measured force")
    axes[0].set_title("SE(3) admittance response to human contact", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Force norm [N]")
    axes[0].grid(alpha=0.3)
    ax_delta = axes[0].twinx()
    ax_delta.plot(t_response[keep_response], delta_max_abs[keep_response], color="#d55e00", linewidth=1.0, label="Correction offset")
    if np.isfinite(delta_limit):
        ax_delta.axhline(delta_limit, color="#cc0000", linestyle="--", linewidth=0.95, label=r"$\delta_{\max}=0.08$ rad")
    ax_delta.set_ylabel("Max. correction offset [rad]")
    lines, labels = axes[0].get_legend_handles_labels()
    lines_r, labels_r = ax_delta.get_legend_handles_labels()
    axes[0].legend(
        lines + lines_r,
        labels + labels_r,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.72",
        fontsize=8.0,
    )

    ref_mean = np.nanmean(ref_arr, axis=0)
    leader_mean = np.nanmean(leader_arr, axis=0)
    axes[1].plot(t_common, ref_mean, color="0.35", linestyle="--", linewidth=1.0, label="Reference")
    axes[1].fill_between(
        t_common,
        np.nanmin(leader_arr, axis=0),
        np.nanmax(leader_arr, axis=0),
        color="#d55e00",
        alpha=0.16,
        linewidth=0,
    )
    axes[1].plot(t_common, leader_mean, color="#d55e00", linewidth=1.0, label="Leader position")
    axes[1].set_title(f"Aggregated sine tracking (J{joint + 1})", fontsize=10.8, fontweight="bold")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Joint position [rad]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.72",
        fontsize=8.0,
    )

    title = plot_title or "BOTA-Based Position Admittance Validation"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.895)
    fig.subplots_adjust(left=0.075, right=0.935, top=0.80, bottom=0.27, wspace=0.34)

    tracking_error = leader_arr - ref_arr
    print("[BOTA admittance source]")
    print(
        f"  force_p95={np.nanpercentile(force_norm, 95):.3f} N "
        f"delta_maxabs_p95={np.nanpercentile(delta_max_abs, 95):.3f} rad "
        f"tracking_p95={np.nanpercentile(np.abs(tracking_error), 95):.3f} rad"
    )
    print(f"    response: {response_path}")
    for row in tracking_rows:
        print(f"    tracking: {row['path']}")
    save_fig(fig, out_dir, "K4_bota_position_admittance", formats, dpi)


def plot_delta_q_histogram(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    max_files: int,
    stem: str = "04_delta_q_histogram",
) -> None:
    files = final_phase_b_files(data_root, max_files)
    rows = []
    rates = []
    for path in files:
        loaded = load_npz(path)
        if loaded is None:
            continue
        delta = as_2d(get_array(loaded.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
        if delta is None:
            continue
        rows.append(delta)
        t = time_axis(loaded.data, delta.shape[0])
        if t.size > 1:
            dt = np.nanmedian(np.diff(t))
            if dt > 0:
                rates.append(np.diff(delta, axis=0) / dt)
    if not rows:
        print("[skip] plot 4: no delta_q arrays found")
        return
    delta_all = np.concatenate(rows, axis=0)
    norm = np.linalg.norm(delta_all, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].hist(norm, bins=60, color="tab:green", alpha=0.85)
    axes[0].set_title("Correction magnitude")
    axes[0].set_xlabel("||delta_q|| [rad]")
    axes[0].set_ylabel("samples")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(
        [delta_all[:, i] for i in range(min(6, delta_all.shape[1]))],
        tick_labels=JOINT_LABELS[: delta_all.shape[1]],
        showfliers=False,
    )
    axes[1].set_title("Per-joint correction distribution")
    axes[1].set_ylabel("delta_q [rad]")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Phase-B delta_q distribution")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)

    if rates:
        rate_all = np.concatenate(rates, axis=0)
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(np.linalg.norm(rate_all, axis=1), bins=60, color="tab:olive", alpha=0.85)
        ax.set_title("Correction smoothness")
        ax.set_xlabel("||d(delta_q)/dt|| [rad/s]")
        ax.set_ylabel("samples")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_dir, f"{stem}_rate", formats, dpi)


def plot_follower_tracking_error(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int, max_files: int) -> None:
    groups = {
        "no intervention": find_npz(
            data_root / "cr_dagger_npz",
            include=["phaseb_calibrated_bota_static_hold_no_interventions"],
        ),
        "intervention": find_npz(
            data_root / "cr_dagger_npz",
            include=["phaseb_calibrated_bota_static_hold_interventions"],
        ),
    }
    rows: dict[str, list[np.ndarray]] = {label: [] for label in groups}
    for label, files in groups.items():
        for path in limit_files(files, max_files):
            loaded = load_npz(path)
            if loaded is None:
                continue
            eps = as_2d(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]), 6)
            if eps is not None:
                rows[label].append(eps)
    if not any(rows.values()):
        print("[skip] plot 4/5: no calibrated static-hold follower epsilon arrays found")
        return
    labels = []
    rms_values = []
    p95_values = []
    norm_rows_all = []
    for label, chunks in rows.items():
        if not chunks:
            continue
        eps = np.concatenate(chunks, axis=0)
        labels.append(label)
        rms_values.append(float(np.sqrt(np.nanmean(np.linalg.norm(eps, axis=1) ** 2))))
        p95_values.append(float(np.nanpercentile(np.linalg.norm(eps, axis=1), 95)))
        norm_rows_all.append(np.linalg.norm(eps, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, rms_values, width=0.36, label="RMS")
    axes[0].bar(x + 0.18, p95_values, width=0.36, label="p95")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("||epsilon_ur5e|| [rad]")
    axes[0].set_title("Static-hold follower tracking")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].boxplot(norm_rows_all, tick_labels=labels, showfliers=False)
    axes[1].set_title("Error distribution")
    axes[1].set_ylabel("||epsilon_ur5e|| [rad]")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "03_follower_tracking_error", formats, dpi)


def plot_calibration_residual_pose_sweep(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    files = find_npz(data_root / "bota_minione_npz", include=["calibration_fit"])
    loaded = load_npz(files[0]) if files else None
    if loaded is None:
        print("[skip] plot 6: no calibration fit NPZ found")
        return
    npz = loaded.data
    calib_y = as_2d(get_array(npz, ["calib_y"]), 6)
    calib_res = as_2d(get_array(npz, ["calib_residual"]), 6)
    val_y = as_2d(get_array(npz, ["val_y"]), 6)
    val_res = as_2d(get_array(npz, ["val_residual"]), 6)
    if calib_y is None or calib_res is None:
        print("[skip] plot 6: missing calibration residual arrays")
        return

    labels = ["train raw", "train corrected"]
    values = [
        np.linalg.norm(calib_y[:, :3], axis=1),
        np.linalg.norm(calib_res[:, :3], axis=1),
    ]
    if val_y is not None and val_res is not None:
        labels += ["val raw", "val corrected"]
        values += [np.linalg.norm(val_y[:, :3], axis=1), np.linalg.norm(val_res[:, :3], axis=1)]

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_title("BOTA pose-sweep calibration residual")
    ax.set_ylabel("|F residual| [N]")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "06_bota_pose_sweep_residual", formats, dpi)


def plot_static_drift(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    stem: str = "02_bota_static_check",
) -> None:
    files = find_npz(data_root / "bota_minione_npz", include=["identity", "calibrated", "static_check"])
    loaded = load_npz(files[0]) if files else None
    if loaded is None:
        print("[skip] plot 7: no calibrated static-check NPZ found")
        return
    wrench = as_2d(get_array(loaded.data, ["wrench_base_calibrated", "wrench_base_calibrated_conditioned", "wrench_base_static_ema"]), 6)
    if wrench is None:
        print("[skip] plot 7: no wrench array found")
        return
    t = time_axis(loaded.data, wrench.shape[0])
    force = wrench[:, :3]
    force_norm = np.linalg.norm(force, axis=1)
    p95 = float(np.nanpercentile(force_norm, 95))
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    for i in range(3):
        ax.plot(t, force[:, i], label=WRENCH_LABELS[i], linewidth=0.9)
    ax.axhline(p95, color="black", linestyle="--", linewidth=0.8, label=f"+p95 |F|={p95:.3f} N")
    ax.axhline(-p95, color="black", linestyle="--", linewidth=0.8, label="-p95 |F|")
    ax.set_ylabel("force [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("BOTA static null stability after calibration")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)


def extract_rising_edges(flag: np.ndarray) -> np.ndarray:
    f = np.asarray(flag, dtype=float).reshape(-1) > 0.5
    if f.size < 2:
        return np.array([], dtype=int)
    return np.flatnonzero(f[1:] & ~f[:-1]) + 1


def interp_window(t: np.ndarray, y: np.ndarray, center_idx: int, grid: np.ndarray) -> np.ndarray | None:
    t0 = float(t[center_idx])
    if t0 + grid[0] < t[0] or t0 + grid[-1] > t[-1]:
        return None
    return np.interp(t0 + grid, t, y)


def plot_intervention_aligned_average(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    max_files: int,
    stem: str = "07_intervention_aligned_average",
) -> None:
    files = limit_files(
        find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"]),
        max_files,
    )
    grid = np.linspace(-1.0, 3.0, 241)
    delta_segments = []
    wrench_segments = []
    for path in files:
        loaded = load_npz(path)
        if loaded is None:
            continue
        flag = get_array(loaded.data, ["intervention_active", "is_correction"])
        delta = norm_rows(get_array(loaded.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
        wrench = as_2d(get_array(loaded.data, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
        if flag is None or delta is None or wrench is None:
            continue
        n = min(flag.size, delta.size, wrench.shape[0])
        t = time_axis(loaded.data, n)
        fmag = np.linalg.norm(wrench[:n, :3], axis=1)
        for idx in extract_rising_edges(flag[:n]):
            ds = interp_window(t, delta[:n], idx, grid)
            ws = interp_window(t, fmag, idx, grid)
            if ds is not None and ws is not None:
                delta_segments.append(ds)
                wrench_segments.append(ws)
    if not delta_segments:
        print("[skip] plot 8: no intervention windows found")
        return
    delta_arr = np.asarray(delta_segments)
    wrench_arr = np.asarray(wrench_segments)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    for ax, arr, label, color in [
        (axes[0], wrench_arr, "||F_BOTA|| [N]", "tab:red"),
        (axes[1], delta_arr, "||delta_q|| [rad]", "tab:green"),
    ]:
        mean = np.nanmean(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
        ax.plot(grid, mean, color=color)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.2)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("time around intervention onset [s]")
    axes[0].set_title(f"Intervention-aligned average (n={len(delta_segments)})")
    fig.tight_layout()
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_admittance_step_response(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"])
    loaded = first_usable(candidates, ["intervention_active", "delta_q", "wrench_base"])
    if loaded is None:
        print("[skip] plot 3: no static-hold intervention episode for step response")
        return
    npz = loaded.data
    flag = np.asarray(get_array(npz, ["intervention_active", "is_correction"]), dtype=float).reshape(-1)
    delta = norm_rows(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
    if delta is None or wrench is None:
        print("[skip] plot 3: missing delta/wrench")
        return
    n = min(flag.size, delta.size, wrench.shape[0])
    t = time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    edges = extract_rising_edges(flag[:n])
    if edges.size == 0:
        print("[skip] plot 3: no intervention edge")
        return
    # Use the strongest event so the compliance signature is visually clear.
    best = max(edges, key=lambda idx: np.nanmax(force[idx : min(n, idx + int(4.0 / max(np.median(np.diff(t)), 1e-6)))]))
    window = (t >= t[best] - 0.6) & (t <= t[best] + 3.2)
    tw = t[window] - t[best]

    fig, ax1 = plt.subplots(figsize=(8.4, 3.8))
    ax2 = ax1.twinx()
    ax1.plot(tw, force[window], color="tab:red", label="||F_BOTA||")
    ax2.plot(tw, delta[:n][window], color="tab:green", label="||delta_q||")
    ax1.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("time around intervention onset [s]")
    ax1.set_ylabel("||F_BOTA|| [N]", color="tab:red")
    ax2.set_ylabel("||delta_q|| [rad]", color="tab:green")
    ax1.grid(alpha=0.3)
    fig.suptitle("Admittance response: force onset to residual correction")
    fig.tight_layout()
    save_fig(fig, out_dir, "03_admittance_step_response", formats, dpi)


def choose_strong_event(npz: object) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    flag = np.asarray(get_array(npz, ["intervention_active", "is_correction"]), dtype=float).reshape(-1)
    delta = norm_rows(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
    if flag.size == 0 or delta is None or wrench is None:
        return None
    n = min(flag.size, delta.size, wrench.shape[0])
    t = time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    edges = extract_rising_edges(flag[:n])
    if edges.size == 0:
        return None
    dt = max(float(np.nanmedian(np.diff(t))), 1e-6) if t.size > 1 else 1.0 / 30.0
    best = max(edges, key=lambda idx: np.nanmax(force[idx : min(n, idx + int(4.0 / dt))]))
    return int(best), t, delta[:n], force[:n], flag[:n]


def plot_intervention_zoom(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"])
    loaded = first_usable(candidates, ["intervention_active", "delta_q", "wrench_base", "q_ref", "q_cmd_ur5e"])
    if loaded is None:
        print("[skip] plot 6: no static-hold intervention episode for zoom")
        return
    npz = loaded.data
    picked = choose_strong_event(npz)
    if picked is None:
        print("[skip] plot 6: no intervention edge")
        return
    best, t, delta_norm, force, flag = picked
    q_ref = as_2d(get_array(npz, ["q_ref_follower", "q_ref"]), 6)
    q_cmd = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower"]), 6)
    q_actual = as_2d(get_array(npz, ["q_follower", "q_actual"]), 6)
    n = min(q_ref.shape[0], q_cmd.shape[0], t.size)
    window = (t[:n] >= t[best] - 0.6) & (t[:n] <= t[best] + 3.2)
    tw = t[:n][window] - t[best]
    j = 0

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(tw, q_ref[:n, j][window], label="q_ref J1", linewidth=1.0)
    axes[0].plot(tw, q_cmd[:n, j][window], label="q_cmd_ur5e J1", linewidth=1.0)
    if q_actual is not None:
        axes[0].plot(tw, q_actual[:n, j][window], label="q_actual J1", linewidth=1.0, alpha=0.85)
    axes[0].set_ylabel("joint [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(tw, delta_norm[:n][window], color="tab:green")
    axes[1].set_ylabel("||delta_q|| [rad]")
    axes[1].grid(alpha=0.3)
    axes[2].plot(tw, force[:n][window], color="tab:red")
    axes[2].set_ylabel("||F_BOTA|| [N]")
    axes[2].grid(alpha=0.3)
    axes[3].plot(tw, flag[:n][window], color="black", drawstyle="steps-post")
    axes[3].set_ylabel("intervention")
    axes[3].set_xlabel("time around intervention onset [s]")
    axes[3].grid(alpha=0.3)
    for ax in axes:
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    fig.suptitle("Single intervention zoom")
    fig.tight_layout()
    save_fig(fig, out_dir, "06_intervention_zoom_single_event", formats, dpi)


def plot_leader_follower_sync(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_calibrated_bota_static_hold_interventions"])
    if not candidates:
        candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_cable_routing_w_intervention_bota6dof"])
    loaded = first_usable(candidates, ["q_leader", "q_follower", "q_cmd_ur5e"])
    if loaded is None:
        print("[skip] plot 4: no leader/follower sync episode")
        return
    npz = loaded.data
    q_leader = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
    q_follower = as_2d(get_array(npz, ["q_follower"]), 6)
    q_cmd = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower"]), 6)
    if q_leader is None or q_follower is None or q_cmd is None:
        print("[skip] plot 4: missing q streams")
        return
    n = min(q_leader.shape[0], q_follower.shape[0], q_cmd.shape[0])
    t = time_axis(npz, n)
    start = int(0.2 * n)
    end = min(n, start + int(10.0 / max(np.nanmedian(np.diff(t)), 1e-6)))
    sl = slice(start, end)
    tw = t[sl] - t[start]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.2), sharex=True)
    for ax, j in zip(axes, [0, 1]):
        # Relative signals avoid frame-offset confusion while preserving synchronization.
        ax.plot(tw, q_leader[sl, j] - q_leader[start, j], label=f"q_leader rel J{j + 1}", linewidth=1.0)
        ax.plot(tw, q_follower[sl, j] - q_follower[start, j], label=f"q_follower rel J{j + 1}", linewidth=1.0)
        ax.plot(tw, q_cmd[sl, j] - q_cmd[start, j], label=f"q_cmd rel J{j + 1}", linewidth=1.0, linestyle="--")
        ax.set_ylabel("relative joint [rad]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[1].set_xlabel("time [s]")
    fig.suptitle("Leader-follower synchronization (relative joint motion)")
    fig.tight_layout()
    save_fig(fig, out_dir, "04_leader_follower_synchronization", formats, dpi)


def select_longest(paths: Sequence[Path], required: Sequence[str]) -> LoadedNpz | None:
    best: LoadedNpz | None = None
    best_n = -1
    for path in paths:
        loaded = load_npz(path)
        if loaded is None:
            continue
        if not all(get_array(loaded.data, [key]) is not None for key in required):
            continue
        n = 0
        for key in ("timestamps", "t_mono", "t_rel"):
            arr = get_array(loaded.data, [key])
            if arr is not None:
                n = max(n, int(arr.size))
        if n > best_n:
            best = loaded
            best_n = n
    return best


def plot_whiteboard_contact_divergence(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_whiteboard_whiping_interventions"])
    loaded = select_longest(candidates, ["q_leader", "q_follower", "epsilon_ur5e", "delta_q", "wrench_base"])
    if loaded is None:
        print("[skip] T1: no whiteboard contact-divergence episode")
        return
    npz = loaded.data
    q_leader = as_2d(get_array(npz, ["q_leader"]), 6)
    q_follower = as_2d(get_array(npz, ["q_follower"]), 6)
    eps = norm_rows(get_array(npz, ["epsilon_ur5e", "epsilon_follower"]))
    delta = norm_rows(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
    if q_leader is None or q_follower is None or eps is None or delta is None or wrench is None:
        print("[skip] T1: missing streams")
        return
    n = min(q_leader.shape[0], q_follower.shape[0], eps.size, delta.size, wrench.shape[0])
    t = time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    j = 2
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 7.6), sharex=True)
    axes[0].plot(t, q_leader[:n, j] - q_leader[0, j], label=f"q_leader rel J{j + 1}")
    axes[0].plot(t, q_follower[:n, j] - q_follower[0, j], label=f"q_follower rel J{j + 1}")
    axes[0].set_ylabel("relative joint [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, eps[:n], color="tab:blue")
    axes[1].set_ylabel("||epsilon_ur5e|| [rad]")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, force, label="BOTA force norm", color="tab:red")
    axes[2].set_ylabel("||F|| [N]")
    axes[2].grid(alpha=0.3)
    axes[3].plot(t, delta[:n], color="tab:green")
    axes[3].set_ylabel("||delta_q|| [rad]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.3)
    fig.suptitle("T1 Whiteboard wiping: contact-induced divergence and correction label")
    fig.tight_layout()
    save_fig(fig, out_dir, "T1_whiteboard_contact_divergence", formats, dpi)


def plot_cable_routing_successful_rollout(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    candidates = find_npz(data_root / "cr_dagger_npz", include=["phaseb_cable_routing_w_intervention_bota6dof"])
    loaded = select_longest(candidates, ["q_ref", "q_cmd_ur5e", "delta_q", "wrench_base"])
    if loaded is None:
        print("[skip] T2: no cable-routing BOTA6DoF episode")
        return
    npz = loaded.data
    q_ref = as_2d(get_array(npz, ["q_ref_follower", "q_ref"]), 6)
    q_cmd = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower"]), 6)
    q_actual = as_2d(get_array(npz, ["q_follower"]), 6)
    delta = norm_rows(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
    flag = get_array(npz, ["intervention_active", "is_correction"])
    eps = norm_rows(get_array(npz, ["epsilon_ur5e", "epsilon_follower"]))
    if q_ref is None or q_cmd is None or delta is None or wrench is None:
        print("[skip] T2: missing rollout streams")
        return
    n = min(q_ref.shape[0], q_cmd.shape[0], delta.size, wrench.shape[0])
    t = time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    flag_arr = np.asarray(flag[:n], dtype=float).reshape(-1) if flag is not None and flag.size >= n else np.zeros(n)
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 8.4), sharex=True)
    for j in [0, 1]:
        axes[0].plot(t, q_ref[:n, j], label=f"q_ref J{j + 1}", linewidth=0.9)
        axes[0].plot(t, q_cmd[:n, j], label=f"q_cmd J{j + 1}", linewidth=0.9, linestyle="--")
    axes[0].set_ylabel("joint [rad]")
    axes[0].legend(ncol=2, loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, delta[:n], color="tab:green")
    axes[1].set_ylabel("||delta_q|| [rad]")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, force, color="tab:red")
    axes[2].set_ylabel("||F_BOTA|| [N]")
    axes[2].grid(alpha=0.3)
    axes[3].plot(t, eps[:n] if eps is not None else np.full(n, np.nan), color="tab:blue")
    axes[3].set_ylabel("||epsilon|| [rad]")
    axes[3].grid(alpha=0.3)
    axes[4].plot(t, flag_arr, color="black", drawstyle="steps-post")
    axes[4].set_ylabel("intervention")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(alpha=0.3)
    fig.suptitle("T2 Cable routing BOTA6DoF rollout")
    fig.tight_layout()
    save_fig(fig, out_dir, "T2_cable_routing_successful_rollout", formats, dpi)


def plot_latency_histogram(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int, max_files: int) -> None:
    loop_rows = []
    age_rows = []
    overrun_values = []
    for path in final_phase_b_files(data_root, max_files):
        loaded = load_npz(path)
        if loaded is None:
            continue
        loop = get_array(loaded.data, ["phase_b_loop_dt_s"])
        age = get_array(loaded.data, ["policy_action_age_s"])
        over = get_array(loaded.data, ["phase_b_overrun"])
        if loop is not None:
            loop_rows.append(np.asarray(loop, dtype=float).reshape(-1))
        if age is not None:
            age_rows.append(np.asarray(age, dtype=float).reshape(-1))
        if over is not None:
            overrun_values.append(float(np.mean(np.asarray(over, dtype=float).reshape(-1) > 0.5)))
    if not loop_rows and not age_rows:
        print("[skip] plot 5: no latency/timing fields found")
        return
    loop_all = np.concatenate(loop_rows) if loop_rows else np.array([])
    age_all = np.concatenate(age_rows) if age_rows else np.array([])
    loop_all = loop_all[np.isfinite(loop_all) & (loop_all > 0.0) & (loop_all < 0.1)]
    age_all = age_all[np.isfinite(age_all) & (age_all >= 0.0) & (age_all < 1.0)]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    if loop_all.size:
        axes[0].hist(loop_all * 1000.0, bins=60, color="tab:blue", alpha=0.85)
        axes[0].axvline(np.nanpercentile(loop_all, 95) * 1000.0, color="black", linestyle="--", linewidth=0.8, label="p95")
    axes[0].set_title("Phase-B loop period")
    axes[0].set_xlabel("loop_dt [ms]")
    axes[0].set_ylabel("samples")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    if age_all.size:
        axes[1].hist(age_all * 1000.0, bins=60, color="tab:orange", alpha=0.85)
        axes[1].axvline(np.nanpercentile(age_all, 95) * 1000.0, color="black", linestyle="--", linewidth=0.8, label="p95")
    overrun_text = f"median overrun: {100.0 * np.nanmedian(overrun_values):.2f}%" if overrun_values else "overrun: n/a"
    axes[1].set_title(f"Policy action age ({overrun_text})")
    axes[1].set_xlabel("policy_action_age [ms]")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "05_latency_timing_histogram", formats, dpi)


def phase_b_timing_files(data_root: Path, max_files: int = 0) -> list[Path]:
    folder = data_root / "cr_dagger_npz" / "phaseB_cable_routing_w_intervention_Bota6DoF"
    if not folder.exists():
        print(f"[warn] phase-b timing folder not found: {folder}")
        return []
    return limit_files(sorted(p for p in folder.rglob("*.npz") if p.is_file()), max_files)


def phase_b_time_axis(npz: object, n: int) -> np.ndarray:
    for key in ("t_rel", "timestamps", "t_mono", "host_t"):
        arr = get_array(npz, [key])
        if arr is not None and arr.size >= n:
            t = np.asarray(arr[:n], dtype=float).reshape(-1)
            if np.count_nonzero(np.isfinite(t)) > 0.9 * n and float(np.nanmax(t) - np.nanmin(t)) > 0.0:
                return t - t[0]
    rec_dt = get_array(npz, ["recording_frame_dt_s"])
    if rec_dt is not None and rec_dt.size >= n:
        dt = np.asarray(rec_dt[:n], dtype=float).reshape(-1)
        dt = np.where(np.isfinite(dt) & (dt > 0.0) & (dt < 1.0), dt, np.nan)
        if np.count_nonzero(np.isfinite(dt)) > 0.9 * n:
            fill = float(np.nanmedian(dt))
            dt = np.where(np.isfinite(dt), dt, fill)
            t = np.cumsum(dt)
            return t - t[0]
    loop = get_array(npz, ["phase_b_loop_dt_s"])
    if loop is not None and loop.size >= n:
        dt = np.asarray(loop[:n], dtype=float).reshape(-1)
        dt = np.where(np.isfinite(dt) & (dt > 0.0) & (dt < 0.1), dt, np.nan)
        if np.count_nonzero(np.isfinite(dt)) > 0.9 * n:
            fill = float(np.nanmedian(dt))
            dt = np.where(np.isfinite(dt), dt, fill)
            t = np.cumsum(dt)
            return t - t[0]
    return time_axis(npz, n)


def shade_active_regions(ax: plt.Axes, t: np.ndarray, flag: np.ndarray, label: str = "Human correction") -> None:
    active = np.asarray(flag, dtype=bool).reshape(-1)
    if active.size == 0 or t.size == 0:
        return
    n = min(t.size, active.size)
    active = active[:n]
    tt = t[:n]
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    stops = np.flatnonzero(active & np.r_[~active[1:], True]) + 1
    first = True
    for start, stop in zip(starts, stops):
        ax.axvspan(
            tt[start],
            tt[min(stop - 1, n - 1)],
            color="0.86",
            alpha=0.55,
            linewidth=0,
            label=label if first else None,
        )
        first = False


def plot_phase_b_timing(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    max_files: int,
    title_override: str = "",
) -> None:
    loop_rows = []
    age_rows = []
    candidates: list[LoadedNpz] = []
    for path in phase_b_timing_files(data_root, max_files):
        loaded = load_npz(path)
        if loaded is None:
            continue
        loop = get_array(loaded.data, ["phase_b_loop_dt_s"])
        age = get_array(loaded.data, ["policy_action_age_s"])
        if loop is not None:
            loop_rows.append(np.asarray(loop, dtype=float).reshape(-1))
        if age is not None:
            age_rows.append(np.asarray(age, dtype=float).reshape(-1))
        if get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]) is not None:
            candidates.append(loaded)

    if not loop_rows and not age_rows:
        print("[skip] phase-b timing: no timing fields found")
        return

    loop_raw = np.concatenate(loop_rows) if loop_rows else np.array([])
    age_raw = np.concatenate(age_rows) if age_rows else np.array([])
    loop_all = loop_raw[np.isfinite(loop_raw) & (loop_raw > 0.0) & (loop_raw < 0.1)]
    age_valid = age_raw[np.isfinite(age_raw) & (age_raw >= 0.0)]
    age_all = age_valid[age_valid < 0.5]
    age_outliers = int(age_valid.size - age_all.size)

    loop_ms = loop_all * 1000.0
    age_ms = age_all * 1000.0

    tracking_loaded = None
    tracking_score = -1
    for loaded in candidates:
        eps = norm_rows(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]))
        flag = get_array(loaded.data, ["intervention_active", "is_correction"])
        if eps is None or eps.size < 500 or flag is None:
            continue
        active_ratio = float(np.nanmean(np.asarray(flag, dtype=float).reshape(-1)[: eps.size] > 0.5))
        score = eps.size + int(active_ratio > 0.1) * 10000
        if score > tracking_score:
            tracking_loaded = loaded
            tracking_score = score
    if tracking_loaded is None and candidates:
        tracking_loaded = max(
            candidates,
            key=lambda loaded: norm_rows(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"])).size,
        )

    fig = plt.figure(figsize=(11.0, 4.25))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.85], wspace=0.32)
    left = outer[0].subgridspec(2, 1, hspace=0.72)
    ax_loop = fig.add_subplot(left[0])
    ax_age = fig.add_subplot(left[1])
    ax_track = fig.add_subplot(outer[1])
    title = title_override or "Phase-B Timing and Follower Tracking"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    if loop_ms.size:
        ax_loop.hist(loop_ms, bins=60, range=(3.0, 4.0), density=True, color="tab:blue", alpha=0.85)
        loop_med = float(np.nanmedian(loop_ms))
        loop_p95 = float(np.nanpercentile(loop_ms, 95))
        loop_std = float(np.nanstd(loop_ms))
        loop_max = float(np.nanmax(loop_ms))
        ax_loop.axvline(3.333, color="tab:red", linestyle="-", linewidth=1.6, label="Target 3.33 ms")
        ax_loop.axvline(loop_p95, color="black", linestyle="--", linewidth=1.3, label=f"p95: {loop_p95:.2f} ms")
        ax_loop.text(
            0.98,
            0.82,
            f"median {loop_med:.2f} ms\nstd {loop_std:.3f} ms\nmax {loop_max:.2f} ms",
            transform=ax_loop.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
    ax_loop.set_title("Loop cycle time")
    ax_loop.set_xlim(3.0, 4.0)
    ax_loop.set_xlabel("Cycle time [ms]")
    ax_loop.set_ylabel("Density")
    ax_loop.grid(axis="y", alpha=0.3)
    ax_loop.legend(loc="lower right", fontsize=7.5, frameon=True, handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    if age_ms.size:
        ax_age.hist(age_ms, bins=80, range=(0.0, 100.0), density=True, color="tab:orange", alpha=0.85)
        age_med = float(np.nanmedian(age_ms))
        age_p95 = float(np.nanpercentile(age_ms, 95))
        ax_age.axvline(50.0, color="tab:red", linestyle="-", linewidth=1.6, label="50 ms interval")
        ax_age.axvline(age_p95, color="black", linestyle="--", linewidth=1.3, label=f"p95: {age_p95:.1f} ms")
        ax_age.text(
            0.98,
            0.82,
            f"median {age_med:.1f} ms",
            transform=ax_age.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
    ax_age.set_title("Policy action age")
    ax_age.set_xlim(0.0, 100.0)
    ax_age.set_xlabel("Action age [ms]")
    ax_age.set_ylabel("Density")
    ax_age.grid(axis="y", alpha=0.3)
    ax_age.legend(loc="lower right", fontsize=7.5, frameon=True, handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    track_p95 = float("nan")
    track_med = float("nan")
    track_source = "n/a"
    if tracking_loaded is not None:
        npz = tracking_loaded.data
        eps = norm_rows(get_array(npz, ["epsilon_ur5e", "epsilon_follower"]))
        flag = get_array(npz, ["intervention_active", "is_correction"])
        n = eps.size
        t = phase_b_time_axis(npz, n)
        track_med = float(np.nanmedian(eps))
        track_p95 = float(np.nanpercentile(eps, 95))
        if flag is not None:
            shade_active_regions(ax_track, t, np.asarray(flag, dtype=float).reshape(-1)[:n] > 0.5)
        ax_track.plot(t, eps, color="tab:blue", linewidth=1.35, label=r"$||\epsilon_{\mathrm{UR5e}}||_2$")
        ax_track.axhline(track_p95, color="black", linestyle="--", linewidth=1.3, label=f"p95: {track_p95:.3f} rad")
        ax_track.text(
            0.98,
            0.92,
            f"median {track_med:.3f} rad\np95 {track_p95:.3f} rad",
            transform=ax_track.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
        track_source = str(tracking_loaded.path)
    ax_track.set_title("Follower tracking during cable routing")
    ax_track.set_xlabel("Time [s]")
    ax_track.set_ylim(bottom=0.0)
    ax_track.set_ylabel(r"$||\epsilon_{\mathrm{UR5e}}||_2$ [rad]")
    ax_track.grid(alpha=0.3)
    ax_track.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.22)
    save_fig(fig, out_dir, "K4_phase_b_timing", formats, dpi)

    if loop_ms.size:
        print(
            "[metrics] phase_b_loop_dt_ms "
            f"median={np.nanmedian(loop_ms):.3f} p95={np.nanpercentile(loop_ms, 95):.3f} "
            f"max={np.nanmax(loop_ms):.3f} n={loop_ms.size}"
        )
    if age_ms.size:
        print(
            "[metrics] policy_action_age_ms "
            f"median={np.nanmedian(age_ms):.3f} p95={np.nanpercentile(age_ms, 95):.3f} "
            f"max={np.nanmax(age_ms):.3f} n={age_ms.size} excluded_ge_500ms={age_outliers}"
        )
    if np.isfinite(track_p95):
        print(f"[metrics] representative_epsilon_ur5e_rad median={track_med:.4f} p95={track_p95:.4f}")
        print(f"[source] representative_tracking={track_source}")
    print(f"[source] phase-b timing files={len(loop_rows)}")


def phase_b_tracking_histogram_files(data_root: Path, input_npz: Sequence[Path] = ()) -> list[Path]:
    if input_npz:
        return expand_input_npz(input_npz)
    folders = [
        data_root / "cr_dagger_npz" / "phaseB_cable_routing_w_intervention_Bota3DoF",
        data_root / "cr_dagger_npz" / "phaseB_cable_routing_w_intervention_Bota6DoF",
        data_root / "cr_dagger_npz" / "phaseB_cable_routing_w_intervention_Bota6DoF_2nd",
    ]
    files: list[Path] = []
    for folder in folders:
        if folder.exists():
            files.extend(sorted(p for p in folder.glob("*.npz") if p.is_file()))
    return files


def plot_phase_b_tracking_histogram(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    title_override: str = "",
) -> None:
    files = phase_b_tracking_histogram_files(data_root, input_npz)
    rows = []
    durations = []
    gains = set()
    used_files = []
    for path in files:
        loaded = load_npz(path)
        if loaded is None:
            continue
        eps = as_2d(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]), 6)
        if eps is None:
            continue
        eps_norm = np.linalg.norm(eps[:, :6], axis=1)
        eps_norm = eps_norm[np.isfinite(eps_norm)]
        if eps_norm.size == 0:
            continue
        rows.append(eps_norm)
        used_files.append(path)
        n = min(eps.shape[0], eps_norm.size)
        t = phase_b_time_axis(loaded.data, n)
        if t.size:
            durations.append(float(t[-1]))
        meta = get_metadata(loaded.data)
        kp = meta.get("follower_kp", meta.get("ur5e_kp", meta.get("kp_follower", "?")))
        kd = meta.get("follower_kd", meta.get("ur5e_kd", meta.get("kd_follower", "?")))
        gains.add((str(kp), str(kd)))

    if not rows:
        print("[skip] phase-b tracking histogram: no epsilon_ur5e arrays found")
        return

    eps_all = np.concatenate(rows)
    median = float(np.nanmedian(eps_all))
    p95 = float(np.nanpercentile(eps_all, 95))
    p99 = float(np.nanpercentile(eps_all, 99))
    peak = float(np.nanmax(eps_all))
    delta_max = 0.08
    total_s = float(np.nansum(durations))

    x_max = max(0.12, min(0.18, max(p99 * 1.25, delta_max * 1.35)))
    bins = np.linspace(0.0, x_max, 55)
    tail_count = int(np.sum(eps_all > x_max))

    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    ax.hist(
        eps_all,
        bins=bins,
        density=True,
        color="#4c78a8",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        label=r"$||\epsilon_{\mathrm{UR5e}}||_2$",
    )
    ax.axvline(median, color="0.15", linestyle="-", linewidth=1.05, label=f"median {median:.3f} rad")
    ax.axvline(p95, color="#1f77b4", linestyle="--", linewidth=1.15, label=f"p95 {p95:.3f} rad")
    ax.axvline(delta_max, color="tab:red", linestyle="--", linewidth=1.15, label=r"$\delta_{\max}=0.08$ rad")
    ax.set_title(title_override or "Pooled UR5e Follower Tracking Error", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(r"$||\epsilon_{\mathrm{UR5e}}||_2$ [rad]")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0.0, x_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    note = f"{len(used_files)} runs, {total_s:.0f} s, n={eps_all.size}"
    if tail_count:
        note += f"\n{tail_count} samples > {x_max:.2f} rad"
    ax.text(
        0.03,
        0.95,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75"},
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.17)
    save_fig(fig, out_dir, "K4_phase_b_tracking_error_histogram", formats, dpi)

    print(
        "[metrics] phase_b_tracking_error_histogram "
        f"runs={len(used_files)} duration_s={total_s:.2f} samples={eps_all.size} "
        f"median={median:.5f}rad p95={p95:.5f}rad p99={p99:.5f}rad max={peak:.5f}rad "
        f"tail_gt_xmax={tail_count} x_max={x_max:.3f} gains={sorted(gains)}"
    )
    roots = sorted({str(path.parent.relative_to(data_root)) for path in used_files})
    for root_name in roots:
        count = sum(1 for path in used_files if str(path.parent.relative_to(data_root)) == root_name)
        print(f"[source] phase_b_tracking_error_histogram {root_name}: {count} files")


def _phase_b_timing_arrays(data_root: Path, max_files: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    loop_rows = []
    age_rows = []
    timing_file_count = 0
    for path in phase_b_timing_files(data_root, max_files):
        loaded = load_npz(path)
        if loaded is None:
            continue
        loop = get_array(loaded.data, ["phase_b_loop_dt_s"])
        age = get_array(loaded.data, ["policy_action_age_s"])
        if loop is not None:
            loop_rows.append(np.asarray(loop, dtype=float).reshape(-1))
            timing_file_count += 1
        if age is not None:
            age_rows.append(np.asarray(age, dtype=float).reshape(-1))

    loop_raw = np.concatenate(loop_rows) if loop_rows else np.array([])
    age_raw = np.concatenate(age_rows) if age_rows else np.array([])
    loop_all = loop_raw[np.isfinite(loop_raw) & (loop_raw > 0.0) & (loop_raw < 0.1)]
    age_valid = age_raw[np.isfinite(age_raw) & (age_raw >= 0.0)]
    age_all = age_valid[age_valid < 0.5]
    age_outliers = int(age_valid.size - age_all.size)
    return loop_all * 1000.0, age_all * 1000.0, age_outliers, timing_file_count


def _pooled_tracking_error(data_root: Path) -> tuple[np.ndarray, list[Path], float, set[tuple[str, str]]]:
    rows = []
    durations = []
    gains = set()
    used_files = []
    for path in phase_b_tracking_histogram_files(data_root):
        loaded = load_npz(path)
        if loaded is None:
            continue
        eps = as_2d(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]), 6)
        if eps is None:
            continue
        eps_norm = np.linalg.norm(eps[:, :6], axis=1)
        eps_norm = eps_norm[np.isfinite(eps_norm)]
        if eps_norm.size == 0:
            continue
        rows.append(eps_norm)
        used_files.append(path)
        t = phase_b_time_axis(loaded.data, eps.shape[0])
        if t.size:
            durations.append(float(t[-1]))
        meta = get_metadata(loaded.data)
        kp = meta.get("follower_kp", meta.get("ur5e_kp", meta.get("kp_follower", "?")))
        kd = meta.get("follower_kd", meta.get("ur5e_kd", meta.get("kd_follower", "?")))
        gains.add((str(kp), str(kd)))
    if not rows:
        return np.array([]), [], 0.0, gains
    return np.concatenate(rows), used_files, float(np.nansum(durations)), gains


def plot_phase_b_timing_with_tracking_histogram(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    max_files: int,
    title_override: str = "",
) -> None:
    loop_ms, age_ms, age_outliers, timing_file_count = _phase_b_timing_arrays(data_root, max_files)
    eps_all, used_files, total_s, gains = _pooled_tracking_error(data_root)
    if not loop_ms.size and not age_ms.size and not eps_all.size:
        print("[skip] phase-b timing/tracking histogram: no usable arrays found")
        return

    fig = plt.figure(figsize=(11.0, 4.25))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.85], wspace=0.32)
    left = outer[0].subgridspec(2, 1, hspace=0.72)
    ax_loop = fig.add_subplot(left[0])
    ax_age = fig.add_subplot(left[1])
    ax_track = fig.add_subplot(outer[1])
    title = title_override or "Phase-B Timing and Follower Tracking"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    if loop_ms.size:
        loop_med = float(np.nanmedian(loop_ms))
        loop_p95 = float(np.nanpercentile(loop_ms, 95))
        loop_std = float(np.nanstd(loop_ms))
        loop_max = float(np.nanmax(loop_ms))
        ax_loop.hist(loop_ms, bins=60, range=(3.0, 4.0), density=True, color="tab:blue", alpha=0.85)
        ax_loop.axvline(3.333, color="tab:red", linestyle="-", linewidth=1.6, label="Target 3.33 ms")
        ax_loop.axvline(loop_p95, color="black", linestyle="--", linewidth=1.3, label=f"p95: {loop_p95:.2f} ms")
        ax_loop.text(
            0.98,
            0.82,
            f"median {loop_med:.2f} ms\nstd {loop_std:.3f} ms\nmax {loop_max:.2f} ms",
            transform=ax_loop.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
    ax_loop.set_title("Loop cycle time")
    ax_loop.set_xlim(3.0, 4.0)
    ax_loop.set_xlabel("Cycle time [ms]")
    ax_loop.set_ylabel("Density")
    ax_loop.grid(axis="y", alpha=0.3)
    ax_loop.legend(loc="lower right", fontsize=7.5, frameon=True, handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    if age_ms.size:
        age_med = float(np.nanmedian(age_ms))
        age_p95 = float(np.nanpercentile(age_ms, 95))
        ax_age.hist(age_ms, bins=80, range=(0.0, 100.0), density=True, color="tab:orange", alpha=0.85)
        ax_age.axvline(50.0, color="tab:red", linestyle="-", linewidth=1.6, label="50 ms interval")
        ax_age.axvline(age_p95, color="black", linestyle="--", linewidth=1.3, label=f"p95: {age_p95:.1f} ms")
        ax_age.text(
            0.98,
            0.82,
            f"median {age_med:.1f} ms",
            transform=ax_age.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
    ax_age.set_title("Policy action age")
    ax_age.set_xlim(0.0, 100.0)
    ax_age.set_xlabel("Action age [ms]")
    ax_age.set_ylabel("Density")
    ax_age.grid(axis="y", alpha=0.3)
    ax_age.legend(loc="lower right", fontsize=7.5, frameon=True, handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    track_med = float("nan")
    track_p95 = float("nan")
    if eps_all.size:
        track_med = float(np.nanmedian(eps_all))
        track_p95 = float(np.nanpercentile(eps_all, 95))
        track_p99 = float(np.nanpercentile(eps_all, 99))
        track_max = float(np.nanmax(eps_all))
        delta_max = 0.08
        x_max = max(0.12, min(0.18, max(track_p99 * 1.25, delta_max * 1.35)))
        bins = np.linspace(0.0, x_max, 55)
        tail_count = int(np.sum(eps_all > x_max))
        ax_track.hist(
            eps_all,
            bins=bins,
            density=True,
            color="#4c78a8",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            label=r"$||\epsilon_{\mathrm{UR5e}}||_2$",
        )
        ax_track.axvline(track_med, color="0.15", linestyle="-", linewidth=1.05, label=f"median {track_med:.3f} rad")
        ax_track.axvline(track_p95, color="#1f77b4", linestyle="--", linewidth=1.15, label=f"p95 {track_p95:.3f} rad")
        ax_track.axvline(delta_max, color="tab:red", linestyle="--", linewidth=1.15, label=r"$\delta_{\max}=0.08$ rad")
        note = f"{len(used_files)} runs, {total_s:.0f} s, n={eps_all.size}"
        if tail_count:
            note += f"\n{tail_count} samples > {x_max:.2f} rad"
        ax_track.text(
            0.03,
            0.95,
            note,
            transform=ax_track.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75"},
        )
        print(
            "[metrics] pooled_epsilon_ur5e_rad "
            f"median={track_med:.5f} p95={track_p95:.5f} p99={track_p99:.5f} max={track_max:.5f} "
            f"runs={len(used_files)} duration_s={total_s:.2f} samples={eps_all.size} gains={sorted(gains)}"
        )
    ax_track.set_title("Follower tracking error")
    ax_track.set_xlabel(r"$||\epsilon_{\mathrm{UR5e}}||_2$ [rad]")
    ax_track.set_ylabel("Density")
    ax_track.set_xlim(0.0, 0.12)
    ax_track.grid(axis="y", alpha=0.3)
    ax_track.legend(loc="upper right", frameon=True, fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.22)
    save_fig(fig, out_dir, "K4_phase_b_timing_tracking_histogram", formats, dpi)

    if loop_ms.size:
        print(
            "[metrics] phase_b_loop_dt_ms "
            f"median={np.nanmedian(loop_ms):.3f} p95={np.nanpercentile(loop_ms, 95):.3f} "
            f"max={np.nanmax(loop_ms):.3f} n={loop_ms.size}"
        )
    if age_ms.size:
        print(
            "[metrics] policy_action_age_ms "
            f"median={np.nanmedian(age_ms):.3f} p95={np.nanpercentile(age_ms, 95):.3f} "
            f"max={np.nanmax(age_ms):.3f} n={age_ms.size} excluded_ge_500ms={age_outliers}"
        )
    print(f"[source] phase-b timing files={timing_file_count}")


def static_hold_correction_runs(data_root: Path) -> tuple[LoadedNpz | None, LoadedNpz | None]:
    with_path = (
        data_root
        / "cr_dagger_npz"
        / "20260529_phaseB_calibrated_bota_static_hold_interventions"
        / "crdagger_phaseB_dummy_hold_20260529_phaseB_calibrated_bota_static_hold_interventions_1917045_ep0000.npz"
    )
    without_path = (
        data_root
        / "cr_dagger_npz"
        / "20260529_phaseB_calibrated_bota_static_hold_no_interventions"
        / "crdagger_phaseB_dummy_hold_20260529_phaseB_calibrated_bota_static_hold_no_interventions_1916897_ep0000.npz"
    )
    with_run = load_npz(with_path) if with_path.exists() else None
    without_run = load_npz(without_path) if without_path.exists() else None
    if with_run is None:
        candidates = find_npz(data_root / "cr_dagger_npz", include=["static_hold_interventions"])
        with_run = first_usable(candidates, ["wrench_base", "delta_q", "intervention_active"])
    if without_run is None:
        candidates = find_npz(data_root / "cr_dagger_npz", include=["static_hold_no_interventions"])
        without_run = first_usable(candidates, ["wrench_base", "delta_q"])
    return with_run, without_run


def correction_signal_arrays(loaded: LoadedNpz) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = loaded.data
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench_base_calibrated", "bota_wrench_conditioned", "wrench"]), 6)
    delta = as_2d(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
    if wrench is None or delta is None:
        raise ValueError(f"missing wrench or delta in {loaded.path}")
    n = min(wrench.shape[0], delta.shape[0])
    t = phase_b_time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    delta_inf = np.nanmax(np.abs(delta[:n, :6]), axis=1)
    flag = get_array(npz, ["intervention_active", "is_correction"])
    if flag is None or flag.size < n:
        active = np.zeros(n, dtype=bool)
    else:
        active = np.asarray(flag[:n], dtype=float).reshape(-1) > 0.5
    return t, force, delta_inf, active


def plot_correction_signal_generation(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str = "",
) -> None:
    with_run, without_run = static_hold_correction_runs(data_root)
    if with_run is None or without_run is None:
        print("[skip] correction signal: static-hold with/without intervention runs not found")
        return

    runs = [
        ("With human intervention", with_run),
        ("Without human intervention", without_run),
    ]
    series = []
    for label, loaded in runs:
        try:
            t, force, delta_inf, active = correction_signal_arrays(loaded)
        except ValueError as exc:
            print(f"[skip] correction signal: {exc}")
            return
        series.append((label, loaded.path, t, force, delta_inf, active))

    force_max = max(float(np.nanmax(force)) for _, _, _, force, _, _ in series)
    delta_max = 0.08
    force_ylim = max(0.25, force_max * 1.12)
    delta_ylim = max(0.09, delta_max * 1.15)
    time_xlim = max(float(np.nanmax(t)) for _, _, t, _, _, _ in series)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharex=False)
    title = title_override or "Phase-B Correction Signal Generation"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.96)

    metrics = []
    for ax, (label, path, t, force, delta_inf, active) in zip(axes, series):
        ax_delta = ax.twinx()
        shade_active_regions(ax, t, active, label="Human correction")
        force_line = ax.plot(t, force, color="tab:blue", linewidth=1.35, label=r"$||F_{\mathrm{BOTA}}||_2$")
        delta_line = ax_delta.plot(t, delta_inf, color="tab:orange", linewidth=1.35, label=r"$||\Delta q||_{\infty}$")
        force_p95 = float(np.nanpercentile(force, 95))
        delta_p95 = float(np.nanpercentile(delta_inf, 95))
        force_peak = float(np.nanmax(force))
        delta_peak = float(np.nanmax(delta_inf))
        saturated_samples = int(np.sum(delta_inf >= 0.078))
        ax.axhline(force_p95, color="tab:blue", linestyle="--", linewidth=1.0, label=f"p95 F: {force_p95:.2f} N")
        ax_delta.axhline(delta_p95, color="tab:orange", linestyle="--", linewidth=1.0, label=f"p95 Δq: {delta_p95:.3f} rad")
        ax_delta.axhline(delta_max, color="tab:red", linestyle="--", linewidth=1.25, label=r"$\delta_{\max}=0.08$ rad")
        if saturated_samples:
            note = f"saturation samples: {saturated_samples}\nmax {delta_peak:.3f} rad"
        else:
            note = f"max F {force_peak:.3f} N\nmax Δq {delta_peak:.3f} rad"
        ax.text(
            0.98,
            0.72,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
        )
        ax.set_title(label)
        ax.set_xlim(0.0, time_xlim)
        ax.set_ylim(0.0, force_ylim)
        ax_delta.set_ylim(0.0, delta_ylim)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"$||F_{\mathrm{BOTA}}||_2$ [N]", color="tab:blue")
        ax_delta.set_ylabel(r"$||\Delta q||_{\infty}$ [rad]", color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax_delta.tick_params(axis="y", labelcolor="tab:orange")
        ax.grid(alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_delta.get_legend_handles_labels()
        ax.legend(
            lines + lines2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            fontsize=8,
            frameon=True,
        )
        metrics.append((label, path, force_p95, force_peak, delta_p95, delta_peak, saturated_samples))

    fig.subplots_adjust(left=0.075, right=0.92, top=0.84, bottom=0.27, wspace=0.42)
    save_fig(fig, out_dir, "K4_phase_b_correction_signal", formats, dpi)

    for label, path, force_p95, force_peak, delta_p95, delta_peak, saturated_samples in metrics:
        print(
            f"[metrics] {label}: F_p95={force_p95:.3f}N F_max={force_peak:.3f}N "
            f"delta_inf_p95={delta_p95:.4f}rad delta_inf_max={delta_peak:.4f}rad "
            f"saturated_samples={saturated_samples}"
        )
        print(f"[source] {label}: {path}")


def intervention_detection_arrays(loaded: LoadedNpz) -> dict[str, np.ndarray | float | Path]:
    npz = loaded.data
    wrench = as_2d(get_array(npz, ["wrench_base", "wrench_base_calibrated", "bota_wrench_conditioned", "wrench"]), 6)
    delta = as_2d(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
    if wrench is None or delta is None:
        raise ValueError(f"missing wrench or delta in {loaded.path}")
    n = min(wrench.shape[0], delta.shape[0])

    q_ref = as_2d(get_array(npz, ["q_ref", "q_ref_follower"]), 6)
    q_cmd_leader = as_2d(get_array(npz, ["q_cmd_leader", "q_cmd_leader_raw"]), 6)
    q_leader = as_2d(get_array(npz, ["q_leader", "q_actual"]), 6)
    q_cmd_ur5e = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower", "q_compliant"]), 6)
    q_follower = as_2d(get_array(npz, ["q_follower", "q_actual_follower"]), 6)
    for arr in (q_ref, q_cmd_leader, q_leader, q_cmd_ur5e, q_follower):
        if arr is not None:
            n = min(n, arr.shape[0])

    t = phase_b_time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    delta = delta[:n, :6]
    delta_inf = np.nanmax(np.abs(delta), axis=1)
    delta_l2 = np.linalg.norm(delta, axis=1)

    meta = get_metadata(npz)
    p_on = float(meta.get("contact_threshold", meta.get("intervention_contact_threshold", 0.25)) or 0.25)
    p_rel = float(meta.get("adm_delta_release_contact_threshold", 0.05) or 0.05)
    delta_thr = float(meta.get("delta_q_threshold", meta.get("intervention_delta_threshold", 0.02)) or 0.02)
    delta_max = float(meta.get("adm_delta_max", 0.08) or 0.08)

    contact_probability = get_array(npz, ["contact_probability"])
    if contact_probability is None or contact_probability.size < n:
        torque = np.linalg.norm(wrench[:n, 3:6], axis=1)
        contact_probability = np.clip(np.maximum(force / 8.0, torque / 0.35), 0.0, 1.0)
    else:
        contact_probability = np.asarray(contact_probability[:n], dtype=float).reshape(-1)

    votes = get_array(npz, ["detector_votes"])
    if votes is not None and votes.ndim == 2 and votes.shape[0] >= n and votes.shape[1] >= 4:
        detector_votes = np.asarray(votes[:n], dtype=float) > 0.5
        delta_vote = detector_votes[:, 1]
        contact_vote = detector_votes[:, 3] | (contact_probability >= p_on)
    else:
        source = get_array(npz, ["intervention_source"])
        if source is not None and source.size >= n:
            source = np.asarray(source[:n], dtype=float).reshape(-1)
            contact_vote = (source == 1.0) | (source == 3.0) | (contact_probability >= p_on)
            delta_vote = (source == 2.0) | (source == 3.0) | (delta_l2 >= delta_thr)
        else:
            contact_vote = contact_probability >= p_on
            delta_vote = delta_l2 >= delta_thr

    flag = get_array(npz, ["intervention_active", "is_correction"])
    if flag is None or flag.size < n:
        active = contact_vote | delta_vote
    else:
        active = np.asarray(flag[:n], dtype=float).reshape(-1) > 0.5

    out: dict[str, np.ndarray | float | Path] = {
        "path": loaded.path,
        "t": t,
        "force": force,
        "contact_probability": contact_probability,
        "contact_vote": contact_vote.astype(float),
        "delta_vote": delta_vote.astype(float),
        "active": active.astype(float),
        "delta": delta,
        "delta_inf": delta_inf,
        "delta_l2": delta_l2,
        "p_on": p_on,
        "p_rel": p_rel,
        "delta_thr": delta_thr,
        "delta_max": delta_max,
    }
    if q_ref is not None:
        out["q_ref"] = q_ref[:n, :6]
    if q_cmd_leader is not None:
        out["q_cmd_leader"] = q_cmd_leader[:n, :6]
    if q_leader is not None:
        out["q_leader"] = q_leader[:n, :6]
    if q_cmd_ur5e is not None:
        out["q_cmd_ur5e"] = q_cmd_ur5e[:n, :6]
    if q_follower is not None:
        out["q_follower"] = q_follower[:n, :6]
    return out


def _default_intervention_detection_run(data_root: Path, input_npz: Sequence[Path]) -> LoadedNpz | None:
    paths = expand_input_npz(input_npz) if input_npz else []
    if paths:
        return first_usable(paths, ["wrench_base", "delta_q", "contact_probability"])
    with_run, _ = static_hold_correction_runs(data_root)
    return with_run


def _best_release_window(t: np.ndarray, contact_probability: np.ndarray, delta_inf: np.ndarray, p_rel: float) -> tuple[float, float]:
    release = (contact_probability < p_rel) & (delta_inf > 0.01)
    idx = np.flatnonzero(release)
    if idx.size == 0:
        peak = int(np.nanargmax(delta_inf))
        center = float(t[peak])
        return max(float(t[0]), center - 2.0), min(float(t[-1]), center + 2.5)
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    p_on_nominal = max(0.25, 3.0 * float(p_rel))
    best = (starts[0], ends[0], -np.inf)
    for start, end in zip(starts, ends):
        drop = float(delta_inf[start] - delta_inf[end])
        dur = float(t[end] - t[start])
        pre = (t >= t[start] - 2.0) & (t <= t[start])
        peak_contact = float(np.nanmax(contact_probability[pre])) if np.any(pre) else 0.0
        score = drop + 0.04 * dur + (peak_contact if peak_contact >= p_on_nominal else 0.0)
        if score > best[2]:
            best = (start, end, score)
    start_i, end_i, _ = best
    return max(float(t[0]), float(t[start_i]) - 1.3), min(float(t[-1]), float(t[end_i]) + 0.7)


def plot_intervention_detection_pipeline(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    input_npz: Sequence[Path] = (),
    title_override: str = "",
) -> None:
    loaded = _default_intervention_detection_run(data_root, input_npz)
    if loaded is None:
        print("[skip] intervention detection: no usable static-hold intervention run found")
        return
    try:
        arrays = intervention_detection_arrays(loaded)
    except ValueError as exc:
        print(f"[skip] intervention detection: {exc}")
        return

    t = arrays["t"]
    force = arrays["force"]
    contact_probability = arrays["contact_probability"]
    contact_vote = arrays["contact_vote"]
    delta_vote = arrays["delta_vote"]
    active = arrays["active"]
    p_on = float(arrays["p_on"])
    p_rel = float(arrays["p_rel"])
    delta_inf = arrays["delta_inf"]
    plot_t_end = min(float(t[-1]), 33.0) if not input_npz else float(t[-1])

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 6.8), sharex=True)
    fig.suptitle(
        title_override or "Intervention Detection Pipeline",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )

    axes[0].plot(t, force, color="tab:blue", linewidth=1.25)
    axes[0].set_ylabel("Force [N]")

    axes[1].plot(t, contact_probability, color="tab:purple", linewidth=1.25, label=r"$p_{\mathrm{contact}}$")
    axes[1].axhline(p_on, color="tab:red", linestyle="--", linewidth=1.0, label=f"activation {p_on:.2f}")
    axes[1].axhline(p_rel, color="0.25", linestyle=":", linewidth=1.2, label=f"release {p_rel:.2f}")
    axes[1].set_ylabel(r"$p_{\mathrm{contact}}$ [-]")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8, ncol=3)

    axes[2].step(t, contact_vote + 0.04, where="post", color="tab:orange", linewidth=1.25, label=r"$I_{\mathrm{contact}}$")
    axes[2].step(t, delta_vote - 0.04, where="post", color="tab:green", linewidth=1.25, label=r"$I_{\Delta q}$")
    axes[2].set_ylabel("Vote [0/1]")
    axes[2].set_ylim(-0.15, 1.2)
    axes[2].legend(loc="upper left", frameon=True, fontsize=8, ncol=2)

    axes[3].step(t, active, where="post", color="0.2", linewidth=1.35, label=r"$I_{\mathrm{intervention}}$")
    axes[3].fill_between(t, 0.0, active, step="post", color="0.82", alpha=0.65)
    axes[3].set_ylabel("Flag [0/1]")
    axes[3].set_ylim(-0.1, 1.15)
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper left", frameon=True, fontsize=8)

    for ax in axes:
        ax.set_xlim(float(t[0]), plot_t_end)
        ax.grid(alpha=0.3)
    axes[0].text(
        0.01,
        0.88,
        rf"p95 $F$={np.nanpercentile(force, 95):.2f} N, p95 $||\Delta q||_\infty$={np.nanpercentile(delta_inf, 95):.3f} rad",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75"},
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.085, hspace=0.22)
    stem_suffix = ""
    if input_npz:
        match = re.search(r"_(\d+_ep\d+)$", loaded.path.stem)
        short_id = match.group(1) if match else loaded.path.stem[-32:]
        if "cable_routing" in str(loaded.path).lower():
            stem_suffix = f"_cable_routing_{short_id}"
        else:
            stem_suffix = f"_{short_id}"
    save_fig(fig, out_dir, f"K3_intervention_detection_pipeline{stem_suffix}", formats, dpi)

    # Close-up: same run, one joint with the strongest correction in the release window.
    if not all(key in arrays for key in ("q_ref", "q_cmd_leader", "q_cmd_ur5e", "q_follower")):
        print("[warn] intervention close-up skipped: missing command/follower joint streams")
        return

    delta = arrays["delta"]
    is_cable_input = bool(input_npz) and "cable_routing" in str(loaded.path).lower()

    def make_closeup(start_s: float, end_s: float, joint: int | None, suffix_extra: str) -> tuple[str, float, float, float, float] | None:
        mask = (t >= start_s) & (t <= end_s)
        if np.count_nonzero(mask) < 5:
            print(f"[warn] intervention close-up skipped: window {start_s:.2f}-{end_s:.2f}s too short")
            return None
        selected_joint = (
            int(joint)
            if joint is not None
            else int(np.nanargmax(np.nanmax(np.abs(delta[mask, :6]), axis=0)))
        )
        tt = t[mask]
        t0 = float(tt[0])
        tt = tt - t0

        q_ref = arrays["q_ref"][mask, selected_joint]
        q_cmd_leader = arrays["q_cmd_leader"][mask, selected_joint]
        q_leader = arrays["q_leader"][mask, selected_joint] if "q_leader" in arrays else None
        q_cmd_ur5e = arrays["q_cmd_ur5e"][mask, selected_joint]
        q_follower = arrays["q_follower"][mask, selected_joint]
        delta_joint = delta[mask, selected_joint]
        cp = contact_probability[mask]

        def rel(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=float) - float(np.asarray(x, dtype=float)[0])

        leader_rel = rel(q_cmd_leader)
        ur_cmd_rel = rel(q_cmd_ur5e)
        finite = np.isfinite(leader_rel) & np.isfinite(ur_cmd_rel)
        leader_sign = 1.0
        if np.count_nonzero(finite) > 3:
            same = float(np.nanmean(np.abs(leader_rel[finite] - ur_cmd_rel[finite])))
            flipped = float(np.nanmean(np.abs((-leader_rel[finite]) - ur_cmd_rel[finite])))
            if flipped < same:
                leader_sign = -1.0
        leader_measured_rel = None
        if q_leader is not None:
            leader_measured_rel = leader_sign * rel(q_leader)

        fig2, axes2 = plt.subplots(2, 1, figsize=(8.7, 4.8), sharex=True)
        fig2.suptitle(
            "Smooth Onset and Release Dynamics",
            fontsize=13,
            fontweight="bold",
            y=0.965,
        )
        axes2[0].plot(tt, rel(q_ref), color="0.1", linestyle="--", linewidth=1.2, label="Policy reference")
        axes2[0].plot(tt, ur_cmd_rel, color="tab:orange", linewidth=1.35, label="UR5e command")
        axes2[0].plot(
            tt,
            leader_sign * leader_rel,
            color="0.45",
            linestyle="--",
            linewidth=1.25,
            label="Leader command (mapped)",
        )
        if leader_measured_rel is not None:
            axes2[0].plot(
                tt,
                leader_measured_rel,
                color="tab:green",
                linewidth=1.1,
                alpha=0.95,
                label="Leader measured (mapped)",
            )
        axes2[0].plot(tt, rel(q_follower), color="tab:blue", linewidth=1.2, label="UR5e measured")
        axes2[0].axhline(0.0, color="0.72", linewidth=0.75, zorder=0)
        axes2[0].set_ylabel("Relative joint position [rad]")
        axes2[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fontsize=8)

        axes2[1].plot(tt, delta_joint, color="tab:orange", linewidth=1.45, label=rf"$\Delta q_{{{JOINT_LABELS[selected_joint]}}}$")
        axes2[1].axhline(0.0, color="0.72", linewidth=0.75, zorder=0)
        axes2[1].axhline(
            float(arrays["delta_max"]),
            color="tab:red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            label=r"$\pm\delta_{\max}$",
        )
        axes2[1].axhline(
            -float(arrays["delta_max"]),
            color="tab:red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            label="_nolegend_",
        )
        ax_cp = axes2[1].twinx()
        ax_cp.plot(tt, cp, color="tab:purple", linewidth=0.9, alpha=0.48, label=r"$p_{\mathrm{contact}}$")
        ax_cp.axhline(p_rel, color="0.25", linestyle=":", linewidth=0.8, alpha=0.85, label=f"release {p_rel:.2f}")

        if is_cable_input:
            event_lines: list[tuple[float, str, str]] = []
            idx_contact = np.flatnonzero(cp >= p_rel)
            if idx_contact.size:
                event_lines.append((float(tt[idx_contact[0]]), "contact", "0.45"))
            idx_activation = np.flatnonzero(cp >= p_on)
            if idx_activation.size:
                event_lines.append((float(tt[idx_activation[0]]), "activation", "tab:red"))
            sat_level = 0.975 * float(arrays["delta_max"])
            idx_sat = np.flatnonzero(np.abs(delta_joint) >= sat_level)
            if idx_sat.size:
                event_lines.append((float(tt[idx_sat[0]]), "saturation", "tab:orange"))
            if idx_activation.size:
                after = np.arange(cp.size) > int(idx_activation[-1])
                idx_release = np.flatnonzero(after & (cp < p_rel))
                if idx_release.size:
                    event_lines.append((float(tt[idx_release[0]]), "release", "0.25"))
            y_top = axes2[0].get_ylim()[1]
            for x_event, label, color in event_lines:
                for ax in axes2:
                    ax.axvline(x_event, color=color, linestyle=":", linewidth=1.25, alpha=0.8)
                is_saturation = label == "saturation"
                axes2[0].text(
                    x_event - 0.025 if is_saturation else x_event + 0.025,
                    y_top,
                    label,
                    rotation=90,
                    ha="right" if is_saturation else "left",
                    va="top",
                    fontsize=7.5,
                    color=color,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.6},
                )

        axes2[1].set_ylabel(r"$\Delta q$ [rad]")
        ax_cp.set_ylabel(r"$p_{\mathrm{contact}}$ [-]", color="tab:purple")
        ax_cp.tick_params(axis="y", labelcolor="tab:purple")
        axes2[1].set_xlabel("Time [s]")
        axes2[1].set_ylim(-float(arrays["delta_max"]) * 1.3, float(arrays["delta_max"]) * 1.3)
        ax_cp.set_ylim(0.0, max(0.32, float(np.nanmax(cp)) * 1.2))
        lines, labels = axes2[1].get_legend_handles_labels()
        lines2, labels2 = ax_cp.get_legend_handles_labels()
        axes2[1].legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=True, fontsize=8)

        for ax in axes2:
            ax.grid(alpha=0.3)
        fig2.subplots_adjust(left=0.095, right=0.905, top=0.86, bottom=0.22, hspace=0.55)
        save_fig(fig2, out_dir, f"K3_smooth_onset_release_closeup{stem_suffix}{suffix_extra}", formats, dpi)
        return (
            JOINT_LABELS[selected_joint],
            start_s,
            end_s,
            float(np.nanmin(delta_joint)),
            float(np.nanmax(delta_joint)),
        )

    closeup_metrics: list[tuple[str, float, float, float, float]] = []
    if is_cable_input:
        for start_s, end_s, suffix_extra in ((11.0, 15.0, ""), (16.0, 19.5, "_16_19p5")):
            metric = make_closeup(start_s, end_s, 1, suffix_extra)
            if metric is not None:
                closeup_metrics.append(metric)
    else:
        start_s, end_s = _best_release_window(t, contact_probability, delta_inf, p_rel)
        metric = make_closeup(start_s, end_s, None, "")
        if metric is not None:
            closeup_metrics.append(metric)

    print(
        "[metrics] intervention_detection "
        f"F_p95={np.nanpercentile(force, 95):.3f}N "
        f"p_contact_max={np.nanmax(contact_probability):.3f} "
        f"delta_inf_p95={np.nanpercentile(delta_inf, 95):.4f}rad "
        f"active_ratio={np.nanmean(active):.3f}"
    )
    for joint_label, start_s, end_s, delta_min, delta_max_value in closeup_metrics:
        print(
            "[metrics] release_closeup "
            f"joint={joint_label} window_abs={start_s:.2f}-{end_s:.2f}s "
            f"delta_min={delta_min:.4f}rad delta_max={delta_max_value:.4f}rad"
        )
    print(f"[source] intervention_detection={loaded.path}")


def representative_whiteboard_run(data_root: Path) -> LoadedNpz | None:
    preferred = (
        data_root
        / "cr_dagger_npz"
        / "20260528_phaseB_whiteboard_whiping_interventions_30mmBehind"
        / "crdagger_phaseB_lerobot_act_20260528_phaseB_whiteboard_whiping_interventions_30mmBehind_1838291_ep0000.npz"
    )
    if preferred.exists():
        loaded = load_npz(preferred)
        if loaded is not None:
            return loaded

    candidates = find_npz(
        data_root / "cr_dagger_npz",
        include=["phaseb_whiteboard_whiping_interventions_30mmbehind"],
    )
    best = None
    best_score = -float("inf")
    for path in candidates:
        loaded = load_npz(path)
        if loaded is None:
            continue
        force = norm_rows(get_array(loaded.data, ["wrench_base", "bota_wrench_conditioned", "wrench"]))
        delta = as_2d(get_array(loaded.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
        eps = norm_rows(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]))
        flag = get_array(loaded.data, ["intervention_active", "is_correction"])
        if force is None or delta is None or eps is None or flag is None:
            continue
        delta_inf = np.nanmax(np.abs(delta[:, :6]), axis=1)
        active_ratio = float(np.nanmean(np.asarray(flag, dtype=float).reshape(-1)[: delta_inf.size] > 0.5))
        score = float(np.nanpercentile(force, 95)) + 2.0 * float(np.nanpercentile(delta_inf, 95)) + active_ratio
        if score > best_score:
            best = loaded
            best_score = score
    return best


def whiteboard_summary(data_root: Path) -> list[tuple[str, int, float, float, float]]:
    folders = {
        "30mm behind": "20260528_phaseB_whiteboard_whiping_interventions_30mmBehind",
        "40mm behind": "20260528_phaseB_whiteboard_whiping_interventions_40mmBehind",
        "40mm in front": "20260528_phaseB_whiteboard_whiping_interventions_40mmInFront",
    }
    rows = []
    for label, folder in folders.items():
        delta_p95_values = []
        active_ratios = []
        total_s = 0.0
        run_count = 0
        for path in sorted((data_root / "cr_dagger_npz" / folder).glob("*.npz")):
            loaded = load_npz(path)
            if loaded is None:
                continue
            delta = as_2d(get_array(loaded.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
            flag = get_array(loaded.data, ["intervention_active", "is_correction"])
            if delta is None:
                continue
            n = delta.shape[0]
            t = phase_b_time_axis(loaded.data, n)
            total_s += float(t[-1]) if t.size else 0.0
            delta_inf = np.nanmax(np.abs(delta[:, :6]), axis=1)
            delta_p95_values.append(float(np.nanpercentile(delta_inf, 95)))
            if flag is not None and flag.size >= n:
                active_ratios.append(float(np.nanmean(np.asarray(flag[:n], dtype=float).reshape(-1) > 0.5)))
            run_count += 1
        if run_count:
            rows.append(
                (
                    label,
                    run_count,
                    total_s,
                    float(np.nanmedian(delta_p95_values)) if delta_p95_values else float("nan"),
                    float(np.nanmedian(active_ratios)) if active_ratios else float("nan"),
                )
            )
    return rows


def plot_whiteboard_task_run(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str = "",
) -> None:
    loaded = representative_whiteboard_run(data_root)
    if loaded is None:
        print("[skip] whiteboard task: no representative 30mmBehind run found")
        return

    npz = loaded.data
    wrench = as_2d(get_array(npz, ["wrench_base", "bota_wrench_conditioned", "wrench", "bota_wrench_base"]), 6)
    delta = as_2d(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
    eps = norm_rows(get_array(npz, ["epsilon_ur5e", "epsilon_follower"]))
    flag = get_array(npz, ["intervention_active", "is_correction"])
    if wrench is None or delta is None or eps is None:
        print(f"[skip] whiteboard task: missing wrench/delta/epsilon in {loaded.path}")
        return

    n = min(wrench.shape[0], delta.shape[0], eps.size)
    t = phase_b_time_axis(npz, n)
    force = np.linalg.norm(wrench[:n, :3], axis=1)
    delta_inf = np.nanmax(np.abs(delta[:n, :6]), axis=1)
    eps = eps[:n]
    active = np.asarray(flag[:n], dtype=float).reshape(-1) > 0.5 if flag is not None and flag.size >= n else np.zeros(n, dtype=bool)

    force_p95 = float(np.nanpercentile(force, 95))
    delta_p95 = float(np.nanpercentile(delta_inf, 95))
    delta_max = 0.08

    run_metrics = []
    folder = data_root / "cr_dagger_npz" / "20260528_phaseB_whiteboard_whiping_interventions_30mmBehind"
    for path in sorted(folder.glob("*.npz")):
        run = load_npz(path)
        if run is None:
            continue
        run_delta = as_2d(get_array(run.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
        run_flag = get_array(run.data, ["intervention_active", "is_correction"])
        if run_delta is None:
            continue
        run_delta_inf = np.nanmax(np.abs(run_delta[:, :6]), axis=1)
        run_n = run_delta_inf.size
        run_t = phase_b_time_axis(run.data, run_n)
        run_duration = float(run_t[-1]) if run_t.size else 0.0
        run_active_ratio = (
            float(np.nanmean(np.asarray(run_flag[:run_n], dtype=float).reshape(-1) > 0.5))
            if run_flag is not None and run_flag.size >= run_n
            else float("nan")
        )
        run_metrics.append(
            {
                "path": path,
                "duration": run_duration,
                "delta_p95": float(np.nanpercentile(run_delta_inf, 95)),
                "delta_max": float(np.nanmax(run_delta_inf)),
                "active_ratio": run_active_ratio,
            }
        )

    delta_p95_all = np.asarray([row["delta_p95"] for row in run_metrics], dtype=float)
    active_ratio_all = np.asarray([row["active_ratio"] for row in run_metrics], dtype=float)
    total_duration = float(np.nansum([row["duration"] for row in run_metrics]))

    fig = plt.figure(figsize=(10.8, 4.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.34)
    left = outer[0].subgridspec(2, 1, hspace=0.28)
    ax_force = fig.add_subplot(left[0])
    ax_delta = fig.add_subplot(left[1], sharex=ax_force)
    ax_box = fig.add_subplot(outer[1])
    axes = [ax_force, ax_delta]
    title = title_override or "Whiteboard Wiping: Phase-B Task Run"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    for ax in axes:
        shade_active_regions(ax, t, active, label="Human correction")
        ax.grid(alpha=0.3)

    ax_force.plot(t, force, color="tab:blue", linewidth=1.25, label=r"$||F_{\mathrm{BOTA}}||_2$")
    ax_force.axhline(force_p95, color="tab:blue", linestyle="--", linewidth=1.0, label=f"p95: {force_p95:.2f} N")
    ax_force.set_ylabel(r"$||F_{\mathrm{BOTA}}||_2$ [N]")
    ax_force.set_title("Representative 30 mm behind run")
    ax_force.tick_params(labelbottom=False)

    ax_delta.plot(t, delta_inf, color="tab:orange", linewidth=1.25, label=r"$||\Delta q||_{\infty}$")
    ax_delta.axhline(delta_max, color="tab:red", linestyle="--", linewidth=1.2, label=r"$\delta_{\max}=0.08$ rad")
    ax_delta.axhline(delta_p95, color="tab:orange", linestyle="--", linewidth=1.0, label=f"p95: {delta_p95:.3f} rad")
    ax_delta.set_ylim(0.0, max(0.09, float(np.nanmax(delta_inf)) * 1.15))
    ax_delta.set_ylabel(r"$||\Delta q||_{\infty}$ [rad]")
    ax_delta.set_xlabel("Time [s]")

    if delta_p95_all.size:
        box = ax_box.boxplot(
            [delta_p95_all],
            widths=0.35,
            patch_artist=True,
            showfliers=False,
            tick_labels=["30 mm behind"],
        )
        box["boxes"][0].set_facecolor("0.88")
        box["boxes"][0].set_edgecolor("0.25")
        x = np.ones(delta_p95_all.size)
        jitter = np.linspace(-0.08, 0.08, delta_p95_all.size) if delta_p95_all.size > 1 else np.zeros(1)
        ax_box.scatter(x + jitter, delta_p95_all, color="tab:orange", edgecolor="black", linewidth=0.4, zorder=3, label="Run p95")
    ax_box.axhline(delta_max, color="tab:red", linestyle="--", linewidth=1.2, label=r"$\delta_{\max}=0.08$ rad")
    ax_box.set_ylim(0.0, max(0.09, float(np.nanmax(delta_p95_all)) * 1.2 if delta_p95_all.size else 0.09))
    ax_box.set_ylabel(r"p95 $||\Delta q||_{\infty}$ [rad]")
    ax_box.set_title("Run-to-run boundedness")
    ax_box.grid(axis="y", alpha=0.3)

    handles, labels = [], []
    for ax in [ax_force, ax_delta, ax_box]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ax_delta.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, -0.30), ncol=3, frameon=True)
    ax_box.text(
        0.5,
        0.12,
        f"runs: {len(run_metrics)}\nrecording time: {total_duration:.1f} s\nmedian p95: {np.nanmedian(delta_p95_all):.3f} rad",
        transform=ax_box.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7"},
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.22)
    save_fig(fig, out_dir, "K4_whiteboard_wiping_task", formats, dpi)

    print(
        "[metrics] whiteboard representative "
        f"F_p95={force_p95:.3f}N F_max={np.nanmax(force):.3f}N "
        f"delta_inf_p95={delta_p95:.4f}rad delta_inf_max={np.nanmax(delta_inf):.4f}rad "
        f"epsilon_p95={np.nanpercentile(eps, 95):.4f}rad epsilon_max={np.nanmax(eps):.4f}rad "
        f"intervention_ratio={np.nanmean(active):.3f} duration_s={t[-1]:.2f}"
    )
    print(f"[source] whiteboard representative={loaded.path}")
    if delta_p95_all.size:
        print(
            "[metrics] whiteboard 30mmBehind runs "
            f"runs={len(run_metrics)} total_recording_s={total_duration:.1f} "
            f"delta_inf_p95_values={np.round(delta_p95_all, 4).tolist()} "
            f"median_delta_inf_p95={np.nanmedian(delta_p95_all):.4f} "
            f"median_intervention_ratio={np.nanmedian(active_ratio_all):.3f}"
        )
    for label, run_count, total_s, delta_p95_med, active_ratio_med in whiteboard_summary(data_root):
        print(
            f"[summary] {label}: runs={run_count} total_s={total_s:.1f} "
            f"median_delta_inf_p95={delta_p95_med:.4f} median_intervention_ratio={active_ratio_med:.3f}"
        )


def policy_debug_arrays(loaded: LoadedNpz):
    npz = loaded.data
    q_policy = as_2d(get_array(npz, ["q_ref_follower", "q_ref"]), 6)
    q_cmd_ur5e = as_2d(get_array(npz, ["q_cmd_ur5e", "q_cmd_follower", "q_compliant"]), 6)
    q_follower = as_2d(get_array(npz, ["q_follower", "q_actual"]), 6)
    q_leader = as_2d(get_array(npz, ["q_leader"]), 6)
    q_cmd_leader = as_2d(get_array(npz, ["q_cmd_leader", "q_ref_leader"]), 6)
    flag = get_array(npz, ["intervention_active", "is_correction"])
    required = [q_policy, q_cmd_ur5e, q_follower, q_leader, q_cmd_leader]
    if any(arr is None for arr in required):
        print(f"[skip] policy debug: missing joint arrays in {loaded.path}")
        return None
    n = min(arr.shape[0] for arr in required if arr is not None)
    t = phase_b_time_axis(npz, n)
    active = np.asarray(flag[:n], dtype=float).reshape(-1) > 0.5 if flag is not None and flag.size >= n else np.zeros(n, dtype=bool)
    return q_policy[:n], q_cmd_ur5e[:n], q_follower[:n], q_leader[:n], q_cmd_leader[:n], t, active


def policy_debug_paths(data_root: Path, input_npz: Sequence[Path]) -> list[Path]:
    if input_npz:
        return expand_input_npz(input_npz)
    rep = representative_whiteboard_run(data_root)
    return [rep.path] if rep is not None else []


def _relative_joint(y: np.ndarray, t_mask: np.ndarray, joint_index: int, sign: float = 1.0) -> np.ndarray:
    yy = float(sign) * np.asarray(y[:, joint_index][t_mask], dtype=float)
    return yy - float(yy[0])


def plot_policy_joint_zoom_loaded(
    loaded: LoadedNpz,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str,
    joint: int,
    t_start: float,
    t_end: float,
    invert_leader: bool,
) -> None:
    arrays = policy_debug_arrays(loaded)
    if arrays is None:
        return
    q_policy, q_cmd_ur5e, q_follower, q_leader, q_cmd_leader, t, active = arrays
    j = int(np.clip(joint, 1, 6)) - 1
    lo = float(min(t_start, t_end))
    hi = float(max(t_start, t_end))
    window = (t >= lo) & (t <= hi)
    if not np.any(window):
        print(f"[skip] policy joint debug: no samples in {lo:g}-{hi:g} s window for {loaded.path}")
        return

    leader_sign = -1.0 if invert_leader else 1.0
    leader_suffix = " (inverted)" if invert_leader else ""
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    inv = "inverted " if invert_leader else ""
    title = title_override or f"Policy, {inv}Leader, and Follower Zoom: J{j + 1}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.96)
    shade_active_regions(ax, t[window], active[window], label="Human correction")
    ax.plot(t[window], _relative_joint(q_policy, window, j), color="black", linestyle="--", linewidth=1.25, label="Policy reference")
    ax.plot(t[window], _relative_joint(q_cmd_ur5e, window, j), color="tab:orange", linewidth=1.2, label="UR5e command")
    ax.plot(t[window], _relative_joint(q_follower, window, j), color="tab:blue", linewidth=1.25, label="UR5e actual")
    ax.plot(t[window], _relative_joint(q_cmd_leader, window, j, leader_sign), color="0.45", linestyle="--", linewidth=1.05, label=f"Leader command{leader_suffix}")
    ax.plot(t[window], _relative_joint(q_leader, window, j, leader_sign), color="tab:green", linewidth=1.05, label=f"Leader actual{leader_suffix}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Relative J{j + 1} position [rad]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.26)
    stem = f"K4_policy_debug_J{j + 1}_{lo:g}_{hi:g}_{short_run_id(loaded.path)}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    save_fig(fig, out_dir, stem, formats, dpi)

    eps_ur5e = np.abs(q_follower[:, j][window] - q_cmd_ur5e[:, j][window])
    eps_leader = np.abs(q_leader[:, j][window] - q_cmd_leader[:, j][window])
    print(
        f"[metrics] policy joint debug J{j + 1} "
        f"ur5e_abs_error_p95={np.nanpercentile(eps_ur5e, 95):.4f}rad "
        f"leader_abs_error_p95={np.nanpercentile(eps_leader, 95):.4f}rad "
        f"window_s={lo:g}-{hi:g} samples={np.count_nonzero(window)}"
    )
    print(f"[source] policy joint debug={loaded.path}")


def plot_policy_debug_loaded(
    loaded: LoadedNpz,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str,
    invert_leader: bool,
    t_start: float,
    t_end: float,
) -> None:
    arrays = policy_debug_arrays(loaded)
    if arrays is None:
        return
    q_policy, q_cmd_ur5e, q_follower, q_leader, q_cmd_leader, t, active = arrays
    lo = float(min(t_start, t_end))
    hi = float(max(t_start, t_end))
    window = (t >= lo) & (t <= hi)
    if not np.any(window):
        print(f"[skip] policy all-joints debug: no samples in {lo:g}-{hi:g} s window for {loaded.path}")
        return
    leader_sign = -1.0 if invert_leader else 1.0
    leader_suffix = " (inverted)" if invert_leader else ""
    fig, axes = plt.subplots(6, 1, figsize=(11.0, 9.2), sharex=True)
    inv = "inverted " if invert_leader else ""
    title = title_override or f"Policy, {inv}Leader, and Follower Joint Traces"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    for j, ax in enumerate(axes):
        shade_active_regions(ax, t[window], active[window], label="Human correction")
        ax.plot(t[window], _relative_joint(q_policy, window, j), color="black", linestyle="--", linewidth=1.05, label="Policy reference")
        ax.plot(t[window], _relative_joint(q_cmd_ur5e, window, j), color="tab:orange", linewidth=1.05, label="UR5e command")
        ax.plot(t[window], _relative_joint(q_follower, window, j), color="tab:blue", linewidth=1.1, label="UR5e actual")
        ax.plot(t[window], _relative_joint(q_cmd_leader, window, j, leader_sign), color="0.45", linestyle="--", linewidth=0.9, label=f"Leader command{leader_suffix}")
        ax.plot(t[window], _relative_joint(q_leader, window, j, leader_sign), color="tab:green", linewidth=0.9, alpha=0.95, label=f"Leader actual{leader_suffix}")
        ax.set_ylabel(f"J{j + 1} [rad]")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(lo, hi)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[-1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.14, hspace=0.22)
    stem = f"K4_policy_debug_all_{lo:g}_{hi:g}_{short_run_id(loaded.path)}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    save_fig(fig, out_dir, stem, formats, dpi)
    print(f"[source] policy all-joints debug={loaded.path}")


def plot_whiteboard_policy_debug(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str = "",
    input_npz: Sequence[Path] = (),
    t_start: float = 0.0,
    t_end: float = 999999.0,
    invert_leader: bool = False,
) -> None:
    for path in policy_debug_paths(data_root, input_npz):
        loaded = load_npz(path)
        if loaded is not None:
            plot_policy_debug_loaded(loaded, out_dir, formats, dpi, title_override, invert_leader, t_start, t_end)


def plot_whiteboard_j3_debug(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str = "",
    input_npz: Sequence[Path] = (),
    joint: int = 3,
    t_start: float = 0.0,
    t_end: float = 20.0,
    invert_leader: bool = False,
    all_joints: bool = False,
) -> None:
    for path in policy_debug_paths(data_root, input_npz):
        loaded = load_npz(path)
        if loaded is None:
            continue
        if all_joints:
            plot_policy_debug_loaded(loaded, out_dir, formats, dpi, title_override, invert_leader, t_start, t_end)
        else:
            plot_policy_joint_zoom_loaded(loaded, out_dir, formats, dpi, title_override, joint, t_start, t_end, invert_leader)


def cable_routing_takeover_paths(data_root: Path, input_npz: Sequence[Path]) -> list[Path]:
    if input_npz:
        return expand_input_npz(input_npz)
    preferred = (
        data_root
        / "cr_dagger_npz"
        / "phaseB_cable_routing_w_intervention_Bota3DoF"
        / "crdagger_phaseB_lerobot_act_phaseB_cable_routing_w_intervention_Bota3DoF_1930021_ep0000.npz"
    )
    if preferred.exists():
        return [preferred]
    candidates = find_npz(
        data_root / "cr_dagger_npz",
        include=["phaseb_cable_routing_w_intervention_bota3dof"],
    )
    return candidates[:1]


def _leader_sign_for_joint(joint_index: int) -> float:
    return -1.0 if joint_index == 2 else 1.0


def plot_cable_routing_takeover_all_axes(
    loaded: LoadedNpz,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str,
    t_start: float,
    t_end: float,
) -> None:
    arrays = policy_debug_arrays(loaded)
    if arrays is None:
        return
    q_policy, q_cmd_ur5e, q_follower, q_leader, q_cmd_leader, t, active = arrays
    lo = float(min(t_start, t_end))
    hi = float(max(t_start, t_end))
    window = (t >= lo) & (t <= hi)
    if not np.any(window):
        print(f"[skip] cable routing all axes: no samples in {lo:g}-{hi:g} s window for {loaded.path}")
        return

    fig, axes = plt.subplots(6, 1, figsize=(11.2, 9.4), sharex=True)
    title = title_override or "Cable-Routing Smooth Takeover: Joint-Level View"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    for j, ax in enumerate(axes):
        leader_sign = _leader_sign_for_joint(j)
        shade_active_regions(ax, t[window], active[window], label="Human correction")
        ax.plot(t[window], _relative_joint(q_policy, window, j), color="black", linestyle="--", linewidth=1.0, label="Policy reference")
        ax.plot(t[window], _relative_joint(q_cmd_ur5e, window, j), color="tab:orange", linewidth=1.0, label="UR5e command")
        ax.plot(t[window], _relative_joint(q_follower, window, j), color="tab:blue", linewidth=1.05, label="UR5e actual")
        ax.plot(t[window], _relative_joint(q_cmd_leader, window, j, leader_sign), color="0.45", linestyle="--", linewidth=0.9, label="Leader command")
        ax.plot(t[window], _relative_joint(q_leader, window, j, leader_sign), color="tab:green", linewidth=0.9, alpha=0.95, label="Leader actual")
        ax.set_ylabel(f"J{j + 1} [rad]")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(lo, hi)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[-1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.14, hspace=0.22)
    stem = f"K4_cable_routing_takeover_appendix_all_axes_{lo:g}_{hi:g}_{short_run_id(loaded.path)}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    save_fig(fig, out_dir, stem, formats, dpi)


def plot_cable_routing_takeover_j2_j3(
    loaded: LoadedNpz,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str,
    t_start: float,
    t_end: float,
) -> None:
    arrays = policy_debug_arrays(loaded)
    if arrays is None:
        return
    q_policy, q_cmd_ur5e, q_follower, q_leader, q_cmd_leader, t, active = arrays
    npz = loaded.data
    delta = as_2d(get_array(npz, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]), 6)
    if delta is None:
        print(f"[skip] cable routing takeover: missing delta array in {loaded.path}")
        return
    n = min(q_policy.shape[0], delta.shape[0], t.size)
    q_policy = q_policy[:n]
    q_cmd_ur5e = q_cmd_ur5e[:n]
    q_follower = q_follower[:n]
    q_leader = q_leader[:n]
    q_cmd_leader = q_cmd_leader[:n]
    delta = delta[:n]
    t = t[:n]
    active = active[:n]

    lo = float(min(t_start, t_end))
    hi = float(max(t_start, t_end))
    window = (t >= lo) & (t <= hi)
    if not np.any(window):
        print(f"[skip] cable routing takeover: no samples in {lo:g}-{hi:g} s window for {loaded.path}")
        return

    delta_j2 = delta[:, 1]
    delta_j3 = delta[:, 2]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 0.75]})
    title = title_override or "Cable-Routing Smooth Takeover During Human Correction"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)

    for ax, j in zip(axes[:2], [1, 2]):
        leader_sign = _leader_sign_for_joint(j)
        shade_active_regions(ax, t[window], active[window], label="Human correction")
        ax.plot(t[window], _relative_joint(q_policy, window, j), color="black", linestyle="--", linewidth=1.25, label="Policy reference")
        ax.plot(t[window], _relative_joint(q_cmd_ur5e, window, j), color="tab:orange", linewidth=1.25, label="UR5e command")
        ax.plot(t[window], _relative_joint(q_follower, window, j), color="tab:blue", linewidth=1.1, label="UR5e actual")
        ax.plot(t[window], _relative_joint(q_cmd_leader, window, j, leader_sign), color="0.45", linestyle="--", linewidth=1.0, label="Leader command")
        ax.plot(t[window], _relative_joint(q_leader, window, j, leader_sign), color="tab:green", linewidth=1.0, label="Leader actual")
        ax.set_ylabel(f"Relative J{j + 1} [rad]")
        ax.grid(alpha=0.3)

    ax_delta = axes[2]
    shade_active_regions(ax_delta, t[window], active[window], label="Human correction")
    ax_delta.plot(t[window], delta_j2[window], color="tab:orange", linewidth=1.35, label=r"$\Delta q_{J2}$")
    ax_delta.plot(t[window], delta_j3[window], color="tab:purple", linewidth=1.25, label=r"$\Delta q_{J3}$")
    ax_delta.axhline(0.08, color="tab:red", linestyle="--", linewidth=1.05, label=r"$\pm\delta_{\max}=0.08$ rad")
    ax_delta.axhline(-0.08, color="tab:red", linestyle="--", linewidth=1.05)
    ax_delta.set_ylabel(r"$\Delta q$ [rad]")
    ax_delta.set_xlabel("Time [s]")
    ax_delta.set_xlim(lo, hi)
    delta_abs_window = np.nanmax(np.abs(np.column_stack([delta_j2[window], delta_j3[window]])))
    ax_delta.set_ylim(-max(0.09, float(delta_abs_window) * 1.15), max(0.09, float(delta_abs_window) * 1.15))
    ax_delta.grid(alpha=0.3)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[-1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.48), ncol=4, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.16, hspace=0.18)
    stem = f"K4_cable_routing_takeover_J2_J3_{lo:g}_{hi:g}_{short_run_id(loaded.path)}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    save_fig(fig, out_dir, stem, formats, dpi)

    print(
        f"[metrics] cable takeover J2/J3 "
        f"delta_j2_p95_abs={np.nanpercentile(np.abs(delta_j2[window]), 95):.4f}rad "
        f"delta_j2_max_abs={np.nanmax(np.abs(delta_j2[window])):.4f}rad "
        f"delta_j3_p95_abs={np.nanpercentile(np.abs(delta_j3[window]), 95):.4f}rad "
        f"delta_j3_max_abs={np.nanmax(np.abs(delta_j3[window])):.4f}rad "
        f"active_ratio={np.mean(active[window]):.3f} "
        f"window_s={lo:g}-{hi:g}"
    )


def plot_cable_routing_takeover(
    data_root: Path,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title_override: str = "",
    input_npz: Sequence[Path] = (),
    t_start: float = 10.0,
    t_end: float = 27.0,
) -> None:
    for path in cable_routing_takeover_paths(data_root, input_npz):
        loaded = load_npz(path)
        if loaded is None:
            continue
        plot_cable_routing_takeover_all_axes(loaded, out_dir, formats, dpi, "", t_start, t_end)
        plot_cable_routing_takeover_j2_j3(loaded, out_dir, formats, dpi, title_override, t_start, t_end)
        print(f"[source] cable routing takeover={loaded.path}")


def plot_whiteboard_offset_limitation(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int, max_files: int) -> None:
    groups = {
        "30mm behind": find_npz(data_root / "cr_dagger_npz", include=["30mmbehind"]),
        "40mm behind": find_npz(data_root / "cr_dagger_npz", include=["40mmbehind"]),
        "40mm in front": find_npz(data_root / "cr_dagger_npz", include=["40mminfront"]),
    }
    eps_groups: list[np.ndarray] = []
    labels: list[str] = []
    for label, paths in groups.items():
        rows = []
        for path in limit_files(paths, max_files):
            loaded = load_npz(path)
            if loaded is None:
                continue
            eps = norm_rows(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]))
            if eps is not None:
                rows.append(eps)
        if rows:
            labels.append(label)
            eps_groups.append(np.concatenate(rows))
    if not eps_groups:
        print("[skip] optional plot 9: no whiteboard offset epsilon rows found")
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.boxplot(eps_groups, tick_labels=labels, showfliers=False)
    ax.set_title("Offset robustness limitation")
    ax.set_ylabel("||epsilon_ur5e|| [rad]")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir, "10_whiteboard_offset_limitation", formats, dpi)


def plot_calibration_train_val_scatter(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    files = find_npz(data_root / "bota_minione_npz", include=["calibration_fit"])
    loaded = load_npz(files[0]) if files else None
    if loaded is None:
        print("[skip] optional plot 10: no calibration fit NPZ found")
        return
    npz = loaded.data
    calib_y = as_2d(get_array(npz, ["calib_y"]), 6)
    calib_pred = as_2d(get_array(npz, ["calib_pred"]), 6)
    val_y = as_2d(get_array(npz, ["val_y"]), 6)
    val_pred = as_2d(get_array(npz, ["val_pred"]), 6)
    if calib_y is None or calib_pred is None or val_y is None or val_pred is None:
        print("[skip] optional plot 10: missing train/val residual arrays")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, actual, pred, color in [
        ("train", calib_y, calib_pred, "tab:blue"),
        ("validation", val_y, val_pred, "tab:orange"),
    ]:
        actual_norm = np.linalg.norm(actual[:, :3], axis=1)
        pred_norm = np.linalg.norm(pred[:, :3], axis=1)
        idx = np.linspace(0, actual_norm.size - 1, min(actual_norm.size, 5000)).astype(int)
        ax.scatter(actual_norm[idx], pred_norm[idx], s=4, alpha=0.25, label=label, color=color)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("actual raw |F| [N]")
    ax.set_ylabel("predicted compensation |F| [N]")
    ax.set_title("Calibration prediction train/validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "11_calibration_train_val_scatter", formats, dpi)


def plot_leader_tracking(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int, max_files: int) -> None:
    files = limit_files(find_npz(data_root / "epsilon_measurements") + find_npz(data_root / "epsilon_measurements_calibrated_bota"), max_files)
    rows = []
    for path in files:
        loaded = load_npz(path)
        if loaded is None:
            continue
        eps = as_2d(get_array(loaded.data, ["epsilon_leader"]), 6)
        if eps is not None:
            rows.append(eps)
    if not rows:
        print("[skip] plot 9: no leader epsilon arrays found")
        return
    eps_all = np.concatenate(rows, axis=0)
    rms = rms_axis(eps_all)
    p95 = p95_abs_axis(eps_all)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(rms))
    ax.bar(x - 0.18, rms, width=0.36, label="RMS")
    ax.bar(x + 0.18, p95, width=0.36, label="p95 abs")
    ax.set_xticks(x, JOINT_LABELS[: len(rms)])
    ax.set_title("Leader mirror tracking")
    ax.set_ylabel("epsilon_leader [rad]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir, "09_leader_tracking_error", formats, dpi)


def plot_3dof_vs_6dof(data_root: Path, out_dir: Path, formats: Sequence[str], dpi: int) -> None:
    groups = {
        "BOTA 3DoF": find_npz(data_root / "cr_dagger_npz", include=["w_intervention_bota3dof"]),
        "BOTA 6DoF": find_npz(data_root / "cr_dagger_npz", include=["w_intervention_bota6dof"]),
    }
    metrics: dict[str, dict[str, list[float]]] = {
        label: {"delta_p95": [], "intervention_ratio": [], "epsilon_rms": [], "force_p95": []}
        for label in groups
    }
    for label, files in groups.items():
        for path in files:
            loaded = load_npz(path)
            if loaded is None:
                continue
            delta = norm_rows(get_array(loaded.data, ["delta_q", "delta_leader", "bota_delta_q_se3_limited"]))
            flag = get_array(loaded.data, ["intervention_active", "is_correction"])
            eps = norm_rows(get_array(loaded.data, ["epsilon_ur5e", "epsilon_follower"]))
            wrench = as_2d(get_array(loaded.data, ["wrench_base", "wrench", "bota_wrench_conditioned"]), 6)
            if delta is not None:
                metrics[label]["delta_p95"].append(float(np.nanpercentile(delta, 95)))
            if flag is not None and flag.size:
                metrics[label]["intervention_ratio"].append(float(np.nanmean(np.asarray(flag, dtype=float) > 0.5)))
            if eps is not None:
                metrics[label]["epsilon_rms"].append(float(np.sqrt(np.nanmean(eps * eps))))
            if wrench is not None:
                metrics[label]["force_p95"].append(float(np.nanpercentile(np.linalg.norm(wrench[:, :3], axis=1), 95)))
    if not any(metrics[label]["delta_p95"] for label in metrics):
        print("[skip] plot 10: no 3DoF/6DoF comparison data found")
        return

    labels = list(metrics.keys())
    fields = [
        ("delta_p95", "p95 ||delta_q|| [rad]"),
        ("intervention_ratio", "intervention ratio [-]"),
        ("epsilon_rms", "RMS ||epsilon_ur5e|| [rad]"),
        ("force_p95", "p95 ||F|| [N]"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6))
    for ax, (field, title) in zip(axes, fields):
        vals = [metrics[label][field] for label in labels]
        ax.boxplot(vals, tick_labels=labels, showfliers=False)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("3DoF vs. 6DoF BOTA Phase-B comparison")
    fig.tight_layout()
    save_fig(fig, out_dir, "09_bota_3dof_vs_6dof", formats, dpi)


def main() -> int:
    global GROUP_BY_PLOT, NO_SAVE, SHOW_FIGURES
    args = parse_args()
    GROUP_BY_PLOT = not bool(args.flat_output)
    SHOW_FIGURES = bool(args.show)
    NO_SAVE = bool(args.no_save)
    if NO_SAVE and not SHOW_FIGURES:
        print("[warn] --no-save was set without --show; figures will be built and discarded.")
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    if not data_root.exists():
        print(f"Data root not found: {data_root}", file=sys.stderr)
        return 1

    core = args.plots in ("core", "all")
    priority2 = args.plots in ("priority2", "all")
    thesis = args.plots in ("thesis", "all")
    minimum = args.plots == "minimum"
    optional = args.plots in ("optional", "all")
    impedance = args.plots == "impedance"
    impedance_appendix = args.plots == "impedance_appendix"
    observer_noise = args.plots == "observer_noise"
    observer_debug = args.plots == "observer_debug"
    observer_admittance = args.plots == "observer_admittance"
    observer_attempts = args.plots == "observer_attempts"
    bota_validation = args.plots == "bota_validation"
    bota_admittance = args.plots == "bota_admittance"
    phaseb_timing = args.plots == "phaseb_timing"
    phaseb_tracking_histogram = args.plots == "phaseb_tracking_histogram"
    phaseb_timing_tracking_histogram = args.plots == "phaseb_timing_tracking_histogram"
    correction_signal = args.plots == "correction_signal"
    intervention_detection = args.plots == "intervention_detection"
    whiteboard_task = args.plots == "whiteboard_task"
    whiteboard_policy_debug = args.plots == "whiteboard_policy_debug"
    whiteboard_j3_debug = args.plots == "whiteboard_j3_debug"
    cable_routing_takeover = args.plots == "cable_routing_takeover"

    if impedance:
        plot_impedance_sine_tracking_j1(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.aggregate_inputs,
            args.plot_title,
        )
        return 0

    if impedance_appendix:
        plot_impedance_appendix(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.plot_title,
        )
        return 0

    if observer_noise:
        plot_sensorless_observer_noise(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
        )
        return 0

    if observer_debug:
        if not args.no_save:
            NO_SAVE = True
            print("[debug] observer_debug defaults to --no-save")
        plot_observer_debug(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
        )
        return 0

    if observer_admittance:
        plot_observer_admittance_static_comparison(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.plot_title,
        )
        return 0

    if observer_attempts:
        plot_sensorless_observer_attempts(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.plot_title,
        )
        return 0

    if bota_validation:
        plot_bota_ft_sensor_validation(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
        )
        return 0

    if bota_admittance:
        plot_bota_admittance_response(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
        )
        return 0

    if phaseb_timing:
        plot_phase_b_timing(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.max_files,
            args.plot_title,
        )
        return 0

    if phaseb_tracking_histogram:
        plot_phase_b_tracking_histogram(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.plot_title,
        )
        return 0

    if phaseb_timing_tracking_histogram:
        plot_phase_b_timing_with_tracking_histogram(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.max_files,
            args.plot_title,
        )
        return 0

    if correction_signal:
        plot_correction_signal_generation(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
        )
        return 0

    if intervention_detection:
        plot_intervention_detection_pipeline(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.input_npz,
            args.plot_title,
        )
        return 0

    if whiteboard_task:
        plot_whiteboard_task_run(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
        )
        return 0

    if whiteboard_policy_debug:
        plot_whiteboard_policy_debug(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
            args.input_npz,
            args.debug_t_start,
            args.debug_t_end,
            args.invert_leader,
        )
        return 0

    if whiteboard_j3_debug:
        plot_whiteboard_j3_debug(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
            args.input_npz,
            args.debug_joint,
            args.debug_t_start,
            args.debug_t_end,
            args.invert_leader,
            args.debug_all_joints,
        )
        return 0

    if cable_routing_takeover:
        t_start = 10.0 if args.debug_t_start == 0.0 and args.debug_t_end == 20.0 else args.debug_t_start
        t_end = 27.0 if args.debug_t_start == 0.0 and args.debug_t_end == 20.0 else args.debug_t_end
        plot_cable_routing_takeover(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.plot_title,
            args.input_npz,
            t_start,
            t_end,
        )
        return 0

    if thesis or minimum:
        # Chapter 3: design narration evidence.
        plot_impedance_baseline(data_root, out_dir, args.formats, args.dpi)
        plot_observer_vs_bota_qualitative(data_root, out_dir, args.formats, args.dpi)
        plot_bota_raw_vs_calibrated(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            stem="C_bota_raw_vs_calibrated_pose_sweep",
        )
        plot_phase_b_timeline(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.timeline_pattern,
            stem="D_phase_b_static_hold_timeline",
        )

        # Chapter 4: minimum viable quantitative evaluation.
        plot_observer_vs_bota_snr(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.max_files,
            stem="01_snr_observer_vs_bota",
        )
        plot_intervention_zoom(data_root, out_dir, args.formats, args.dpi)
        plot_delta_q_histogram(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.max_files,
            stem="07_delta_q_histogram_final_phase_b",
        )
        plot_intervention_aligned_average(
            data_root,
            out_dir,
            args.formats,
            args.dpi,
            args.max_files,
            stem="08_intervention_aligned_average",
        )
        plot_cable_routing_successful_rollout(data_root, out_dir, args.formats, args.dpi)

    if thesis:
        plot_static_drift(data_root, out_dir, args.formats, args.dpi, stem="02_bota_static_check")
        plot_follower_tracking_error(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_leader_follower_sync(data_root, out_dir, args.formats, args.dpi)
        plot_latency_histogram(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_3dof_vs_6dof(data_root, out_dir, args.formats, args.dpi)
        plot_whiteboard_contact_divergence(data_root, out_dir, args.formats, args.dpi)

    if optional:
        plot_whiteboard_offset_limitation(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_calibration_train_val_scatter(data_root, out_dir, args.formats, args.dpi)

    if core:
        plot_observer_vs_bota_snr(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_phase_b_timeline(data_root, out_dir, args.formats, args.dpi, args.timeline_pattern)
        plot_bota_raw_vs_calibrated(data_root, out_dir, args.formats, args.dpi)
        plot_delta_q_histogram(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_follower_tracking_error(data_root, out_dir, args.formats, args.dpi, args.max_files)

    if priority2:
        plot_calibration_residual_pose_sweep(data_root, out_dir, args.formats, args.dpi)
        plot_static_drift(data_root, out_dir, args.formats, args.dpi)
        plot_intervention_aligned_average(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_leader_tracking(data_root, out_dir, args.formats, args.dpi, args.max_files)
        plot_3dof_vs_6dof(data_root, out_dir, args.formats, args.dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
