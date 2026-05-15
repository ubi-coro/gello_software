from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _parse_metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta_json = data.get("metadata_json")
    if meta_json is None:
        return {}
    try:
        if isinstance(meta_json, np.ndarray):
            meta_json = meta_json.item()
        return json.loads(meta_json)
    except Exception:
        return {}


def _epsilon_metrics(epsilon: np.ndarray, joint_index: int) -> dict[str, float]:
    eps_j = epsilon[:, joint_index]
    eps_norm = np.linalg.norm(epsilon, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(eps_j ** 2))),
        "max": float(np.max(np.abs(eps_j))),
        "mean": float(np.mean(eps_j)),
        "rms_norm": float(np.sqrt(np.mean(eps_norm ** 2))),
        "max_norm": float(np.max(eps_norm)),
    }


def _actual_array(data: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
    target = str(meta.get("target", "leader"))
    if target == "follower":
        return np.asarray(data["q_follower"], dtype=float)
    return np.asarray(data["q_leader"], dtype=float)


def _tracking_metrics(
    data: dict[str, Any],
    meta: dict[str, Any],
    joint_index: int,
) -> dict[str, float]:
    q_ref = np.asarray(data["q_ref"], dtype=float)
    q_actual = _actual_array(data, meta)
    q_ref_j = q_ref[:, joint_index]
    q_actual_j = q_actual[:, joint_index]
    ref_range = float(np.max(q_ref_j) - np.min(q_ref_j))
    actual_range = float(np.max(q_actual_j) - np.min(q_actual_j))
    tracking_gain = actual_range / ref_range if ref_range > 1e-9 else float("nan")
    return {
        "ref_rng": ref_range,
        "act_rng": actual_range,
        "track_gain": tracking_gain,
    }


def _format_meta_value(value: Any, joint_index: int | None = None) -> str:
    if value is None:
        return "-"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if joint_index is not None and 0 <= joint_index < len(value):
            try:
                return f"{float(value[joint_index]):.4g}"
            except (TypeError, ValueError):
                return str(value[joint_index])
        if len(value) == 0:
            return "[]"
        return "[" + ",".join(_format_meta_value(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _meta_column(meta: dict[str, Any], name: str, joint_index: int) -> str:
    if name == "kp_j":
        return _format_meta_value(meta.get("kp"), joint_index)
    if name == "kd_j":
        return _format_meta_value(meta.get("kd"), joint_index)
    if name == "amp_j":
        multi_amplitudes = meta.get("multi_amplitudes")
        if isinstance(multi_amplitudes, (list, tuple)) and len(multi_amplitudes) > 0:
            return _format_meta_value(multi_amplitudes, joint_index)
        return _format_meta_value(meta.get("amplitude"))
    if name == "freq_j":
        multi_frequencies = meta.get("multi_frequencies")
        if isinstance(multi_frequencies, (list, tuple)) and len(multi_frequencies) > 0:
            return _format_meta_value(multi_frequencies, joint_index)
        return _format_meta_value(meta.get("frequency"))
    return _format_meta_value(meta.get(name))


def _print_table(rows: list[dict[str, Any]]) -> None:
    metric_headers = [
        "rms",
        "max",
        "mean",
        "rms_norm",
        "max_norm",
        "ref_rng",
        "act_rng",
        "track_gain",
    ]
    meta_headers = [
        "mode",
        "architecture",
        "leader_command_mode",
        "test_mode",
        "joint_index",
        "amp_j",
        "freq_j",
        "kp_j",
        "kd_j",
        "admittance_observer",
        "admittance_inner_loop",
        "impedance_ramp_time",
        "settle_time",
    ]
    headers = ["file", *metric_headers, *meta_headers]
    col_widths = {h: max(len(h), 10) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))
    header_line = " ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        cells = [str(row["file"]).ljust(col_widths["file"])]
        for h in metric_headers:
            value = float(row[h])
            if np.isnan(value):
                cells.append("nan".rjust(col_widths[h]))
            else:
                cells.append(f"{value:+.5f}".rjust(col_widths[h]))
        for h in meta_headers:
            cells.append(str(row.get(h, "-")).ljust(col_widths[h]))
        line = " ".join(cells)
        print(line)


def _configure_thesis_style() -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.8,
            "grid.color": "0.86",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )
    return {
        "ref": "#1f77b4",
        "actual": "0.1",
        "epsilon": "#d95f02",
        "delta": "#7570b3",
        "tau": "#1b9e77",
        "contact": "#b2182b",
        "gray": "0.55",
    }


def _plot_single(data: dict[str, Any], meta: dict[str, Any], joint_index: int, save_dir: Path | None, no_show: bool) -> None:
    import matplotlib.pyplot as plt

    colors = _configure_thesis_style()
    t_mono = np.asarray(data["t_mono"], dtype=float)
    if t_mono.size == 0:
        print("No samples in file")
        return
    t = t_mono - t_mono[0]

    target = str(meta.get("target", "leader"))
    if target == "follower":
        q_actual = np.asarray(data["q_follower"], dtype=float)
        actual_label = "q_follower"
    else:
        q_actual = np.asarray(data["q_leader"], dtype=float)
        actual_label = "q_leader"

    q_ref = np.asarray(data["q_ref"], dtype=float)
    epsilon = np.asarray(data["epsilon"], dtype=float)
    delta_corr = np.asarray(data.get("delta_corr", np.zeros_like(epsilon)), dtype=float)
    tau_residual = np.asarray(data.get("tau_residual", np.zeros_like(epsilon)), dtype=float)
    contact_probability = np.asarray(
        data.get("contact_probability", np.zeros(t.shape[0])),
        dtype=float,
    )

    test_mode = str(meta.get("test_mode", ""))
    add_fft = test_mode == "chirp"
    n_plots = 6 if add_fft else 5

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(8.0, 1.85 * n_plots),
        sharex=not add_fft,
    )
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(t, q_ref[:, joint_index], label="reference", color=colors["ref"], lw=1.4)
    axes[0].plot(
        t,
        q_actual[:, joint_index],
        label=actual_label,
        color=colors["actual"],
        lw=1.2,
        ls="--",
    )
    axes[0].set_ylabel("rad")
    axes[0].set_title("Joint tracking")
    axes[0].legend(loc="best")

    axes[1].plot(t, epsilon[:, joint_index], color=colors["epsilon"], lw=1.2)
    axes[1].set_ylabel("rad")
    axes[1].set_title("Tracking error")

    eps_norm = np.linalg.norm(epsilon, axis=1)
    axes[2].plot(t, eps_norm, color=colors["epsilon"], lw=1.2)
    axes[2].set_ylabel("rad")
    axes[2].set_title("Tracking error norm")

    axes[3].plot(
        t,
        delta_corr[:, joint_index],
        color=colors["delta"],
        lw=1.2,
    )
    axes[3].set_ylabel("rad")
    axes[3].set_title("Compliant correction")

    ax_tau = axes[4]
    ax_tau.plot(
        t,
        tau_residual[:, joint_index],
        color=colors["tau"],
        lw=1.2,
        label="tau residual",
    )
    ax_tau.set_ylabel("Nm")
    ax_tau.set_title("Observer/contact signal")
    ax_prob = ax_tau.twinx()
    ax_prob.plot(
        t,
        contact_probability,
        color=colors["contact"],
        lw=1.0,
        ls=":",
        label="contact probability",
    )
    ax_prob.set_ylabel("prob.")
    ax_prob.set_ylim(-0.05, 1.05)

    if add_fft:
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        eps_j = epsilon[:, joint_index] - np.mean(epsilon[:, joint_index])
        if dt > 0:
            freqs = np.fft.rfftfreq(len(eps_j), d=dt)
            amps = np.abs(np.fft.rfft(eps_j))
            axes[5].plot(freqs, amps, color=colors["gray"], lw=1.2)
            axes[5].set_xlabel("Hz")
            axes[5].set_ylabel("|FFT|")
            axes[5].set_title("Tracking error spectrum")

    for ax in axes:
        ax.grid(True)
    if add_fft:
        axes[4].set_xlabel("time [s]")
    else:
        axes[-1].set_xlabel("time [s]")

    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{meta.get('mode', 'epsilon')}_{meta.get('test_mode', 'test')}"
        out_path = save_dir / f"epsilon_plot_{stem}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        print(f"Saved plot: {out_path}")

    if not no_show:
        plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze epsilon test logs")
    p.add_argument("path", type=str, help="NPZ file or directory")
    p.add_argument("--compare-all", action="store_true")
    p.add_argument("--joint-index", type=int, default=1)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.path)

    if args.compare_all:
        if not path.is_dir():
            print("compare-all requires a directory")
            return 1
        rows = []
        for file_path in sorted(path.glob("*.npz")):
            data = _load_npz(file_path)
            meta = _parse_metadata(data)
            metrics = _epsilon_metrics(np.asarray(data["epsilon"], dtype=float), int(args.joint_index))
            tracking_metrics = _tracking_metrics(data, meta, int(args.joint_index))
            row = {"file": file_path.name, **metrics, **tracking_metrics}
            for name in [
                "mode",
                "architecture",
                "leader_command_mode",
                "test_mode",
                "joint_index",
                "amp_j",
                "freq_j",
                "kp_j",
                "kd_j",
                "admittance_observer",
                "admittance_inner_loop",
                "impedance_ramp_time",
                "settle_time",
            ]:
                row[name] = _meta_column(meta, name, int(args.joint_index))
            rows.append(row)
        _print_table(rows)
        return 0

    if not path.exists():
        print(f"Not found: {path}")
        return 1

    data = _load_npz(path)
    meta = _parse_metadata(data)
    _plot_single(data, meta, int(args.joint_index), Path(args.save_dir) if args.save_dir else None, bool(args.no_show))

    metrics = {
        **_epsilon_metrics(np.asarray(data["epsilon"], dtype=float), int(args.joint_index)),
        **_tracking_metrics(data, meta, int(args.joint_index)),
    }
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
