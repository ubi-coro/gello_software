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


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["file", "rms", "max", "mean", "rms_norm", "max_norm"]
    col_widths = {h: max(len(h), 10) for h in headers}
    for row in rows:
        col_widths["file"] = max(col_widths["file"], len(row["file"]))
    header_line = " ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = " ".join(
            [
                row["file"].ljust(col_widths["file"]),
                f"{row['rms']:+.5f}".rjust(col_widths["rms"]),
                f"{row['max']:+.5f}".rjust(col_widths["max"]),
                f"{row['mean']:+.5f}".rjust(col_widths["mean"]),
                f"{row['rms_norm']:+.5f}".rjust(col_widths["rms_norm"]),
                f"{row['max_norm']:+.5f}".rjust(col_widths["max_norm"]),
            ]
        )
        print(line)


def _plot_single(data: dict[str, Any], meta: dict[str, Any], joint_index: int, save_dir: Path | None, no_show: bool) -> None:
    import matplotlib.pyplot as plt

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

    test_mode = str(meta.get("test_mode", ""))
    add_fft = test_mode == "chirp"
    n_plots = 4 if add_fft else 3

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(t, q_ref[:, joint_index], label="q_ref")
    axes[0].plot(t, q_actual[:, joint_index], label=actual_label)
    axes[0].set_ylabel("rad")
    axes[0].set_title("Joint tracking")
    axes[0].legend()

    axes[1].plot(t, epsilon[:, joint_index])
    axes[1].set_ylabel("rad")
    axes[1].set_title("epsilon (joint)")

    eps_norm = np.linalg.norm(epsilon, axis=1)
    axes[2].plot(t, eps_norm)
    axes[2].set_ylabel("rad")
    axes[2].set_title("|epsilon|")
    axes[2].set_xlabel("time [s]")

    if add_fft:
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        eps_j = epsilon[:, joint_index] - np.mean(epsilon[:, joint_index])
        if dt > 0:
            freqs = np.fft.rfftfreq(len(eps_j), d=dt)
            amps = np.abs(np.fft.rfft(eps_j))
            axes[3].plot(freqs, amps)
            axes[3].set_xlabel("Hz")
            axes[3].set_ylabel("|FFT|")
            axes[3].set_title("epsilon FFT")

    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = meta.get("test_mode", "epsilon")
        out_path = save_dir / f"epsilon_plot_{stem}.png"
        fig.savefig(out_path, dpi=150)
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
            metrics = _epsilon_metrics(np.asarray(data["epsilon"], dtype=float), int(args.joint_index))
            rows.append({"file": file_path.name, **metrics})
        _print_table(rows)
        return 0

    if not path.exists():
        print(f"Not found: {path}")
        return 1

    data = _load_npz(path)
    meta = _parse_metadata(data)
    _plot_single(data, meta, int(args.joint_index), Path(args.save_dir) if args.save_dir else None, bool(args.no_show))

    metrics = _epsilon_metrics(np.asarray(data["epsilon"], dtype=float), int(args.joint_index))
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
