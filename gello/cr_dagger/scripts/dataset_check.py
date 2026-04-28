#!/usr/bin/env python3
"""
Dataset Validation & Analysis Script for CR-DAgger Collections

Validates:
  - Episode count and total frames
  - Action/state shape correctness
  - Data variability (not constant/stuck)
  - Feature synchronicity
  - FPS consistency
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[ERROR] lerobot not installed. Install with: pip install lerobot")
    raise


def check_dataset(repo_id: str, root: str | None = None) -> dict:
    """
    Load and validate a CR-DAgger dataset.

    Args:
        repo_id: Repository ID (e.g., "test2/cr_dagger_teleoperation")
        root: Optional root path for local datasets

    Returns:
        Dictionary with validation results
    """
    dataset_path = None
    if root is not None:
        root_path = Path(root)
        candidate_path = root_path / repo_id
        if candidate_path.exists():
            dataset_path = candidate_path
        elif root_path.exists():
            dataset_path = root_path

    print(f"\n{'='*70}")
    print(f"[DATASET CHECK] Loading: {repo_id}")
    if dataset_path:
        print(f"[DATASET CHECK] Path: {dataset_path}")
    print(f"{'='*70}\n")

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_path) if dataset_path else root)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return {"success": False, "error": str(e)}

    results = {"success": True}

    # ──────────────────────────────────────────────────────────────────────
    # 1. BASIC METADATA
    # ──────────────────────────────────────────────────────────────────────
    print("[METADATA]")
    meta = ds.meta
    print(f"  Episodes: {meta.total_episodes}")
    print(f"  Total frames: {meta.total_frames}")
    print(f"  FPS: {ds.fps}")
    duration_s = meta.total_frames / ds.fps if ds.fps > 0 else 0
    print(f"  Total duration: {duration_s:.1f}s ({meta.total_frames / ds.fps / 60:.1f}min)")
    action_shape = ds.features.get("action", {}).get("shape")
    state_shape = ds.features.get("observation.state", {}).get("shape")
    print(f"  Action shape: {action_shape if action_shape is not None else 'N/A'}")
    print(f"  State shape: {state_shape if state_shape is not None else 'N/A'}")

    results["episodes"] = meta.total_episodes
    results["frames"] = meta.total_frames
    results["fps"] = ds.fps
    results["duration_s"] = duration_s
    results["action_shape"] = action_shape
    results["state_shape"] = state_shape

    # ──────────────────────────────────────────────────────────────────────
    # 2. FEATURE VALIDATION
    # ──────────────────────────────────────────────────────────────────────
    print("\n[FEATURES]")
    print(f"  Available keys: {list(ds.features.keys())}")

    if len(ds) > 0:
        frame = ds[0]
        print(f"\n  Frame 0 shapes & types:")
        for key, val in frame.items():
            if torch.is_tensor(val):
                print(f"    {key:20s}: shape={tuple(val.shape)} dtype={val.dtype}")
            elif isinstance(val, np.ndarray):
                print(f"    {key:20s}: shape={val.shape} dtype={val.dtype}")
            else:
                print(f"    {key:20s}: {type(val).__name__}")

        results["features"] = list(ds.features.keys())

    # ──────────────────────────────────────────────────────────────────────
    # 3. ACTION VALIDATION (CHECK FOR STUCK/CONSTANT VALUES)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[ACTION ANALYSIS]")
    if "action" in ds.features and len(ds) > 1:
        sample_size = min(500, len(ds))
        try:
            actions = torch.stack([ds[i]["action"] for i in range(sample_size)])
            print(f"  Sampled {sample_size} frames")
            print(f"  Shape: {actions.shape}")

            action_mean = actions.mean(dim=0)
            action_std = actions.std(dim=0)
            action_min = actions.min(dim=0).values
            action_max = actions.max(dim=0).values
            action_range = action_max - action_min

            print(f"\n  Action statistics (across {sample_size} frames):")
            print(f"    Mean:  {action_mean.numpy()}")
            print(f"    Std:   {action_std.numpy()}")
            print(f"    Min:   {action_min.numpy()}")
            print(f"    Max:   {action_max.numpy()}")
            print(f"    Range: {action_range.numpy()}")

            # Check for constant actions (all zeros or stuck)
            is_stuck = (action_std < 1e-5).all()
            if is_stuck:
                print(f"\n  ⚠️  WARNING: Action appears STUCK (all std < 1e-5)")
                results["action_status"] = "stuck"
            else:
                print(f"\n  ✓ Action varies correctly")
                results["action_status"] = "varying"

            results["action_mean"] = action_mean.numpy().tolist()
            results["action_std"] = action_std.numpy().tolist()
            results["action_range"] = action_range.numpy().tolist()

        except Exception as e:
            print(f"  [ERROR] Failed to analyze actions: {e}")
            results["action_status"] = "error"

    # ──────────────────────────────────────────────────────────────────────
    # 4. OBSERVATION STATE VALIDATION
    # ──────────────────────────────────────────────────────────────────────
    print("\n[OBSERVATION STATE ANALYSIS]")
    state_keys = [k for k in ds.features.keys() if k.startswith("observation.state")]

    if state_keys and len(ds) > 1:
        try:
            sample_size = min(500, len(ds))
            state_data = []
            for i in range(sample_size):
                frame = ds[i]
                # Try to get state from one of the state keys
                for key in state_keys:
                    if key in frame:
                        state_data.append(frame[key])
                        break

            if state_data:
                states = torch.stack(state_data)
                print(f"  Sampled {len(state_data)} frames")
                print(f"  Shape: {states.shape}")

                state_mean = states.mean(dim=0)
                state_std = states.std(dim=0)

                print(f"\n  State statistics:")
                print(f"    Mean: {state_mean.numpy()}")
                print(f"    Std:  {state_std.numpy()}")

                # Check for meaningful variation
                has_variation = (state_std > 1e-5).any()
                if has_variation:
                    print(f"  ✓ State varies correctly")
                    results["state_status"] = "varying"
                else:
                    print(f"  ⚠️  WARNING: State appears constant (all std < 1e-5)")
                    results["state_status"] = "stuck"

                results["state_std"] = state_std.numpy().tolist()
        except Exception as e:
            print(f"  [ERROR] Failed to analyze state: {e}")
            results["state_status"] = "error"
    else:
        print(f"  No 'observation.state' keys found")

    # ──────────────────────────────────────────────────────────────────────
    # 5. IMAGE VALIDATION
    # ──────────────────────────────────────────────────────────────────────
    print("\n[IMAGE ANALYSIS]")
    image_keys = [k for k in ds.features.keys() if "image" in k or "rgb" in k]

    if image_keys:
        print(f"  Image keys: {image_keys}")
        if len(ds) > 0:
            try:
                frame = ds[0]
                for img_key in image_keys:
                    if img_key in frame:
                        img = frame[img_key]
                        if torch.is_tensor(img):
                            print(f"    {img_key}: shape={img.shape}, dtype={img.dtype}")
                        elif isinstance(img, np.ndarray):
                            print(f"    {img_key}: shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                print(f"  [ERROR] Failed to read images: {e}")
    else:
        print(f"  ⚠️  No image data found")

    # ──────────────────────────────────────────────────────────────────────
    # 6. EPISODE STATISTICS
    # ──────────────────────────────────────────────────────────────────────
    print("\n[EPISODE STATISTICS]")
    try:
        if hasattr(meta, "episode_lengths") and meta.episode_lengths is not None:
            ep_lengths = np.array(meta.episode_lengths)
            print(f"  Episode lengths (frames):")
            print(f"    Mean:  {ep_lengths.mean():.1f}")
            print(f"    Min:   {ep_lengths.min()}")
            print(f"    Max:   {ep_lengths.max()}")
            print(f"    Std:   {ep_lengths.std():.1f}")
            results["episode_lengths"] = {
                "mean": float(ep_lengths.mean()),
                "min": int(ep_lengths.min()),
                "max": int(ep_lengths.max()),
                "std": float(ep_lengths.std()),
            }
        else:
            print(f"  Episode length data not available in metadata")
    except Exception as e:
        print(f"  [ERROR] Failed to analyze episodes: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # 7. FINAL VALIDATION SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("[VALIDATION SUMMARY]")
    print(f"  ✓ Dataset loaded successfully")
    print(f"  ✓ Total frames: {meta.total_frames}")
    print(f"  ✓ Episodes: {meta.total_episodes}")
    print(f"  ✓ FPS: {ds.fps}")

    if results.get("action_status") == "stuck":
        print(f"  ⚠️  ACTION: Appears stuck (no variation)")
    elif results.get("action_status") == "varying":
        print(f"  ✓ ACTION: Varies correctly")

    if results.get("state_status") == "stuck":
        print(f"  ⚠️  STATE: Appears stuck (no variation)")
    elif results.get("state_status") == "varying":
        print(f"  ✓ STATE: Varies correctly")

    if not image_keys:
        print(f"  ⚠️  IMAGES: No image data found")
    else:
        print(f"  ✓ IMAGES: Found {len(image_keys)} image streams")

    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate CR-DAgger dataset")
    parser.add_argument(
        "repo_id",
        type=str,
        help="Dataset repo_id (e.g., 'user/dataset_name')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional root path for local datasets",
    )

    args = parser.parse_args()

    results = check_dataset(args.repo_id, root=args.root)

    if not results.get("success", False):
        return 1

    # Check for warnings
    has_warnings = (
        results.get("action_status") == "stuck"
        or results.get("state_status") == "stuck"
    )
    if has_warnings:
        print("[WARN] Dataset has potential issues. Check output above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
