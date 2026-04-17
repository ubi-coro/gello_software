#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gello.dynamixel.driver import DynamixelDriver


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    baudrate: int = 4000000
    """The baudrate for the Dynamixel servos."""

    gripper: bool = True
    """Whether the gripper motor is attached (7th motor)."""

    num_arm_joints: int = 6
    """Number of arm joints (without gripper)."""

    servo_types: Tuple[str, ...] = (
        "XC330_T288_T",
        "XM430_W350_T",
        "XM430_W350_T",
        "XC330_T288_T",
        "XC330_T288_T",
        "XC330_T288_T",
        "XC330_T288_T",
    )
    """Servo types for the connected motors."""

    watch: bool = False
    """If true, continuously print current raw joint values."""

    hz: float = 2.0
    """Print/update rate in watch mode."""

    samples: int = 1
    """Number of samples to average per print."""

    def __post_init__(self) -> None:
        if self.num_arm_joints <= 0:
            raise ValueError("num_arm_joints must be > 0")
        if self.hz <= 0:
            raise ValueError("hz must be > 0")
        if self.samples <= 0:
            raise ValueError("samples must be > 0")

    @property
    def num_joints(self) -> int:
        return self.num_arm_joints + (1 if self.gripper else 0)


def _read_mean_joints(driver: DynamixelDriver, num_joints: int, samples: int) -> np.ndarray:
    vals = []
    for _ in range(samples):
        q = np.asarray(driver.get_joints()[:num_joints], dtype=float)
        vals.append(q)
    return np.mean(np.asarray(vals), axis=0)


def _print_raw(q_raw: np.ndarray) -> None:
    q_deg = np.rad2deg(q_raw)
    print(f"raw_rad: {[float(f'{x:.4f}') for x in q_raw]}")
    print(f"raw_deg: {[float(f'{x:.2f}') for x in q_deg]}")


def run(args: Args) -> None:
    joint_ids = list(range(1, args.num_joints + 1))
    servo_types = list(args.servo_types[: args.num_joints])

    driver = DynamixelDriver(
        joint_ids,
        port=args.port,
        baudrate=args.baudrate,
        servo_types=servo_types,
    )

    try:
        for _ in range(5):
            driver.get_joints()

        print("=" * 60)
        print("GELLO CURRENT RAW POSE")
        print("=" * 60)
        print(f"port: {args.port}")
        print(f"num_joints: {args.num_joints}")

        if not args.watch:
            q_raw = _read_mean_joints(driver, args.num_joints, args.samples)
            _print_raw(q_raw)
            return

        period = 1.0 / args.hz
        print(f"watch mode: {args.hz:.2f} Hz (samples={args.samples})")
        print("Press Ctrl+C to stop.")

        while True:
            t0 = time.monotonic()
            q_raw = _read_mean_joints(driver, args.num_joints, args.samples)
            _print_raw(q_raw)
            dt = time.monotonic() - t0
            sleep_s = period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()


if __name__ == "__main__":
    run(tyro.cli(Args))
