import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro

from gello.dynamixel.driver import DynamixelDriver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MENAGERIE_ROOT: Path = Path(__file__).parent / "third_party" / "mujoco_menagerie"


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, -1, -1, 1, 1, 1)
    """The joint sign mapping from motor to URDF joint direction."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    baudrate: int = 4000000
    """The baudrate for the Dynamixel servos."""
    
    # Default servo types for UR5e GELLO
    servo_types: Tuple[str, ...] = (
        "XC330_T288_T", "XM430_W350_T", "XM430_W350_T",
        "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T"
    )
    """Servo types for torque-current mapping (7 = 6 arm + 1 gripper)."""

    snap_to_pi_over_2: bool = False
    """If true, output offsets snapped to multiples of pi/2 as primary result."""

    show_comparison: bool = True
    """If true, print comparison between best-fit and snapped offsets including errors."""

    print_yaml: bool = True
    """If true, print YAML-ready lines for initialization.joint_offsets and use_precomputed_offsets."""

    snap_base: str = "pi"
    """Snap base for permanent offsets: 'pi' or 'pi_over_2'."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."
        assert self.snap_base in ("pi", "pi_over_2"), "snap_base must be 'pi' or 'pi_over_2'."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def get_config(args: Args) -> None:
    joint_ids = list(range(1, args.num_joints + 1))
    servo_types = list(args.servo_types[:args.num_joints])
        
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=args.baudrate, servo_types=servo_types)

    # Calibration: find offsets that minimize error between current position and expected start_joints
    # Using fine-grained search (every ~5°) for better accuracy

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    print("\n" + "="*60)
    print("GELLO OFFSET CALIBRATION")
    print("="*60)
    print(f"Port: {args.port}")
    print(f"Expected pose: {[f'{np.rad2deg(x):.1f}°' for x in args.start_joints]}")
    print(f"Joint signs: {list(args.joint_signs)}")
    
    snap_step = np.pi if args.snap_base == "pi" else (np.pi / 2.0)
    snap_label = "pi" if args.snap_base == "pi" else "pi/2"

    for _ in range(1):
        best_offsets = []
        best_errors = []
        curr_joints = driver.get_joints()
        
        print(f"\nRaw motor positions: {[f'{np.rad2deg(x):.1f}°' for x in curr_joints[:args.num_robot_joints]]}")
        print("\nCalculating offsets (searching every ~5°):")
        print("-"*60)
        
        for i in range(args.num_robot_joints):
            best_offset = 0
            best_error = 1e6
            target = args.start_joints[i]
            sign = args.joint_signs[i]
            raw = curr_joints[i]
            
            # Fine-grained search: every ~5° instead of 90°
            for offset in np.linspace(-10 * np.pi, 10 * np.pi, 721):
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            
            best_offsets.append(best_offset)
            best_errors.append(best_error)
            calibrated = sign * (raw - best_offset)
            
            print(f"  Joint {i+1}: raw={np.rad2deg(raw):+7.1f}°, "
                  f"offset={np.rad2deg(best_offset):+7.1f}° ({best_offset/np.pi:.2f}π), "
                  f"result={np.rad2deg(calibrated):+7.1f}° (error={np.rad2deg(best_error):.2f}°)")
        
        print("-"*60)
        print("\nResults:")
        best_offsets_arr = np.asarray(best_offsets, dtype=float)
        best_errors_arr = np.asarray(best_errors, dtype=float)
        snapped_offsets_arr = np.round(best_offsets_arr / snap_step) * snap_step

        # Recompute errors for snapped offsets at current measured pose.
        snapped_errors_arr = np.zeros_like(best_errors_arr)
        for i in range(args.num_robot_joints):
            snapped_errors_arr[i] = get_error(float(snapped_offsets_arr[i]), i, curr_joints)

        selected_offsets = snapped_offsets_arr if args.snap_to_pi_over_2 else best_offsets_arr
        selected_name = f"SNAPPED ({snap_label})" if args.snap_to_pi_over_2 else "BEST-FIT"
        selected_errors = snapped_errors_arr if args.snap_to_pi_over_2 else best_errors_arr

        # If a joint has best_offset = permanent_offset + hold_error, this residual term
        # should be small when the permanent offset grid matches hardware reality.
        hold_error_arr = best_offsets_arr - snapped_offsets_arr

        print(f"  Selected mode: {selected_name}")
        print(f"  Offsets (rad): {[f'{x:.4f}' for x in selected_offsets]}")
        print(
            f"  Offsets ({snap_label}): ["
            + ", ".join([f"{int(np.round(x/snap_step))}*{snap_label}" for x in selected_offsets])
            + "]"
        )
        print(f"  Fit error (deg): {[f'{np.rad2deg(x):.2f}' for x in selected_errors]}")
        print(f"  Hold-error estimate (deg): {[f'{np.rad2deg(x):+.2f}' for x in hold_error_arr]}")

        if args.show_comparison:
            print("\nComparison:")
            print(f"  Best-fit offsets (rad): {[f'{x:.4f}' for x in best_offsets_arr]}")
            print(f"  Best-fit error (deg):  {[f'{np.rad2deg(x):.2f}' for x in best_errors_arr]}")
            print(f"  Snapped offsets (rad): {[f'{x:.4f}' for x in snapped_offsets_arr]}")
            print(f"  Snapped error (deg):   {[f'{np.rad2deg(x):.2f}' for x in snapped_errors_arr]}")
            delta_err = np.rad2deg(snapped_errors_arr - best_errors_arr)
            print(f"  Delta error (deg):     {[f'{x:+.2f}' for x in delta_err]}")
            print(f"  Snap base:             {snap_label}")

        if args.print_yaml:
            print("\nYAML snippet:")
            print(
                "  joint_offsets: ["
                + ", ".join([f"{float(x):.4f}" for x in selected_offsets])
                + "]"
            )
            print("  use_precomputed_offsets: true")
        
        if args.gripper:
            gripper_raw = driver.get_joints()[-1]
            print(f"\nGripper:")
            print(f"  Raw position: {np.rad2deg(gripper_raw):.1f}°")
            print(f"  Suggested open (deg):  {np.rad2deg(gripper_raw) - 0.2:.1f}")
            print(f"  Suggested close (deg): {np.rad2deg(gripper_raw) - 42:.1f}")
    
    driver.close()


def main(args: Args) -> None:
    get_config(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
