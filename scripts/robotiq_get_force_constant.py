#!/usr/bin/env python3
"""Script to calibrate the force constant of a Robotiq 2F-85 gripper.

This script helps determine the relationship between the FOR setting (0-255)
and the actual grip force in Newtons by:
1. Commanding the gripper to close at different FOR settings
2. Recording when object detection triggers (motor current limit reached)
3. Allowing manual input of measured force from an external force gauge

From Robotiq manual:
- FOR setting 0-255 maps linearly to 20-235 N grip force
- The FOR setting defines maximum motor current during motion
- When current limit is exceeded, object detection triggers

Usage:
    1. Place a force gauge/load cell between the gripper fingers
    2. Run this script
    3. Enter the measured force for each FOR setting
    4. The script will calculate and display the force constant

Requirements:
    - Robotiq 2F-85 gripper connected via socket (default: 192.168.1.10:63352)
    - External force measurement device (optional, for calibration)
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, "/home/lennart/Python_Projects/gello_software")

from gello.robots.robotiq_gripper import RobotiqGripper


@dataclass
class ForceCalibrationPoint:
    """A single calibration data point."""
    for_setting: int          # FOR value (0-255)
    expected_force_N: float   # Expected force from Robotiq spec
    measured_force_N: float   # Actual measured force (0 if not measured)
    position_at_grip: int     # Position when object detected
    object_detected: bool     # Whether object was detected


class RobotiqForceCalibrator:
    """Calibrate force constant for Robotiq 2F-85 gripper."""

    # Robotiq 2F-85 specifications
    FOR_MIN = 0
    FOR_MAX = 255
    FORCE_MIN_N = 20.0   # Force at FOR=0 (from manual)
    FORCE_MAX_N = 235.0  # Force at FOR=255 (from manual)

    def __init__(self, hostname: str = "192.168.1.10", port: int = 63352):
        self.hostname = hostname
        self.port = port
        self.gripper: Optional[RobotiqGripper] = None
        self.calibration_data: List[ForceCalibrationPoint] = []

    def connect(self) -> bool:
        """Connect to the gripper."""
        try:
            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=self.hostname, port=self.port)
            print(f"Connected to gripper at {self.hostname}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the gripper."""
        if self.gripper:
            self.gripper.disconnect()
            print("Disconnected from gripper")

    def activate(self) -> bool:
        """Activate and calibrate the gripper."""
        if not self.gripper:
            return False
        try:
            print("Activating gripper (this will perform auto-calibration)...")
            self.gripper.activate(auto_calibrate=True)
            print(f"Gripper activated. Position range: [{self.gripper.get_min_position()}, {self.gripper.get_max_position()}]")
            return True
        except Exception as e:
            print(f"Activation failed: {e}")
            return False

    def expected_force(self, for_setting: int) -> float:
        """Calculate expected force in Newtons for a given FOR setting.
        
        Uses linear interpolation based on Robotiq specifications:
        FOR 0 -> 20 N, FOR 255 -> 235 N
        """
        force_range = self.FORCE_MAX_N - self.FORCE_MIN_N  # 215 N
        return self.FORCE_MIN_N + (for_setting / 255.0) * force_range

    def grip_at_force(self, for_setting: int, speed: int = 50) -> Tuple[bool, int]:
        """Command gripper to close with specified force setting.
        
        Args:
            for_setting: Force setting (0-255)
            speed: Speed setting (0-255), default slow for safety
            
        Returns:
            Tuple of (object_detected, final_position)
        """
        if not self.gripper:
            return False, 0

        # First open the gripper fully
        print(f"  Opening gripper...")
        self.gripper.move_and_wait_for_pos(
            self.gripper.get_open_position(), speed=100, force=50
        )
        time.sleep(0.5)

        # Now close with the specified force
        print(f"  Closing with FOR={for_setting} (expected {self.expected_force(for_setting):.1f} N)...")
        final_pos, status = self.gripper.move_and_wait_for_pos(
            self.gripper.get_closed_position(), speed=speed, force=for_setting
        )

        object_detected = status in (
            RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT,
            RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
        )

        print(f"  Result: position={final_pos}, status={status.name}, object_detected={object_detected}")
        return object_detected, final_pos

    def run_calibration_sweep(
        self,
        for_values: Optional[List[int]] = None,
        interactive: bool = True
    ) -> List[ForceCalibrationPoint]:
        """Run a calibration sweep across multiple FOR settings.
        
        Args:
            for_values: List of FOR values to test. Default: [0, 50, 100, 150, 200, 255]
            interactive: If True, prompt for measured force values
            
        Returns:
            List of calibration points
        """
        if for_values is None:
            for_values = [0, 50, 100, 150, 200, 255]

        self.calibration_data = []

        print("\n" + "=" * 60)
        print("ROBOTIQ 2F-85 FORCE CALIBRATION")
        print("=" * 60)
        print("\nInstructions:")
        print("1. Place a force gauge between the gripper fingers")
        print("2. The gripper will close at each FOR setting")
        print("3. Record the force shown on your gauge")
        print("4. Enter the measured force when prompted (or press Enter to skip)")
        print("\n" + "-" * 60)

        for for_setting in for_values:
            print(f"\n[FOR={for_setting}] Testing force setting...")
            
            object_detected, position = self.grip_at_force(for_setting)
            expected = self.expected_force(for_setting)

            measured = 0.0
            if interactive and object_detected:
                try:
                    user_input = input(f"  Enter measured force in N (expected ~{expected:.1f} N) [skip]: ").strip()
                    if user_input:
                        measured = float(user_input)
                except ValueError:
                    print("  Invalid input, skipping measurement")
                    measured = 0.0

            point = ForceCalibrationPoint(
                for_setting=for_setting,
                expected_force_N=expected,
                measured_force_N=measured,
                position_at_grip=position,
                object_detected=object_detected,
            )
            self.calibration_data.append(point)

        return self.calibration_data

    def calculate_force_constant(self) -> Tuple[float, float, float]:
        """Calculate force constant from calibration data.
        
        Uses linear regression on measured data points.
        
        Returns:
            Tuple of (slope, intercept, r_squared):
                - slope: N per FOR unit (ideally ~0.843 N/unit)
                - intercept: Force at FOR=0 (ideally ~20 N)
                - r_squared: Coefficient of determination
        """
        # Filter to only points with measured values
        measured_points = [p for p in self.calibration_data if p.measured_force_N > 0]

        if len(measured_points) < 2:
            print("Not enough measured data points for regression")
            # Return theoretical values
            slope = (self.FORCE_MAX_N - self.FORCE_MIN_N) / 255.0  # ~0.843
            return slope, self.FORCE_MIN_N, 0.0

        x = np.array([p.for_setting for p in measured_points])
        y = np.array([p.measured_force_N for p in measured_points])

        # Linear regression: y = slope * x + intercept
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(slope), float(intercept), float(r_squared)

    def print_results(self) -> None:
        """Print calibration results and analysis."""
        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)

        # Print data table
        print("\n{:>6} | {:>12} | {:>12} | {:>8} | {}".format(
            "FOR", "Expected (N)", "Measured (N)", "Position", "Detected"
        ))
        print("-" * 60)
        for p in self.calibration_data:
            measured_str = f"{p.measured_force_N:.1f}" if p.measured_force_N > 0 else "-"
            print("{:>6} | {:>12.1f} | {:>12} | {:>8} | {}".format(
                p.for_setting, p.expected_force_N, measured_str,
                p.position_at_grip, "Yes" if p.object_detected else "No"
            ))

        # Calculate and print force constant
        slope, intercept, r_squared = self.calculate_force_constant()
        
        print("\n" + "-" * 60)
        print("FORCE CONSTANT ANALYSIS")
        print("-" * 60)
        
        # Theoretical values
        theoretical_slope = (self.FORCE_MAX_N - self.FORCE_MIN_N) / 255.0
        print(f"\nTheoretical (from Robotiq spec):")
        print(f"  Force = {self.FORCE_MIN_N:.1f} + (FOR / 255) × {self.FORCE_MAX_N - self.FORCE_MIN_N:.1f}")
        print(f"  Slope: {theoretical_slope:.4f} N/unit")
        print(f"  Intercept: {self.FORCE_MIN_N:.1f} N")

        measured_points = [p for p in self.calibration_data if p.measured_force_N > 0]
        if len(measured_points) >= 2:
            print(f"\nMeasured (from your calibration):")
            print(f"  Force = {intercept:.1f} + FOR × {slope:.4f}")
            print(f"  Slope: {slope:.4f} N/unit")
            print(f"  Intercept: {intercept:.1f} N")
            print(f"  R²: {r_squared:.4f}")
            
            # Compare to theoretical
            slope_error = abs(slope - theoretical_slope) / theoretical_slope * 100
            intercept_error = abs(intercept - self.FORCE_MIN_N) / self.FORCE_MIN_N * 100
            print(f"\n  Deviation from spec:")
            print(f"    Slope error: {slope_error:.1f}%")
            print(f"    Intercept error: {intercept_error:.1f}%")
        else:
            print("\nNot enough measured data for calibration.")
            print("Run with a force gauge to get actual measurements.")

        # Code snippet for updating robotiq_gripper.py
        print("\n" + "-" * 60)
        print("UPDATE CODE")
        print("-" * 60)
        if len(measured_points) >= 2:
            print("\nTo use your calibrated values, update robotiq_gripper.py:")
            print(f"    self._force_min_N = {intercept:.1f}")
            print(f"    self._force_max_N = {intercept + slope * 255:.1f}")
        else:
            print("\nUsing theoretical values (update if you measure actual forces):")
            print(f"    self._force_min_N = {self.FORCE_MIN_N:.1f}")
            print(f"    self._force_max_N = {self.FORCE_MAX_N:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Robotiq 2F-85 gripper force constant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive calibration with default FOR values
    python robotiq_get_force_constant.py

    # Custom FOR values
    python robotiq_get_force_constant.py --for-values 25 75 125 175 225

    # Non-interactive mode (just measure detection, no force input)
    python robotiq_get_force_constant.py --no-interactive

    # Custom gripper IP
    python robotiq_get_force_constant.py --ip 192.168.1.20
        """
    )
    parser.add_argument(
        "--ip", type=str, default="192.168.1.10",
        help="Gripper hostname/IP (default: 192.168.1.10)"
    )
    parser.add_argument(
        "--port", type=int, default=63352,
        help="Gripper port (default: 63352)"
    )
    parser.add_argument(
        "--for-values", type=int, nargs="+",
        default=[0, 50, 100, 150, 200, 255],
        help="FOR settings to test (default: 0 50 100 150 200 255)"
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Don't prompt for force measurements"
    )
    parser.add_argument(
        "--skip-activation", action="store_true",
        help="Skip gripper activation (use if already activated)"
    )

    args = parser.parse_args()

    calibrator = RobotiqForceCalibrator(hostname=args.ip, port=args.port)

    try:
        # Connect
        if not calibrator.connect():
            return 1

        # Activate
        if not args.skip_activation:
            if not calibrator.activate():
                return 1
        else:
            print("Skipping activation (assuming gripper is already active)")

        # Run calibration
        calibrator.run_calibration_sweep(
            for_values=args.for_values,
            interactive=not args.no_interactive
        )

        # Print results
        calibrator.print_results()

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
    finally:
        calibrator.disconnect()

    return 0


if __name__ == "__main__":
    exit(main())
