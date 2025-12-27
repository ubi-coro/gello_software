from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gello.robots.robot import Robot


class DynamixelRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        real: bool = False,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        gripper_config: Optional[Tuple[int, float, float]] = None,
        start_joints: Optional[np.ndarray] = None,
        servo_types: Optional[Sequence[str]] = None,
    ):
        from gello.dynamixel.driver import (
            DynamixelDriver,
            DynamixelDriverProtocol,
            FakeDynamixelDriver,
        )

        print(f"attempting to connect to port: {port}")
        self.gripper_open_close: Optional[Tuple[float, float]]
        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None

            # joint_ids.append(gripper_config[0])
            # joint_offsets.append(0.0)
            # joint_signs.append(1)
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
            if servo_types is not None:
                # Assuming gripper is same type as last joint or generic?
                # For now let's just append the last type if available or handle it in driver
                # Actually driver expects servo_types to match ids length if provided.
                # If gripper is added, we should add a type for it.
                # Usually gripper is a different motor.
                # Let's assume the user provides servo_types for ALL motors including gripper if they provide it.
                # But wait, the user config in gello_agent.py usually defines arm joints.
                # If we append gripper here, we might need to append gripper type.
                # Let's assume for now servo_types covers the arm joints.
                # If gripper is added, we need to add a type for it.
                # Let's default to the last type in the list if we don't know.
                # Or better, let's ask the user or assume a default.
                # For now, let's just pass what we have and see if driver complains.
                # Actually, let's append a default type for gripper if servo_types is present.
                # Most grippers in GELLO are XL330 or XC330.
                pass
        else:
            self.gripper_open_close = None

        self._joint_ids = joint_ids
        self._driver: DynamixelDriverProtocol

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        if real:
            # Handle servo_types length mismatch if gripper was added
            if servo_types is not None and len(servo_types) < len(joint_ids):
                # If gripper was added (length diff is 1), append a default gripper type
                # Most GELLO grippers are XC330
                if len(joint_ids) - len(servo_types) == 1:
                    servo_types = list(servo_types) + ["XC330_T288_T"]
            
            self._driver = DynamixelDriver(
                joint_ids, 
                port=port, 
                baudrate=baudrate, 
                servo_types=servo_types
            )
            self._driver.set_torque_mode(False)
        else:
            self._driver = FakeDynamixelDriver(joint_ids)
        self._torque_on = False
        self._last_pos = None
        self._alpha = 0.99

        if start_joints is not None:
            # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
            new_joint_offsets = []
            current_joints = self.get_joint_state()
            assert current_joints.shape == start_joints.shape
            if gripper_config is not None:
                current_joints = current_joints[:-1]
                start_joints = start_joints[:-1]
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi
                    * 2
                    * np.round((-s_joint + c_joint) / (2 * np.pi))
                    * self._joint_signs[idx]
                    + joint_offset
                )
            if gripper_config is not None:
                new_joint_offsets.append(self._joint_offsets[-1])
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.set_joints((joint_state + self._joint_offsets).tolist())

    def command_joint_torques(self, torques: np.ndarray) -> None:
        """Command joint torques to the robot.

        Args:
            torques (np.ndarray): The torques to command.
        """
        # Apply joint signs to torques
        # Note: offsets don't affect torques, but signs do
        driver_torques = torques * self._joint_signs
        self._driver.set_torque(driver_torques.tolist())

    def get_joint_velocities(self) -> np.ndarray:
        """Get the current joint velocities of the robot.

        Returns:
            np.ndarray: The joint velocities.
        """
        _, vels = self._driver.get_positions_and_velocities()
        return vels * self._joint_signs

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {
            "joint_state": self.get_joint_state(),
            "joint_velocities": self.get_joint_velocities(),
        }
