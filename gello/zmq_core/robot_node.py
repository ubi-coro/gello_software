import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq

from gello.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "get_joint_torques":
                    # For robots that support torque feedback
                    if hasattr(self._robot, "get_joint_torques"):
                        result = self._robot.get_joint_torques()
                    else:
                         result = np.zeros(self._robot.num_dofs())
                elif method == "tare_jacobian_torques":
                    if hasattr(self._robot, "tare_jacobian_torques"):
                        self._robot.tare_jacobian_torques(**args)
                        result = True
                    else:
                        print("Warning: robot does not support tare_jacobian_torques")
                        result = False
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                # Timeout occurred - don't spam the console
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        with self._lock:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        try:
            with self._lock:
                self._socket.send(send_message)
                result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        with self._lock:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        try:
            with self._lock:
                self._socket.send(send_message)
                result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def get_joint_torques(self) -> np.ndarray:
        """Get the current external joint torques from the remote robot.

        Returns:
            np.ndarray: The joint torques
        """
        request = {"method": "get_joint_torques"}
        send_message = pickle.dumps(request)
        try:
            with self._lock:
                self._socket.send(send_message)
                result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            # Return zeros if timeout to avoid crashing control loop
            # BUT: ideally we should raise or log
            print("ZMQ timeout getting torques")
            return np.zeros(0)  # Caller will handle shape mismatch or zero

    def tare_jacobian_torques(self, num_samples: int = 100) -> None:
        """Tare the torque sensors on the remote robot.
        
        Args:
            num_samples: Number of samples to average for the tare.
        """
        request = {
            "method": "tare_jacobian_torques",
            "args": {"num_samples": num_samples}
        }
        send_message = pickle.dumps(request)
        try:
            with self._lock:
                self._socket.send(send_message)
                # We expect a boolean or None, just consuming the reply
                pickle.loads(self._socket.recv())
        except zmq.Again:
            print("ZMQ timeout taring robot")

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        with self._lock:
            self._socket.close()
            self._context.term()
