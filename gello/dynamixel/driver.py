"""
Optimized Dynamixel Driver for GELLO Gravity Compensation

Key optimizations:
1. Lock-free reading using atomic state snapshots
2. Minimized lock contention (lock only during USB I/O)
3. No sleep in read loop (maximum throughput)
4. Optional combined read+write for lowest latency
5. State age tracking for debugging

"""

import os
import subprocess
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Additional control table addresses and lengths for current mode and velocities

ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3
ADDR_PRESENT_CURRENT = 126
LEN_PRESENT_CURRENT = 2

# Servo-specific mappings and limits

TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 880.28,  # 1.0 Nm @ 0.88A -> 1.136 Nm/A
    "XM430_W210_T": 285.0,   # 3.0 Nm @ 2.3A -> 1.304 Nm/A
    "XM430_W350_T": 208.5,   # 4.1 Nm @ 2.3A -> 1.783 Nm/A
}

SERVO_CURRENT_LIMITS = {
    "XC330_T288_T": 880,   # Stall Current 0.88A (unit 1mA)
    "XM430_W210_T": 855,   # Stall Current 2.3A (unit 2.69mA -> ~855)
    "XM430_W350_T": 855,   # Stall Current 2.3A (unit 2.69mA -> ~855)
}

CURRENT_UNIT_MA = {
    "XC330_T288_T": 1.0,
    "XM430_W210_T": 2.69,
    "XM430_W350_T": 2.69,
}


@dataclass
class JointState:
    """Immutable joint state snapshot for lock-free reading."""
    positions: np.ndarray  # radians
    velocities: np.ndarray  # rad/s
    currents: np.ndarray    # mA (physical units)
    timestamp: float  # time.time() when captured
    
    def age_ms(self) -> float:
        """Get age of this state in milliseconds."""
        return (time.time() - self.timestamp) * 1000.0


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos."""
        ...

    def set_current(self, currents: Sequence[float]):
        """Set motor currents (mA) for current control mode."""
        ...

    def set_torque(self, torques: Sequence[float]):
        """Set joint torques (Nm), mapped to motor currents using servo mappings."""
        ...

    def set_operating_mode(self, mode: int):
        """Set the operating mode (e.g., CURRENT_CONTROL_MODE or POSITION_CONTROL_MODE)."""
        ...

    def verify_operating_mode(self, expected_mode: int):
        """Verify that servos are in the expected operating mode."""
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos."""
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos."""
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians."""
        ...

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions (rad) and velocities (rad/s)."""
        ...

    def get_currents(self) -> np.ndarray:
        """Get motor currents in mA (physical units)."""
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=float)
        self._velocities = np.zeros(len(ids), dtype=float)
        self._currents = np.zeros(len(ids), dtype=float)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError("The length of joint_angles must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles, dtype=float)

    def set_current(self, currents: Sequence[float]):
        if len(currents) != len(self._ids):
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")
        self._currents = np.array(currents, dtype=float)

    def set_torque(self, torques: Sequence[float]):
        self.set_current(torques)

    def set_operating_mode(self, mode: int):
        pass

    def verify_operating_mode(self, expected_mode: int):
        pass

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._joint_angles.copy(), self._velocities.copy()

    def get_currents(self) -> np.ndarray:
        return self._currents.copy()

    def get_positions(self) -> np.ndarray:
        return self.get_joints()

    def get_state_age_ms(self) -> float:
        return 0.0

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        ids: Sequence[int],
        servo_types: Optional[Sequence[str]] = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        max_retries: int = 3,
        use_fake_fallback: bool = True,
        velocity_filter_alpha: float = 0.5,
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids: A list of IDs for the Dynamixel servos.
            servo_types: Optional servo model names for torque->current mapping.
            port: The USB port to connect to the arm.
            baudrate: The baudrate for communication.
            max_retries: Maximum number of initialization attempts.
            use_fake_fallback: Whether to fallback to FakeDynamixelDriver on failure.
            velocity_filter_alpha: EMA filter alpha for velocities (0.0 to 1.0).
        """
        self._ids = list(ids)
        self._num_joints = len(ids)
        self._lock = Lock()
        self._port = port
        self._baudrate = baudrate
        self._max_retries = max_retries
        self._use_fake_fallback = use_fake_fallback
        self._is_fake = False
        self._torque_enabled = False
        self._stop_thread = Event()

        self._velocity_filter_alpha = velocity_filter_alpha
        self._filtered_velocities: Optional[np.ndarray] = None

        # Lock-free state: atomic reference swap (Python GIL guarantees atomicity)
        self._latest_state: Optional[JointState] = None
        
        # Statistics for debugging
        self._read_count = 0
        self._read_errors = 0
        self._last_read_duration_ms = 0.0

        # Optional torque-current mapping
        self._servo_types = list(servo_types) if servo_types is not None else None
        if self._servo_types is not None:
            self.torque_to_current_map = np.array(
                [TORQUE_TO_CURRENT_MAPPING[s] for s in self._servo_types]
            )
            self.current_limits = np.array(
                [SERVO_CURRENT_LIMITS[s] for s in self._servo_types]
            )
            self.current_conversions = np.array(
                [CURRENT_UNIT_MA.get(s, 1.0) for s in self._servo_types], dtype=np.float64
            )
        else:
            self.torque_to_current_map = None
            self.current_limits = None
            self.current_conversions = np.full(self._num_joints, 1.0, dtype=np.float64)

        # Fake driver fallback storage
        self._fake_joint_angles: Optional[np.ndarray] = None
        self._fake_velocities: Optional[np.ndarray] = None
        self._fake_currents: Optional[np.ndarray] = None

        # Initialize with retry logic
        if not self._initialize_with_retries():
            if self._use_fake_fallback:
                print("Using fake Dynamixel driver")
                self._initialize_fake_driver()
            else:
                raise RuntimeError(
                    "Failed to initialize Dynamixel driver after all retries"
                )

    def _initialize_with_retries(self) -> bool:
        """Initialize the Dynamixel driver with retry logic."""
        for attempt in range(self._max_retries):
            print(
                f"Attempting to initialize Dynamixel driver (attempt {attempt + 1}/{self._max_retries})"
            )

            if not self._check_port_availability():
                print("Port is busy, attempting to free it...")
                if not self._kill_processes_using_port():
                    print("Failed to free port, trying to fix permissions...")
                    self._fix_port_permissions()
                time.sleep(2)

            try:
                self._initialize_hardware()
                print(f"Successfully initialized Dynamixel driver on {self._port}")
                return True
            except Exception as e:
                print(f"Failed to initialize Dynamixel driver: {e}")
                if attempt < self._max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("Max retries reached")

        return False

    def _initialize_hardware(self):
        """Initialize the hardware connection."""
        self._prepare_port()

        # Initialize handlers
        self._portHandler = PortHandler(self._port)
        self._packetHandler = PacketHandler(2.0)
        
        # Read both velocity and position in one transaction (8 bytes total)
        # self._groupSyncRead = GroupSyncRead(
        #     self._portHandler,
        #     self._packetHandler,
        #     ADDR_PRESENT_VELOCITY,
        #     LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION,
        # )

        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_CURRENT,  # 126 statt 128
            LEN_PRESENT_CURRENT + LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION,  # 2+4+4=10
        )
        
        # Separate writers for position and current
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self._groupSyncWriteCurrent = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
        )

        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(self._baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {self._baudrate}")

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {self._port}, {e}")

        self._start_reading_thread()

    def _initialize_fake_driver(self):
        """Initialize as a fake driver."""
        self._is_fake = True
        self._fake_joint_angles = np.zeros(self._num_joints, dtype=float)
        self._fake_velocities = np.zeros(self._num_joints, dtype=float)
        self._fake_currents = np.zeros(self._num_joints, dtype=float)

    def _start_reading_thread(self):
        """Start the background reading thread."""
        self._reading_thread = Thread(target=self._read_joint_states, daemon=True)
        self._reading_thread.start()

    def _read_joint_states(self):
        """Continuously read joint states - optimized version.
        
        Key optimizations:
        - No sleep (read as fast as USB allows)
        - Lock only during USB I/O (not during parsing)
        - Atomic state update via reference swap

        """
        # Pre-allocate arrays outside loop
        raw_positions = np.zeros(self._num_joints, dtype=np.int32)
        raw_velocities = np.zeros(self._num_joints, dtype=np.int32)
        raw_currents = np.zeros(self._num_joints, dtype=np.int32)
        
        while not self._stop_thread.is_set():
            read_start = time.time()
            
            # === USB I/O under lock (minimize lock duration) ===
            with self._lock:
                dxl_comm_result = self._groupSyncRead.txRxPacket()
            
            if dxl_comm_result != COMM_SUCCESS:
                self._read_errors += 1
                # Only sleep on error to avoid busy-spin on persistent failures
                time.sleep(0.001)
                continue
            
            # === Parse results OUTSIDE lock (getData is thread-safe for reading) ===
            try:
                for i, dxl_id in enumerate(self._ids):
                    # Current (2 Bytes bei Adresse 126, signed 16-bit)
                    current_raw = self._groupSyncRead.getData(
                        dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
                    )
                    if current_raw > 0x7FFF:
                        current_raw -= 0x10000
                    raw_currents[i] = current_raw
                    # Velocity (4 bytes)
                    velocity = self._groupSyncRead.getData(
                        dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                    )
                    # Two's complement for signed 32-bit
                    if velocity > 0x7FFFFFFF:
                        velocity -= 0x100000000
                    raw_velocities[i] = velocity
                    
                    # Position (4 bytes)
                    position = self._groupSyncRead.getData(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    )
                    if position > 0x7FFFFFFF:
                        position -= 0x100000000
                    raw_positions[i] = position
                
                # Convert to physical units
                positions_rad = raw_positions.astype(np.float64) / 2048.0 * np.pi
                # Velocity unit: 0.229 rev/min -> rad/s
                velocities_rad_s = raw_velocities.astype(np.float64) * 0.229 * 2.0 * np.pi / 60.0
                # convert current
                current_mA = raw_currents.astype(np.float64) * self.current_conversions
                
                # Apply Exponential Moving Average (EMA) filter to velocities
                if self._filtered_velocities is None:
                    self._filtered_velocities = velocities_rad_s.copy()
                else:
                    self._filtered_velocities = (
                        self._velocity_filter_alpha * velocities_rad_s +
                        (1.0 - self._velocity_filter_alpha) * self._filtered_velocities
                    )

                # Atomic state update (Python GIL guarantees reference assignment is atomic)
                self._latest_state = JointState(
                    positions=positions_rad,
                    velocities=self._filtered_velocities.copy(),
                    currents=current_mA,
                    timestamp=time.time()
                )
                
                self._read_count += 1
                self._last_read_duration_ms = (time.time() - read_start) * 1000.0
                
                # Yield to other threads (writer) to avoid lock starvation
                time.sleep(0.0005)
                
            except Exception as e:
                self._read_errors += 1
                print(f"Read parse error: {e}")
                time.sleep(0.001)

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint positions (rad) and velocities (rad/s) - lock-free read."""
        if self._is_fake:
            return self._fake_joint_angles.copy(), self._fake_velocities.copy()
        
        # Wait for first state (only on startup)
        timeout = 5.0
        start = time.time()
        while self._latest_state is None:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for first joint state")
            time.sleep(0.01)
        
        # Lock-free read: grab reference (atomic in Python)
        state = self._latest_state
        return state.positions.copy(), state.velocities.copy()

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians."""
        positions, _ = self.get_positions_and_velocities()
        return positions

    def get_currents(self) -> np.ndarray:
        """Get motor currents in mA (physical units) - lock-free read."""
        if self._is_fake:
            return self._fake_currents.copy()

        timeout = 5.0
        start = time.time()
        while self._latest_state is None:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for first joint state")
            time.sleep(0.01)

        state = self._latest_state
        return state.currents.copy()

    def get_positions(self) -> np.ndarray:
        """Alias for get_joints()."""
        return self.get_joints()

    def get_state_age_ms(self) -> float:
        """Get age of latest state in milliseconds (for debugging)."""
        if self._is_fake:
            return 0.0
        if self._latest_state is None:
            return float('inf')
        return self._latest_state.age_ms()

    def get_read_stats(self) -> dict:
        """Get reading statistics for debugging."""
        return {
            "read_count": self._read_count,
            "read_errors": self._read_errors,
            "last_read_duration_ms": self._last_read_duration_ms,
            "state_age_ms": self.get_state_age_ms(),
            "error_rate": self._read_errors / max(self._read_count, 1),
        }

    def set_joints(self, joint_angles: Sequence[float]):
        """Set goal positions for position control mode."""
        if len(joint_angles) != self._num_joints:
            raise ValueError("The length of joint_angles must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        if self._is_fake:
            self._fake_joint_angles = np.array(joint_angles)
            return

        with self._lock:
            for dxl_id, angle in zip(self._ids, joint_angles):
                position_value = int(angle * 2048 / np.pi)
                param_goal_position = [
                    DXL_LOBYTE(DXL_LOWORD(position_value)),
                    DXL_HIBYTE(DXL_LOWORD(position_value)),
                    DXL_LOBYTE(DXL_HIWORD(position_value)),
                    DXL_HIBYTE(DXL_HIWORD(position_value)),
                ]
                if not self._groupSyncWrite.addParam(dxl_id, param_goal_position):
                    raise RuntimeError(
                        f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                    )

            dxl_comm_result = self._groupSyncWrite.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError("Failed to syncwrite goal position")
            self._groupSyncWrite.clearParam()

    def set_current(self, currents: Sequence[float]):
        """Set goal currents for current control mode."""
        if len(currents) != self._num_joints:
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")

        if self._is_fake:
            self._fake_currents = np.array(currents, dtype=float)
            return

        # Clip currents to servo-specific limits
        currents_array = np.array(currents, dtype=float)
        if self.current_limits is not None:
            currents_array = np.clip(currents_array, -self.current_limits, self.current_limits)

        with self._lock:
            for dxl_id, current in zip(self._ids, currents_array):
                current_value = int(current)
                param_goal_current = [
                    DXL_LOBYTE(current_value),
                    DXL_HIBYTE(current_value),
                ]
                if not self._groupSyncWriteCurrent.addParam(dxl_id, param_goal_current):
                    raise RuntimeError(
                        f"Failed to set current for Dynamixel with ID {dxl_id}"
                    )
            
            dxl_comm_result = self._groupSyncWriteCurrent.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError("Failed to syncwrite goal current")
            self._groupSyncWriteCurrent.clearParam()

    def set_torque(self, torques: Sequence[float]):
        """Set joint torques (Nm), converted to motor currents."""
        if self.torque_to_current_map is None:
            raise RuntimeError(
                "Torque-to-current mapping is not configured. Provide servo_types to the driver."
            )
        currents = (self.torque_to_current_map * np.array(torques)).tolist()
        self.set_current(currents)

    def torque_enabled(self) -> bool:
        """Check if torque is enabled."""
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        """Enable or disable torque on all servos."""
        if self._is_fake:
            self._torque_enabled = enable
            return

        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                retries = 3
                for attempt in range(retries):
                    dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                        self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                    )
                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )
        self._torque_enabled = enable

    def set_operating_mode(self, mode: int):
        """Set operating mode for all servos."""
        if self._is_fake:
            return
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(
                        f"Failed to set operating mode for Dynamixel with ID {dxl_id}"
                    )

    def verify_operating_mode(self, expected_mode: int):
        """Verify all servos are in the expected operating mode."""
        if self._is_fake:
            return
        with self._lock:
            for dxl_id in self._ids:
                mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                    raise RuntimeError(
                        f"Operating mode mismatch for Dynamixel ID {dxl_id} "
                        f"(got {mode}, expected {expected_mode})"
                    )

    def _check_port_availability(self) -> bool:
        """Check if the port is available."""
        try:
            if not os.path.exists(self._port):
                print(f"Port {self._port} does not exist")
                return False

            result = subprocess.run(["lsof", self._port], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    print(f"Port {self._port} is being used by other processes:")
                    for line in lines[1:]:
                        print(f"  {line}")
                    return False
            return True
        except Exception as e:
            print(f"Error checking port availability: {e}")
            return False

    def _kill_processes_using_port(self) -> bool:
        """Kill processes using the port."""
        try:
            result = subprocess.run(["fuser", "-k", self._port], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Killed processes using {self._port}")
                time.sleep(1)
                return True
            return False
        except Exception as e:
            print(f"Error killing processes: {e}")
            return False

    def _fix_port_permissions(self) -> bool:
        """Fix port permissions."""
        try:
            result = subprocess.run(
                ["sudo", "chmod", "666", self._port], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"Fixed permissions for {self._port}")
                return True
            return False
        except Exception as e:
            print(f"Error fixing port permissions: {e}")
            return False

    def _prepare_port(self):
        """Prepare the port for connection."""
        if not self._check_port_availability():
            print(f"Port {self._port} is not available, attempting to fix...")
            self._kill_processes_using_port()
            self._fix_port_permissions()
            if not self._check_port_availability():
                print(f"Warning: Port {self._port} may still have issues")

    def close(self):
        """Close the driver and release resources."""
        if self._is_fake:
            return

        self._stop_thread.set()
        if hasattr(self, '_reading_thread'):
            self._reading_thread.join(timeout=2.0)
        if hasattr(self, '_portHandler'):
            self._portHandler.closePort()
        
        # Print final stats
        stats = self.get_read_stats()
        print(f"Driver closed. Read stats: {stats['read_count']} reads, "
              f"{stats['read_errors']} errors ({stats['error_rate']*100:.2f}%)")


def main():
    """Test the driver."""
    ids = [1, 2, 3, 4, 5, 6, 7]
    servo_types = [
        "XC330_T288_T", "XM430_W350_T", "XM430_W350_T",
        "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T"
    ]

    try:
        driver = DynamixelDriver(
            ids, 
            servo_types=servo_types,
            port="/dev/ttyDXL_gello",
            baudrate=4000000
        )
    except Exception as e:
        print(f"Failed to create driver: {e}")
        return

    print("\nReading joint states (Ctrl+C to stop)...")
    try:
        loop_count = 0
        start_time = time.time()
        
        while True:
            positions, velocities = driver.get_positions_and_velocities()
            loop_count += 1
            
            # Print every 100 iterations
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                hz = loop_count / elapsed
                stats = driver.get_read_stats()
                
                print(f"\n[Loop {loop_count}, {hz:.1f} Hz effective]")
                print(f"  Positions (deg): {[f'{np.rad2deg(p):+7.1f}' for p in positions]}")
                print(f"  Velocities (rad/s): {[f'{v:+.3f}' for v in velocities]}")
                print(f"  State age: {stats['state_age_ms']:.2f} ms")
                print(f"  Read duration: {stats['last_read_duration_ms']:.2f} ms")
                print(f"  Error rate: {stats['error_rate']*100:.2f}%")
            
            # Small sleep to simulate control loop (remove for max throughput test)
            time.sleep(0.003)  # ~333 Hz
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        driver.close()


if __name__ == "__main__":
    main()