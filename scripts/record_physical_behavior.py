#!/usr/bin/env python3
"""
Record Physical Behavior of GELLO Leader for URDF Validation

This script records the physical behavior of the GELLO leader arm while you
manually move it. It captures:
  - Joint positions (rad)
  - Joint velocities (rad/s) 
  - Estimated gravity torques from URDF model
  - (Optional) Motor currents if available

The recorded data can be used to:
  1. Compare real dynamics with URDF predictions
  2. Identify friction/damping not captured in URDF
  3. Validate inertia parameters
  4. Debug joint sign/offset issues

Usage:
    python scripts/record_physical_behavior.py --urdf <path> --port <port> [--duration 30]
    
Example:
    python scripts/record_physical_behavior.py \
        --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/urdf/GELLO_Assembly_URDF_V4.urdf \
        --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0 \
        --duration 60

After recording, move each joint slowly and deliberately to see how well
the URDF gravity model matches reality.
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Warning: Pinocchio not installed. URDF dynamics comparison will be disabled.")
    print("Install with: pip install pin")

from gello.dynamixel.driver import DynamixelDriver


class PhysicalBehaviorRecorder:
    """Records physical behavior data from GELLO leader for URDF comparison."""
    
    def __init__(
        self,
        urdf_path: Optional[str],
        port: str,
        joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        joint_offsets: Optional[Tuple[float, ...]] = None,
        joint_signs: Optional[Tuple[int, ...]] = None,
        baudrate: int = 57600,
        servo_types: Optional[Tuple[str, ...]] = None,
    ):
        """
        Initialize the recorder.
        
        Args:
            urdf_path: Path to URDF file for dynamics comparison (optional)
            port: Serial port for Dynamixel communication
            joint_ids: Motor IDs for each joint
            joint_offsets: Joint angle offsets (rad)
            joint_signs: Joint direction signs (+1 or -1)
            baudrate: Serial communication baudrate
            servo_types: Servo model names for torque estimation
        """
        self.port = port
        self.joint_ids = joint_ids
        self.n_joints = len(joint_ids)
        self.baudrate = baudrate
        self.servo_types = servo_types
        
        # Joint transformations
        if joint_offsets is None:
            self.joint_offsets = np.zeros(self.n_joints)
        else:
            self.joint_offsets = np.array(joint_offsets)
            
        if joint_signs is None:
            self.joint_signs = np.ones(self.n_joints)
        else:
            self.joint_signs = np.array(joint_signs)
        
        # Pinocchio model for URDF dynamics
        self.model = None
        self.data = None
        if urdf_path and HAS_PINOCCHIO:
            # Resolve URDF path - try absolute, then relative to project root
            urdf_resolved = urdf_path
            if not os.path.isabs(urdf_path):
                # Try relative to project root (parent of scripts/)
                project_root = Path(__file__).parent.parent
                urdf_resolved = str(project_root / urdf_path)
            
            if os.path.exists(urdf_resolved):
                try:
                    self.model = pin.buildModelFromUrdf(urdf_resolved)
                    self.data = self.model.createData()
                    print(f"Loaded URDF model: {urdf_resolved}")
                    print(f"  Model has {self.model.nq} joints, {self.model.nv} DoFs")
                except Exception as e:
                    print(f"Warning: Failed to load URDF: {e}")
            else:
                print(f"Warning: URDF file not found: {urdf_path}")
                print(f"  Tried: {urdf_resolved}")
        
        # Initialize driver
        self.driver: Optional[DynamixelDriver] = None
        self.running = False
        
        # Data storage
        self.recorded_data: List[dict] = []
        
    def _raw_to_joint(self, raw_pos: np.ndarray) -> np.ndarray:
        """Convert raw motor positions to joint angles."""
        return (raw_pos - self.joint_offsets) * self.joint_signs
    
    def _compute_urdf_gravity(self, q: np.ndarray) -> Optional[np.ndarray]:
        """Compute gravity torques from URDF model."""
        if self.model is None or self.data is None:
            return None
        
        # Ensure q has the right size
        if len(q) != self.model.nq:
            # Pad or truncate if needed
            q_model = np.zeros(self.model.nq)
            n = min(len(q), self.model.nq)
            q_model[:n] = q[:n]
        else:
            q_model = q
            
        try:
            return pin.computeGeneralizedGravity(self.model, self.data, q_model)
        except Exception as e:
            print(f"Warning: Gravity computation failed: {e}")
            return None
    
    def _compute_urdf_mass_matrix(self, q: np.ndarray) -> Optional[np.ndarray]:
        """Compute mass matrix from URDF model."""
        if self.model is None or self.data is None:
            return None
        
        if len(q) != self.model.nq:
            q_model = np.zeros(self.model.nq)
            n = min(len(q), self.model.nq)
            q_model[:n] = q[:n]
        else:
            q_model = q
            
        try:
            pin.crba(self.model, self.data, q_model)
            return self.data.M.copy()
        except Exception:
            return None
    
    def _compute_urdf_coriolis(self, q: np.ndarray, dq: np.ndarray) -> Optional[np.ndarray]:
        """Compute Coriolis/centrifugal torques C(q,dq)*dq from URDF model."""
        if self.model is None or self.data is None:
            return None
        
        # Ensure vectors have the right size
        if len(q) != self.model.nq:
            q_model = np.zeros(self.model.nq)
            q_model[:min(len(q), self.model.nq)] = q[:min(len(q), self.model.nq)]
        else:
            q_model = q
            
        if len(dq) != self.model.nv:
            dq_model = np.zeros(self.model.nv)
            dq_model[:min(len(dq), self.model.nv)] = dq[:min(len(dq), self.model.nv)]
        else:
            dq_model = dq
            
        try:
            # Compute Coriolis matrix C(q, dq)
            pin.computeCoriolisMatrix(self.model, self.data, q_model, dq_model)
            # Return C(q,dq) * dq
            return self.data.C @ dq_model
        except Exception:
            return None
    
    def _compute_urdf_rnea(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute full inverse dynamics using RNEA: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        
        This tells us what torque would be needed to produce the observed motion.
        """
        if self.model is None or self.data is None:
            return None
        
        # Ensure vectors have the right size
        if len(q) != self.model.nq:
            q_model = np.zeros(self.model.nq)
            q_model[:min(len(q), self.model.nq)] = q[:min(len(q), self.model.nq)]
        else:
            q_model = q
            
        if len(dq) != self.model.nv:
            dq_model = np.zeros(self.model.nv)
            dq_model[:min(len(dq), self.model.nv)] = dq[:min(len(dq), self.model.nv)]
        else:
            dq_model = dq
            
        if len(ddq) != self.model.nv:
            ddq_model = np.zeros(self.model.nv)
            ddq_model[:min(len(ddq), self.model.nv)] = ddq[:min(len(ddq), self.model.nv)]
        else:
            ddq_model = ddq
            
        try:
            # RNEA: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
            return pin.rnea(self.model, self.data, q_model, dq_model, ddq_model)
        except Exception:
            return None
    
    def connect(self) -> bool:
        """Connect to the Dynamixel servos."""
        try:
            print(f"Connecting to port: {self.port}")
            self.driver = DynamixelDriver(
                ids=self.joint_ids,
                port=self.port,
                baudrate=self.baudrate,
                servo_types=self.servo_types,
                use_fake_fallback=False,
            )
            # Try to disable torque - we want passive reading
            # The driver may have already disabled it during init, so don't fail if this errors
            try:
                self.driver.set_torque_mode(False)
            except Exception as e:
                print(f"Note: Could not explicitly disable torque: {e}")
                print("  This is usually OK - torque may already be disabled.")
            
            # Verify we can read positions
            try:
                test_pos = self.driver.get_joints()
                print(f"Connected! Read initial positions: {test_pos}")
            except Exception as e:
                print(f"Warning: Could not read initial positions: {e}")
            
            print("Ready for passive recording.")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from servos."""
        if self.driver is not None:
            try:
                self.driver.set_torque_mode(False)
                self.driver.close()
            except Exception:
                pass
            self.driver = None
    
    def record_sample(self, prev_vel: Optional[np.ndarray] = None, prev_time: Optional[float] = None) -> dict:
        """Record a single data sample.
        
        Args:
            prev_vel: Previous velocity for acceleration computation
            prev_time: Previous timestamp for acceleration computation
        """
        if self.driver is None:
            raise RuntimeError("Not connected")
        
        timestamp = time.time()
        
        # Get positions and velocities
        raw_pos, raw_vel = self.driver.get_positions_and_velocities()
        
        # Convert to joint space
        joint_pos = self._raw_to_joint(raw_pos)
        joint_vel = raw_vel * self.joint_signs  # Velocities also need sign correction
        
        # Compute acceleration by differentiating velocity
        joint_acc = np.zeros(self.n_joints)
        if prev_vel is not None and prev_time is not None:
            dt = timestamp - prev_time
            if dt > 0:
                joint_acc = (joint_vel - prev_vel) / dt
        
        # Compute URDF-predicted gravity torques: g(q)
        gravity_torques = self._compute_urdf_gravity(joint_pos)
        
        # Compute Coriolis torques: C(q,dq)*dq
        coriolis_torques = self._compute_urdf_coriolis(joint_pos, joint_vel)
        
        # Compute full RNEA inverse dynamics: τ = M(q)q̈ + C(q,dq)*dq + g(q)
        # This is the torque that would be needed to produce the observed motion
        rnea_torques = self._compute_urdf_rnea(joint_pos, joint_vel, joint_acc)
        
        # Get mass matrix diagonal (inertias)
        mass_matrix = self._compute_urdf_mass_matrix(joint_pos)
        inertias = np.diag(mass_matrix) if mass_matrix is not None else None
        
        sample = {
            'timestamp': timestamp,
            'joint_positions': joint_pos.copy(),
            'joint_velocities': joint_vel.copy(),
            'joint_accelerations': joint_acc.copy(),
            'raw_positions': raw_pos.copy(),
            'raw_velocities': raw_vel.copy(),
        }
        
        if gravity_torques is not None:
            sample['urdf_gravity_torques'] = gravity_torques[:self.n_joints].copy()
        if coriolis_torques is not None:
            sample['urdf_coriolis_torques'] = coriolis_torques[:self.n_joints].copy()
        if rnea_torques is not None:
            sample['urdf_rnea_torques'] = rnea_torques[:self.n_joints].copy()
        if inertias is not None:
            sample['urdf_inertias'] = inertias[:self.n_joints].copy()
            
        return sample
    
    def record_loop(
        self,
        duration: float,
        sample_rate: float = 100.0,
        show_live: bool = True,
    ):
        """
        Main recording loop.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Samples per second
            show_live: Show live data in terminal
        """
        if self.driver is None:
            raise RuntimeError("Not connected")
        
        self.running = True
        self.recorded_data = []
        
        dt = 1.0 / sample_rate
        start_time = time.time()
        sample_count = 0
        
        # For acceleration computation
        prev_vel = None
        prev_time = None
        
        print(f"\n{'='*60}")
        print("RECORDING STARTED")
        print(f"Duration: {duration}s | Sample rate: {sample_rate} Hz")
        print("Move the joints slowly and deliberately!")
        print("Press Ctrl+C to stop early.")
        print(f"{'='*60}\n")
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                # Record sample with acceleration computation
                sample = self.record_sample(prev_vel=prev_vel, prev_time=prev_time)
                self.recorded_data.append(sample)
                sample_count += 1
                
                # Update previous values for next iteration
                prev_vel = sample['joint_velocities'].copy()
                prev_time = sample['timestamp']
                
                # Live display
                if show_live and sample_count % 10 == 0:
                    elapsed = time.time() - start_time
                    pos = sample['joint_positions']
                    vel = sample['joint_velocities']
                    
                    pos_str = ' '.join([f'{p:+6.2f}' for p in pos])
                    vel_str = ' '.join([f'{v:+6.2f}' for v in vel])
                    
                    print(f"\r[{elapsed:5.1f}s] Pos: [{pos_str}] | Vel: [{vel_str}]", end='')
                    
                    if 'urdf_rnea_torques' in sample:
                        rnea = sample['urdf_rnea_torques']
                        rnea_str = ' '.join([f'{t:+5.2f}' for t in rnea])
                        print(f" | RNEA: [{rnea_str}]", end='')
                    elif 'urdf_gravity_torques' in sample:
                        grav = sample['urdf_gravity_torques']
                        grav_str = ' '.join([f'{g:+5.2f}' for g in grav])
                        print(f" | Grav: [{grav_str}]", end='')
                
                # Maintain sample rate
                elapsed_loop = time.time() - loop_start
                if elapsed_loop < dt:
                    time.sleep(dt - elapsed_loop)
                    
        except KeyboardInterrupt:
            print("\n\nRecording stopped by user.")
        
        self.running = False
        print(f"\n\nRecorded {len(self.recorded_data)} samples.")
    
    def stop(self):
        """Stop the recording loop."""
        self.running = False
    
    def save_to_csv(self, output_path: str):
        """Save recorded data to CSV file."""
        if not self.recorded_data:
            print("No data to save!")
            return
        
        # Flatten the data for CSV
        rows = []
        for sample in self.recorded_data:
            row = {'timestamp': sample['timestamp']}
            
            for i, (pos, vel) in enumerate(zip(
                sample['joint_positions'],
                sample['joint_velocities']
            )):
                row[f'joint_{i+1}_pos'] = pos
                row[f'joint_{i+1}_vel'] = vel
                
            if 'urdf_gravity_torques' in sample:
                for i, g in enumerate(sample['urdf_gravity_torques']):
                    row[f'joint_{i+1}_urdf_gravity'] = g
                    
            if 'urdf_inertias' in sample:
                for i, inertia in enumerate(sample['urdf_inertias']):
                    row[f'joint_{i+1}_urdf_inertia'] = inertia
                    
            rows.append(row)
        
        # Write CSV
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Saved {len(rows)} samples to: {output_path}")
    
    def save_to_numpy(self, output_path: str):
        """Save recorded data to numpy .npz file."""
        if not self.recorded_data:
            print("No data to save!")
            return
        
        # Stack arrays
        timestamps = np.array([s['timestamp'] for s in self.recorded_data])
        positions = np.array([s['joint_positions'] for s in self.recorded_data])
        velocities = np.array([s['joint_velocities'] for s in self.recorded_data])
        accelerations = np.array([s['joint_accelerations'] for s in self.recorded_data])
        raw_positions = np.array([s['raw_positions'] for s in self.recorded_data])
        raw_velocities = np.array([s['raw_velocities'] for s in self.recorded_data])
        
        save_dict = {
            'timestamps': timestamps,
            'joint_positions': positions,
            'joint_velocities': velocities,
            'joint_accelerations': accelerations,
            'raw_positions': raw_positions,
            'raw_velocities': raw_velocities,
            'joint_offsets': self.joint_offsets,
            'joint_signs': self.joint_signs,
        }
        
        if 'urdf_gravity_torques' in self.recorded_data[0]:
            gravity = np.array([s['urdf_gravity_torques'] for s in self.recorded_data])
            save_dict['urdf_gravity_torques'] = gravity
        
        if 'urdf_coriolis_torques' in self.recorded_data[0]:
            coriolis = np.array([s['urdf_coriolis_torques'] for s in self.recorded_data])
            save_dict['urdf_coriolis_torques'] = coriolis
        
        if 'urdf_rnea_torques' in self.recorded_data[0]:
            rnea = np.array([s['urdf_rnea_torques'] for s in self.recorded_data])
            save_dict['urdf_rnea_torques'] = rnea
            
        if 'urdf_inertias' in self.recorded_data[0]:
            inertias = np.array([s['urdf_inertias'] for s in self.recorded_data])
            save_dict['urdf_inertias'] = inertias
        
        np.savez(output_path, **save_dict)
        print(f"Saved {len(self.recorded_data)} samples to: {output_path}")
    
    def print_summary(self):
        """Print a summary of the recorded data."""
        if not self.recorded_data:
            print("No data recorded!")
            return
        
        positions = np.array([s['joint_positions'] for s in self.recorded_data])
        velocities = np.array([s['joint_velocities'] for s in self.recorded_data])
        
        print(f"\n{'='*70}")
        print("RECORDING SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.recorded_data)}")
        duration = self.recorded_data[-1]['timestamp'] - self.recorded_data[0]['timestamp']
        print(f"Duration: {duration:.2f} seconds")
        print(f"Effective sample rate: {len(self.recorded_data)/duration:.1f} Hz")
        
        print(f"\n{'Joint Position Statistics (rad)':^70}")
        print("-" * 70)
        print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Range':>10} {'Mean':>10} {'Std':>10}")
        print("-" * 70)
        for i in range(self.n_joints):
            j_pos = positions[:, i]
            print(f"Joint {i+1:<2} {j_pos.min():>10.3f} {j_pos.max():>10.3f} "
                  f"{j_pos.max()-j_pos.min():>10.3f} {j_pos.mean():>10.3f} {j_pos.std():>10.3f}")
        
        print(f"\n{'Joint Velocity Statistics (rad/s)':^70}")
        print("-" * 70)
        print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Peak':>10}")
        print("-" * 70)
        for i in range(self.n_joints):
            j_vel = velocities[:, i]
            print(f"Joint {i+1:<2} {j_vel.min():>10.3f} {j_vel.max():>10.3f} "
                  f"{j_vel.mean():>10.3f} {j_vel.std():>10.3f} {np.abs(j_vel).max():>10.3f}")
        
        if 'urdf_gravity_torques' in self.recorded_data[0]:
            gravity = np.array([s['urdf_gravity_torques'] for s in self.recorded_data])
            print(f"\n{'URDF Gravity Torques Statistics (Nm)':^70}")
            print("-" * 70)
            print(f"{'Joint':<8} {'Min':>10} {'Max':>10} {'Range':>10} {'Mean':>10}")
            print("-" * 70)
            for i in range(min(self.n_joints, gravity.shape[1])):
                g = gravity[:, i]
                print(f"Joint {i+1:<2} {g.min():>10.3f} {g.max():>10.3f} "
                      f"{g.max()-g.min():>10.3f} {g.mean():>10.3f}")
        
        print(f"{'='*70}\n")


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from a YAML config file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dyn_config = config.get('dynamixel', {})
    arm_config = config.get('arm_teleop', {})
    
    # Get number of joints from servo_types or default to 6
    servo_types = dyn_config.get('servo_types', None)
    n_joints = len(servo_types) if servo_types else 6
    
    # For recording, we typically want arm joints only (exclude gripper)
    # Most configs have 7 servos (6 arm + 1 gripper)
    if n_joints == 7:
        n_joints = 6
        if servo_types:
            servo_types = servo_types[:6]
    
    joint_signs = dyn_config.get('joint_signs', [1] * n_joints)
    if len(joint_signs) > n_joints:
        joint_signs = joint_signs[:n_joints]
    
    return {
        'joint_ids': tuple(range(1, n_joints + 1)),
        'joint_offsets': tuple([0.0] * n_joints),  # Offsets are applied by the driver, not here
        'joint_signs': tuple(joint_signs),
        'baudrate': dyn_config.get('baudrate', 57600),
        'servo_types': tuple(servo_types) if servo_types else None,
        'port': dyn_config.get('dynamixel_port', dyn_config.get('port', None)),
        'urdf': arm_config.get('leader_urdf', None),
    }


def get_config_for_port(port: str) -> dict:
    """Get default configuration for known GELLO ports."""
    # Import configurations from gello_agent
    from gello.agents.gello_agent import PORT_CONFIG_MAP
    
    if port in PORT_CONFIG_MAP:
        config = PORT_CONFIG_MAP[port]
        return {
            'joint_ids': config.joint_ids[:-1] if config.gripper_config else config.joint_ids,
            'joint_offsets': config.joint_offsets[:-1] if config.gripper_config else config.joint_offsets,
            'joint_signs': config.joint_signs[:-1] if config.gripper_config else config.joint_signs,
            'baudrate': config.baudrate,
            'servo_types': config.servo_types,
        }
    
    # Default 6-DOF configuration with HIGH baudrate (most GELLO setups use 4000000)
    return {
        'joint_ids': (1, 2, 3, 4, 5, 6),
        'joint_offsets': (0, 0, 0, 0, 0, 0),
        'joint_signs': (1, 1, 1, 1, 1, 1),
        'baudrate': 4000000,  # Most GELLO setups use 4M baud
        'servo_types': None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Record physical behavior of GELLO leader for URDF validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Use a config file (automatically gets baudrate, servo types, etc.)
  python scripts/record_physical_behavior.py \\
      --config configs/ur5e_gello_factr_sim.yaml \\
      --duration 60

  # Record for 60 seconds with URDF comparison
  python scripts/record_physical_behavior.py \\
      --urdf gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf \\
      --port /dev/ttyDXL_gello \\
      --baudrate 4000000 \\
      --duration 60

  # Record without URDF (positions/velocities only)
  python scripts/record_physical_behavior.py \\
      --port /dev/ttyUSB0 \\
      --duration 30

Tips for recording:
  1. Move ONE joint at a time, slowly
  2. Move through the full range of motion
  3. Pause at different positions to capture static gravity
  4. Repeat movements to get consistent data
        """
    )
    
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file (recommended - sets baudrate, servo types, etc.)'
    )
    
    parser.add_argument(
        '--urdf', type=str, default=None,
        help='Path to URDF file for dynamics comparison'
    )
    parser.add_argument(
        '--port', type=str, default=None,
        help='Serial port for Dynamixel communication (required if no --config)'
    )
    parser.add_argument(
        '--duration', type=float, default=30.0,
        help='Recording duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--sample-rate', type=float, default=100.0,
        help='Sample rate in Hz (default: 100)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--format', type=str, choices=['csv', 'npz', 'both'], default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--joint-ids', type=int, nargs='+', default=None,
        help='Joint motor IDs (default: auto-detect from port)'
    )
    parser.add_argument(
        '--joint-offsets', type=float, nargs='+', default=None,
        help='Joint offsets in radians'
    )
    parser.add_argument(
        '--joint-signs', type=int, nargs='+', default=None,
        help='Joint signs (+1 or -1)'
    )
    parser.add_argument(
        '--baudrate', type=int, default=None,
        help='Serial baudrate (default: auto-detect from port)'
    )
    parser.add_argument(
        '--no-live', action='store_true',
        help='Disable live display during recording'
    )
    
    args = parser.parse_args()
    
    # Load config from YAML file if provided
    if args.config:
        project_root = Path(__file__).parent.parent
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = str(project_root / config_path)
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        print(f"Loading config from: {config_path}")
        config = load_config_from_yaml(config_path)
        
        # Use port from config if not specified on command line
        if args.port is None:
            args.port = config.get('port')
        
        # Use URDF from config if not specified on command line  
        if args.urdf is None:
            args.urdf = config.get('urdf')
    else:
        # Get configuration for this port (fallback)
        if args.port is None:
            print("Error: Either --config or --port must be specified")
            sys.exit(1)
        config = get_config_for_port(args.port)
    
    # Verify we have a port
    if args.port is None:
        print("Error: No port specified. Use --port or provide a config with dynamixel_port")
        sys.exit(1)
    
    # Override with command-line arguments
    if args.joint_ids is not None:
        config['joint_ids'] = tuple(args.joint_ids)
    if args.joint_offsets is not None:
        config['joint_offsets'] = tuple(args.joint_offsets)
    if args.joint_signs is not None:
        config['joint_signs'] = tuple(args.joint_signs)
    if args.baudrate is not None:
        config['baudrate'] = args.baudrate
    
    # Print configuration being used
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Port:        {args.port}")
    print(f"  Baudrate:    {config['baudrate']}")
    print(f"  Joint IDs:   {config['joint_ids']}")
    print(f"  Joint signs: {config['joint_signs']}")
    print(f"  Servo types: {config.get('servo_types', 'Not specified')}")
    print(f"  URDF:        {args.urdf or 'Not specified'}")
    print(f"{'='*60}\n")
    
    # Generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"physical_behavior_{timestamp}"
    else:
        base_name = Path(args.output).stem
    
    output_dir = Path('recordings')
    output_dir.mkdir(exist_ok=True)
    
    # Create recorder
    recorder = PhysicalBehaviorRecorder(
        urdf_path=args.urdf,
        port=args.port,
        joint_ids=config['joint_ids'],
        joint_offsets=config['joint_offsets'],
        joint_signs=config['joint_signs'],
        baudrate=config['baudrate'],
        servo_types=config.get('servo_types'),
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nStopping recording...")
        recorder.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Connect and record
    if not recorder.connect():
        print("Failed to connect to GELLO. Exiting.")
        sys.exit(1)
    
    try:
        recorder.record_loop(
            duration=args.duration,
            sample_rate=args.sample_rate,
            show_live=not args.no_live,
        )
        
        # Print summary
        recorder.print_summary()
        
        # Save data
        if args.format in ['csv', 'both']:
            csv_path = output_dir / f"{base_name}.csv"
            recorder.save_to_csv(str(csv_path))
            
        if args.format in ['npz', 'both']:
            npz_path = output_dir / f"{base_name}.npz"
            recorder.save_to_numpy(str(npz_path))
            
    finally:
        recorder.disconnect()
    
    print("\nDone! You can now analyze the recorded data to compare with your URDF.")
    print("\nSuggested next steps:")
    print("  1. Plot joint positions over time to verify joint signs/offsets")
    print("  2. Compare recorded gravity torques at static positions with URDF predictions")
    print("  3. Look for hysteresis/friction effects in velocity data")


if __name__ == '__main__':
    main()
