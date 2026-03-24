"""GELLO-specific minimalist compliance controller wrapper.

This script implements both the 6D Wrench Estimator and the Cartesian
Admittance Controller for a GELLO/UR5e teleoperation setup. It uses MuJoCo 
for dynamics and contact Jacobians, combining the dynamixel motor model
with the minimalist compliance logic.
"""

import mujoco
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Optional

# Import minimalist compliance utilities
from minimalist_compliance_control.wrench_estimation import (
    WrenchEstimateConfig,
    estimate_wrench,
)
from minimalist_compliance_control.compliance_ref import (
    COMMAND_LAYOUT,
    ComplianceReference,
    ComplianceState,
)

@dataclass
class GelloMotorParams:
    """Dynamixel motor parameters for the GELLO arm."""
    kt: npt.NDArray[np.float32]             # Torque constant [Nm/A] per motor
    gear_ratio: npt.NDArray[np.float32]     # Gear ratio per motor
    eta: npt.NDArray[np.float32]            # Efficiency per motor (0, 1]


@dataclass
class GelloComplianceConfig:
    urdf_path: str
    ee_site_name: str
    motor_params: GelloMotorParams
    estimate_config: WrenchEstimateConfig
    dt: float = 0.02
    mass: float = 1.0                       # Virtual mass for admittance
    inertia_diag: npt.NDArray[np.float32] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    vel_threshold: float = 0.05             # For direction-dependent efficiency
    ik_damping: float = 0.05
    ik_num_iter: int = 20


class GelloMinimalistCompliance:
    """Wrench observer and Admittance controller for GELLO setup."""

    def __init__(self, config: GelloComplianceConfig):
        self.config = config
        
        # Load MuJoCo model for dynamics & Jacobian
        self.model = mujoco.MjModel.from_xml_path(config.urdf_path)
        self.data = mujoco.MjData(self.model)
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nv  # Assuming fully actuated

        # Get Site ID for End-Effector
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Site '{config.ee_site_name}' not found in URDF. Please add a <site> at your end effector.")

        # Motor parameters
        self.mp = config.motor_params
        self.d_prev = np.ones(self.nv)

        # Basic identity mappings for joint <-> actuator
        # If your GELLO has direct 1:1 mapping in MuJoCo between joints and actuators:
        actuator_indices = np.arange(self.nu, dtype=np.int32)
        joint_indices = np.arange(self.nv, dtype=np.int32)
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.nv)]
        
        # Build Reference Controller (Admittance/Impedance + IK)
        self.ref_ctrl = ComplianceReference(
            dt=config.dt,
            model=self.model,
            site_names=[config.ee_site_name],
            actuator_indices=actuator_indices,
            joint_indices=joint_indices,
            joint_names=joint_names,
            joint_to_actuator_fn=lambda x: x,
            actuator_to_joint_fn=lambda x: x,
            default_motor_pos=np.zeros(self.nu, dtype=np.float32),
            default_qpos=np.zeros(self.nq, dtype=np.float32),
            mass=config.mass,
            inertia_diag=config.inertia_diag
        )
        # Tweak IK params if needed
        self.ref_ctrl.mink_num_iter = config.ik_num_iter
        self.ref_ctrl.mink_damping = config.ik_damping
        
        self.compliance_state = self.ref_ctrl.get_default_state()

    def estimate_external_wrench(
        self, 
        q: npt.NDArray[np.float32], 
        dq: npt.NDArray[np.float32], 
        current_mA: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Convert dynamixel currents + robot state into Cartesian External Wrench."""
        
        # 1. Update MuJoCo kinematics & dynamics
        self.data.qpos[:self.nq] = q[:self.nq]
        self.data.qvel[:self.nv] = dq[:self.nv]
        mujoco.mj_forward(self.model, self.data)
        
        # 2. Get Dynamixel motor torque (Direction-dependent efficiency)
        i_A = current_mA / 1000.0
        tau_act = self.mp.kt * i_A
        
        tau_load = np.zeros(self.nv, dtype=np.float32)
        for i in range(self.nv):
            # Direction debouncing
            if dq[i] > self.config.vel_threshold:
                dir_i = 1.0
            elif dq[i] < -self.config.vel_threshold:
                dir_i = -1.0
            else:
                dir_i = self.d_prev[i]
            self.d_prev[i] = dir_i

            # Efficiency applied bidirectionally
            if tau_act[i] * dir_i >= 0:
                tau_load[i] = self.mp.eta[i] * tau_act[i]
            else:
                tau_load[i] = (1.0 / self.mp.eta[i]) * tau_act[i]
                
        # Raw mapped torque applied at the joint output
        tau_raw_joint = self.mp.gear_ratio * tau_load
        
        # 3. Obtain Bias Torques (Gravity + Coriolis)
        tau_bias = self.data.qfrc_bias[:self.nv]
        
        # 4. Joint External Torque Residual
        tau_ext = -(tau_raw_joint - tau_bias)
        
        # 5. Extract Jacobians
        jacp = np.zeros((3, self.nv), dtype=np.float64)
        jacr = np.zeros((3, self.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        
        # 6. Wrench Estimation (Regularized pseudo-inverse)
        site_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        wrench = estimate_wrench(
            jacp.astype(np.float32),
            jacr.astype(np.float32),
            tau_ext.astype(np.float32),
            site_rot.astype(np.float32),
            self.config.estimate_config
        )
        return wrench

    def step(
        self, 
        q: npt.NDArray[np.float32], 
        dq: npt.NDArray[np.float32], 
        current_mA: npt.NDArray[np.float32],
        target_pos: npt.NDArray[np.float32],
        target_ori_rotvec: npt.NDArray[np.float32],
        kp_pos: npt.NDArray[np.float32] = np.array([100, 100, 100]),
        kd_pos: npt.NDArray[np.float32] = np.array([10, 10, 10]),
        kp_rot: npt.NDArray[np.float32] = np.array([10, 10, 10]),
        kd_rot: npt.NDArray[np.float32] = np.array([1, 1, 1])
    ) -> npt.NDArray[np.float32]:
        """
        Runs the observer and the Admittance controller pipeline.
        
        Returns:
            q_target (NDArray): Compliant joint targets to send to the motors.
        """
        
        # 1. Estimate 6D Cartesian Wrench
        wrench = self.estimate_external_wrench(q, dq, current_mA)
        
        # 2. Construct the layout matrix expected by `ComplianceReference.integrate_commands`
        command_matrix = np.zeros((1, COMMAND_LAYOUT.width), dtype=np.float32)
        
        # Assign targets
        command_matrix[0, COMMAND_LAYOUT.position] = target_pos
        command_matrix[0, COMMAND_LAYOUT.orientation] = target_ori_rotvec
        
        # Assign measured force from observer
        command_matrix[0, COMMAND_LAYOUT.measured_force] = wrench[:3]
        command_matrix[0, COMMAND_LAYOUT.measured_torque] = wrench[3:]
        
        # Impedance Controller Gains (Diagonal matrices)
        command_matrix[0, COMMAND_LAYOUT.kp_pos] = np.diag(kp_pos).flatten()
        command_matrix[0, COMMAND_LAYOUT.kd_pos] = np.diag(kd_pos).flatten()
        command_matrix[0, COMMAND_LAYOUT.kp_rot] = np.diag(kp_rot).flatten()
        command_matrix[0, COMMAND_LAYOUT.kd_rot] = np.diag(kd_rot).flatten()

        # 3. Step Compliance Reference (Admittance integration + IK execution)
        self.compliance_state = self.ref_ctrl.get_state_ref(
            command_matrix=command_matrix,
            last_state=self.compliance_state,
            data=self.data
        )

        # Returns the targeted joint angles optimized by Mink IK 
        return self.compliance_state.qpos


if __name__ == "__main__":
    # Smoke Test Example
    n_joints = 6
    motor_params = GelloMotorParams(
        kt=np.full(n_joints, 0.00504, dtype=np.float32),
        gear_ratio=np.full(n_joints, 353.5, dtype=np.float32),
        eta=np.full(n_joints, 0.65, dtype=np.float32)
    )
    
    # Must represent the URDF containing a site tagging the leader end effector
    # Note: Replace with your actual URDF. Must have a site named 'ee_site'.
    urdf_path = "your_gello_urdf_with_site.xml"  
    
    try:
        config = GelloComplianceConfig(
            urdf_path=urdf_path,
            ee_site_name="ee_site", # Ensure your URDF has a <site name="ee_site" .../>
            motor_params=motor_params,
            estimate_config=WrenchEstimateConfig(
                force_reg=1e-3, 
                torque_reg=1e-2, 
                axis_aligned=False
            )
        )
        
        controller = GelloMinimalistCompliance(config)
        
        # Dummy loop
        currents = np.zeros(n_joints, dtype=np.float32)
        q = np.array([0, -1.57, 0, -1.57, 0, 0], dtype=np.float32)
        dq = np.zeros(n_joints, dtype=np.float32)
        
        target_pos = np.array([0.5, 0.0, 0.5])
        target_rotvec = np.zeros(3) # Identity Orientation
        
        q_target = controller.step(q, dq, currents, target_pos, target_rotvec)
        print("Obtained output target positions:", q_target)
        
    except ValueError as e:
        print("Note: Ensure you have a valid URDF to run the script. Error:", e)
