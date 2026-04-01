"""Leader-side minimalist torque estimator for external torque detection.

Estimates external joint torques from Dynamixel motor current (or load)
readings using the quasi-static motor torque model from Shi et al.,
"Minimalist Compliance Control", 2026.

This provides a simpler alternative to the full DOB (Yamane et al.) for
detecting human intervention on the GELLO leader device. The key
assumption is quasi-static operation (negligible inertia and Coriolis),
which holds well for slow human corrections on a leader device.

References:
    - Shi et al., "Minimalist Compliance Control", 2026.

      Equations 2-10 (motor torque model, wrench estimation).
    - Dynamixel XM/XH series control table for Present Current.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import pinocchio as pin


class MotorType(Enum):
    """Dynamixel motor type, determines how torque is estimated."""
    CURRENT = "current"   # XM/XH series: reads Present Current (mA)
    LOAD = "load"         # XL/AX series: reads Present Load (% of stall)


@dataclass
class MotorParams:
    """Per-joint motor parameters for torque estimation.

    These must be calibrated or taken from the Dynamixel datasheet.
    All arrays have shape (n_joints,).

    Attributes:
        kt:           Torque constant [Nm/A] per motor.
                      For Dynamixel XM430-W350: ~1.78 Nm/A (from datasheet
                      stall torque / stall current). Calibrate if possible.
        gear_ratio:   Output-side gear ratio per joint. For Dynamixel
                      XM430-W350: 353.5:1.
        eta:          Motor/transmission efficiency η ∈ (0, 1].
                      Accounts for friction and transmission losses.
                      Shi et al. use direction-dependent application:
                        forward drive:  τ_load = η * Kt * Iw
                        backward drive: τ_load = (1/η) * Kt * Iw
        motor_type:   Whether the servo provides current or load readings.
        stall_torque: Stall torque [Nm] per motor (only needed for LOAD type,
                      to convert Present Load percentage to Nm).
    """
    kt: np.ndarray
    gear_ratio: np.ndarray
    eta: np.ndarray
    motor_type: MotorType = MotorType.CURRENT
    stall_torque: np.ndarray | None = None  # Only for LOAD type


@dataclass
class MinimalistEstimatorConfig:
    """Configuration for the minimalist torque estimator.

    Attributes:
        urdf_path:     Path to the GELLO leader URDF for gravity computation.
        motor_params:  Motor parameters (see MotorParams).
        alpha_ema:     EMA filter coefficient ∈ (0, 1].
                       Higher α → more responsive but noisier.
                       Lower α → smoother but more latent.
                       Shi et al. use 0.9 for servos, 0.1 for QDD.
                       For GELLO intervention detection, start with 0.5.
        vel_threshold: Velocity threshold [rad/s] for drive direction
                       debouncing (Shi et al. Eq. 6, ε_vel).
                       Below this threshold, the previous drive direction
                       is held to avoid chattering.
    """
    urdf_path: str
    motor_params: MotorParams
    alpha_ema: float = 0.5
    vel_threshold: float = 0.05


class MinimalistTorqueEstimator:
    """Estimate external joint torques from Dynamixel current/load readings.

    Implements Shi et al.'s motor torque model (Eq. 2-8) with
    direction-dependent efficiency and EMA filtering.

    The estimation pipeline per joint is:
        1. Convert raw motor reading to winding current Iw [A]
        2. Compute motor torque: τ_motor = Kt * Iw
        3. Apply direction-dependent efficiency (Eq. 6-7):

             forward drive:  τ_load = η * τ_motor
             backward drive: τ_load = η⁻¹ * τ_motor
        4. Scale by gear ratio: τ_output = r * τ_load
        5. Subtract gravity: τ_ext = -(τ_output - τ_grav)   (Eq. 8)
        6. Apply EMA filter for noise reduction

    Note: This assumes quasi-static operation (Shi et al. Section III-B):
        "We assume a quasi-static interaction regime, in which inertial
         and velocity-dependent terms [...] are negligible compared to
         gravity and externally applied torques."
    This is a reasonable assumption for the GELLO leader during
    human corrections, which are typically slow.
    """

    def __init__(self, config: MinimalistEstimatorConfig):
        """
        Args:
            config: Estimator configuration (see MinimalistEstimatorConfig).
        """
        self.config = config

        # --- Pinocchio model (for gravity computation only) ---
        self.model = pin.buildModelFromUrdf(config.urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        # --- Motor parameters (copy to avoid external mutation) ---
        mp = config.motor_params
        self.kt = np.array(mp.kt, dtype=float)
        self.gear_ratio = np.array(mp.gear_ratio, dtype=float)
        self.eta = np.array(mp.eta, dtype=float)
        self.motor_type = mp.motor_type
        self.stall_torque = (
            np.array(mp.stall_torque, dtype=float)
            if mp.stall_torque is not None
            else None
        )

        # Validate
        n_active = min(len(self.kt), self.nv)
        self._n_active = n_active
        assert len(self.gear_ratio) >= n_active
        assert len(self.eta) >= n_active
        if self.motor_type == MotorType.LOAD:
            assert self.stall_torque is not None, (
                "stall_torque must be provided for LOAD-type motors"
            )

        # --- Filter state ---
        self.alpha = float(config.alpha_ema)
        self.vel_threshold = float(config.vel_threshold)

        # Drive direction state: +1 = forward, -1 = backward
        # Initialized to forward (Shi et al.: d_prev = 1)
        self.d_prev = np.ones(n_active)

        # EMA-filtered external torque estimate
        self.tau_ext_filtered = np.zeros(n_active)

    def _pad_q(self, q: np.ndarray) -> np.ndarray:
        """Pad joint positions to model size."""
        out = np.zeros(self.nq)
        n = min(len(q), self.nq)
        out[:n] = q[:n]
        return out

    def reset(self) -> None:
        """Reset all filter states."""
        self.d_prev = np.ones(self._n_active)
        self.tau_ext_filtered = np.zeros(self._n_active)

    def update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        motor_reading: np.ndarray,
    ) -> np.ndarray:
        """Run one estimation step.

        Args:
            q:              Joint positions [rad] from encoders.
            dq:             Joint velocities [rad/s].
                            Used ONLY for determining drive direction
                            (forward vs. backward), not for dynamics.
                            Can be from Dynamixel's internal differentiation
                            or from the Yamane VOB — either works here since
                            only the sign matters.
            motor_reading:  Raw motor reading per joint.
                            - CURRENT type: Present Current [mA] from Dynamixel

                              (signed: positive = CCW torque convention)
                            - LOAD type: Present Load [%] from Dynamixel

                              (signed: positive = CCW)

        Returns:
            tau_ext_hat: Estimated external joint torques [Nm],
                         sliced to active joint count. Positive = torque
                         applied by human in positive joint direction.
        """
        n = self._n_active

        # Slice inputs to active joints
        q_active = q[:n]
        dq_active = dq[:n]
        reading = motor_reading[:n].astype(float)

        # =====================================================================
        # 1. Convert raw reading to winding current Iw [A]
        #    Shi et al. Eq. 5: τ_load = η * Kt * Iw  (for forward drive)
        # =====================================================================
        if self.motor_type == MotorType.CURRENT:
            # Dynamixel Present Current is in mA
            # Convert to Amps
            iw = reading / 1000.0  # [A]
        elif self.motor_type == MotorType.LOAD:
            # Dynamixel Present Load is a percentage of stall torque.
            # We back-compute an equivalent current:
            #   load_pct * stall_torque = Kt * Iw * gear_ratio
            #   => Iw = (load_pct / 100) * stall_torque / (Kt * gear_ratio)
            # But it's simpler to go directly to τ_output:
            #   τ_output = (load_pct / 100) * stall_torque
            # and skip the Kt * Iw * gear_ratio path.
            # We handle this below with a flag.
            iw = None  # Will use direct path
        else:
            raise ValueError(f"Unknown motor type: {self.motor_type}")

        # =====================================================================
        # 2. Compute motor torque and apply direction-dependent efficiency
        #    Shi et al. Eq. 6-7
        # =====================================================================
        if self.motor_type == MotorType.CURRENT:
            # Motor torque before transmission
            tau_motor = self.kt[:n] * iw  # [Nm] at motor shaft

            # Determine drive direction (Shi et al. Eq. 6)
            # d = sign(τ_w * dq) when |dq| > ε_vel, else d_prev
            power_flow = tau_motor * dq_active
            fast_enough = np.abs(dq_active) > self.vel_threshold

            d = np.where(fast_enough, np.sign(power_flow), self.d_prev)
            # Handle zero power flow when fast enough: default to forward
            d = np.where(d == 0, self.d_prev, d)
            self.d_prev = d.copy()

            # Direction-dependent efficiency (Shi et al. Eq. 7)
            # Forward drive (d > 0): τ_load = η * Kt * Iw
            # Backward drive (d ≤ 0): τ_load = η⁻¹ * Kt * Iw
            eta_effective = np.where(d > 0, self.eta[:n], 1.0 / self.eta[:n])
            tau_load = eta_effective * tau_motor

            # Output torque after gear reduction (Shi et al. Eq. 8 prefix)
            tau_output = self.gear_ratio[:n] * tau_load

        elif self.motor_type == MotorType.LOAD:
            # Direct path: Present Load gives output torque directly
            # as a percentage of stall torque (already after gearbox)
            tau_output = (reading / 100.0) * self.stall_torque[:n]

            # Still apply direction-dependent efficiency correction
            # since Present Load may not perfectly account for
            # transmission asymmetry
            power_flow = tau_output * dq_active
            fast_enough = np.abs(dq_active) > self.vel_threshold
            d = np.where(fast_enough, np.sign(power_flow), self.d_prev)
            d = np.where(d == 0, self.d_prev, d)
            self.d_prev = d.copy()

            # For LOAD type, the efficiency correction is smaller
            # since the firmware partially accounts for it.
            # Apply a milder correction:
            eta_correction = np.where(d > 0, 1.0, 1.0 / (self.eta[:n] ** 2))
            tau_output = tau_output * eta_correction

        # =====================================================================
        # 3. Gravity compensation (Shi et al. Eq. 8)
        #    τ_ext = -(r · τ_load - τ_grav)
        #    Since τ_output = r · τ_load, this becomes:
        #    τ_ext = -(τ_output - τ_grav)
        # =====================================================================
        q_full = self._pad_q(q)
        tau_grav = pin.computeGeneralizedGravity(
            self.model, self.data, q_full
        )
        tau_grav_active = tau_grav[:n]

        tau_ext_raw = -(tau_output - tau_grav_active)

        # =====================================================================
        # 4. EMA filter (exponential moving average)
        #    Shi et al. use this for noise reduction.
        #    y[k] = α * x[k] + (1 - α) * y[k-1]
        # =====================================================================
        self.tau_ext_filtered = (
            self.alpha * tau_ext_raw
            + (1.0 - self.alpha) * self.tau_ext_filtered

        )

        return self.tau_ext_filtered.copy()

    def get_raw_motor_torque(
        self, dq: np.ndarray, motor_reading: np.ndarray
    ) -> np.ndarray:
        """Return the output torque τ_output (before gravity subtraction).

        Useful for debugging and calibration: compare this against
        the gravity torque at known static configurations.

        Args:
            dq:             Joint velocities [rad/s] (for drive direction).
            motor_reading:  Raw motor reading (same as update()).

        Returns:
            tau_output: Motor output torque [Nm] per active joint.
        """
        n = self._n_active
        reading = motor_reading[:n].astype(float)
        dq_active = dq[:n]

        if self.motor_type == MotorType.CURRENT:
            iw = reading / 1000.0
            tau_motor = self.kt[:n] * iw
            power_flow = tau_motor * dq_active
            fast_enough = np.abs(dq_active) > self.vel_threshold
            d = np.where(fast_enough, np.sign(power_flow), self.d_prev)
            d = np.where(d == 0, 1.0, d)
            eta_effective = np.where(d > 0, self.eta[:n], 1.0 / self.eta[:n])
            tau_load = eta_effective * tau_motor
            return self.gear_ratio[:n] * tau_load
        elif self.motor_type == MotorType.LOAD:
            return (reading / 100.0) * self.stall_torque[:n]
        else:
            raise ValueError(f"Unknown motor type: {self.motor_type}")


# =============================================================================

# Convenience: Intervention detector using the estimator

# =============================================================================

class InterventionDetector:
    """Detect human intervention on the GELLO leader using torque estimates.

    Wraps either a MinimalistTorqueEstimator or a LeaderObserver (Yamane)
    and applies a simple threshold + debounce logic to produce a binary
    intervention signal.

    This can be compared against the ground-truth button signal
    to evaluate detection accuracy (precision, recall, latency).
    """

    def __init__(
        self,
        n_joints: int,
        torque_threshold: float = 0.3,
        activation_count: int = 3,
        deactivation_count: int = 5,
    ):
        """
        Args:
            n_joints:           Number of active joints.
            torque_threshold:   Minimum absolute torque on any joint to
                                consider as potential intervention [Nm].
            activation_count:   Number of consecutive above-threshold
                                readings required to trigger intervention.
            deactivation_count: Number of consecutive below-threshold
                                readings required to clear intervention.
        """
        self.n_joints = n_joints
        self.threshold = torque_threshold
        self.act_count = activation_count
        self.deact_count = deactivation_count

        # State
        self.is_intervening = False
        self._above_counter = 0
        self._below_counter = 0

    def reset(self) -> None:
        """Reset detector state."""
        self.is_intervening = False
        self._above_counter = 0
        self._below_counter = 0

    def update(self, tau_ext_hat: np.ndarray) -> bool:
        """Update intervention state given new torque estimate.

        Uses a simple hysteresis (Schmitt trigger) logic:
        - Activate after `activation_count` consecutive above-threshold steps
        - Deactivate after `deactivation_count` consecutive below-threshold steps

        Args:
            tau_ext_hat: Estimated external torques [Nm], shape (n_joints,).

        Returns:
            is_intervening: True if human intervention is currently detected.
        """
        max_torque = np.max(np.abs(tau_ext_hat[: self.n_joints]))

        if max_torque > self.threshold:
            self._above_counter += 1
            self._below_counter = 0
        else:
            self._below_counter += 1
            self._above_counter = 0

        if not self.is_intervening:
            if self._above_counter >= self.act_count:
                self.is_intervening = True
        else:
            if self._below_counter >= self.deact_count:
                self.is_intervening = False

        return self.is_intervening


# =============================================================================

# Smoke test / example usage

# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example: GELLO with Dynamixel XM430-W350 servos
    # -------------------------------------------------------------------------
    urdf_file = (
        "gello/factr/urdf/GELLO_Assembly_URDF_V6/robot.urdf"
    )
    n_joints = 6

    # Motor parameters for XM430-W350
    # Datasheet: stall torque 4.1 Nm @ 12V, stall current 2.3A
    # => Kt ≈ 4.1 / (2.3 * 353.5) ≈ 0.00504 Nm/A at motor shaft
    # But Dynamixel's "Present Current" is after the current controller
    # and reports the winding current. The relationship is:
    #   output_torque = Kt_effective * I_mA / 1000
    # where Kt_effective ≈ stall_torque / stall_current = 4.1/2.3 ≈ 1.783
    # This Kt_effective already includes the gear ratio implicitly
    # if we set gear_ratio = 1.
    #
    # HOWEVER, for the Shi et al. formulation to be correct, we should
    # separate motor-side and output-side:
    #   τ_motor = Kt * Iw  (motor shaft)
    #   τ_output = gear_ratio * η * τ_motor  (output shaft, forward drive)
    #
    # For XM430-W350:
    #   Kt (motor-side) ≈ stall_torque / (stall_current * gear_ratio * η)
    # But since we don't know η exactly, it's easier to calibrate
    # Kt_effective = Kt * η as a lumped parameter:
    #   τ_output = gear_ratio * Kt_effective * Iw  (forward drive)
    #
    # Approach: use Kt_effective and set η to capture ONLY the
    # forward/backward asymmetry, not the absolute efficiency.

    # Option A: If you have current readings (XM/XH series)
    motor_params_current = MotorParams(
        kt=np.full(n_joints, 0.00504),       # Nm/A at motor shaft (calibrate!)
        gear_ratio=np.full(n_joints, 353.5),  # XM430-W350 gear ratio
        eta=np.full(n_joints, 0.65),          # Typical for high-ratio gearbox
        motor_type=MotorType.CURRENT,
    )

    # Option B: If you only have load readings (XL/AX series)
    motor_params_load = MotorParams(
        kt=np.full(n_joints, 1.0),            # Unused for LOAD type
        gear_ratio=np.full(n_joints, 1.0),    # Unused for LOAD type
        eta=np.full(n_joints, 0.65),
        motor_type=MotorType.LOAD,
        stall_torque=np.full(n_joints, 4.1),  # Nm (from datasheet)
    )

    # --- Create estimator (using current-based for this example) ---
    config = MinimalistEstimatorConfig(
        urdf_path=urdf_file,
        motor_params=motor_params_current,
        alpha_ema=0.5,
        vel_threshold=0.05,
    )
    estimator = MinimalistTorqueEstimator(config)

    # --- Create intervention detector ---
    detector = InterventionDetector(
        n_joints=n_joints,
        torque_threshold=0.3,  # Nm — tune based on GELLO noise floor
        activation_count=3,    # ~9 ms at 330 Hz
        deactivation_count=5,  # ~15 ms at 330 Hz
    )

    # --- Simulate a few steps ---
    q0 = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    dq = np.zeros(n_joints)

    print("=" * 60)
    print("Minimalist Torque Estimator (Shi et al.) — Smoke Test")
    print("=" * 60)

    # Step 1: At rest, zero current → should estimate ~0 external torque
    # (but gravity compensation means τ_output ≈ τ_grav, so τ_ext ≈ 0)
    current_mA = np.zeros(n_joints)
    tau_ext = estimator.update(q0, dq, current_mA)
    is_intervening = detector.update(tau_ext)
    print(f"\nStep 1 (rest, zero current):")
    print(f"  tau_ext_hat:    {tau_ext}")
    print(f"  intervening:    {is_intervening}")
    print(f"  (Note: non-zero because zero current ≠ gravity compensation)")

    # Step 2: Simulate gravity-compensating current
    # At static equilibrium: τ_output = τ_grav
    # => gear_ratio * η * Kt * Iw = τ_grav
    # => Iw = τ_grav / (gear_ratio * η * Kt)
    q_full = np.zeros(estimator.nq)
    q_full[:n_joints] = q0
    tau_grav = pin.computeGeneralizedGravity(
        estimator.model, estimator.data, q_full
    )[:n_joints]

    # Back-compute the current that would produce this gravity torque
    # (forward drive assumed)
    i_grav_A = tau_grav / (
        estimator.gear_ratio[:n_joints]
        * estimator.eta[:n_joints]
        * estimator.kt[:n_joints]

    )
    i_grav_mA = i_grav_A * 1000.0

    estimator.reset()
    detector.reset()
    for _ in range(20):
        tau_ext = estimator.update(q0, dq, i_grav_mA)
        is_intervening = detector.update(tau_ext)

    print(f"\nStep 2 (gravity-compensating current, 20 steps):")
    print(f"  gravity torque: {tau_grav}")
    print(f"  current [mA]:   {i_grav_mA}")
    print(f"  tau_ext_hat:    {tau_ext}")
    print(f"  intervening:    {is_intervening}")
    print(f"  (Should be ~0 since motor torque matches gravity)")

    # Step 3: Simulate human pushing joint 1 with ~0.5 Nm
    # The current increases beyond what's needed for gravity
    human_torque = np.zeros(n_joints)
    human_torque[1] = 0.5  # Nm on joint 1

    # Motor must now produce τ_grav + human_torque reaction
    # (In reality the servo fights the human push, so current increases)
    i_pushed_A = (tau_grav + human_torque) / (
        estimator.gear_ratio[:n_joints]
        * estimator.eta[:n_joints]
        * estimator.kt[:n_joints]

    )
    i_pushed_mA = i_pushed_A * 1000.0

    estimator.reset()
    detector.reset()
    for _ in range(20):
        tau_ext = estimator.update(q0, dq, i_pushed_mA)
        is_intervening = detector.update(tau_ext)

    print(f"\nStep 3 (human pushing joint 1 with 0.5 Nm, 20 steps):")
    print(f"  current [mA]:   {i_pushed_mA}")
    print(f"  tau_ext_hat:    {tau_ext}")
    print(f"  intervening:    {is_intervening}")
    print(f"  (Joint 1 should show ~-0.5 Nm, others ~0)")

    # -------------------------------------------------------------------------
    # Calibration helper: print expected current at a known configuration
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Calibration Helper")
    print(f"{'=' * 60}")
    print(
        "\nTo calibrate Kt and η, hold the GELLO at a known static"
        "\nconfiguration and record the Dynamixel Present Current."
        "\nThe expected gravity torque at each joint is:"
    )
    for i in range(n_joints):
        print(f"  Joint {i}: τ_grav = {tau_grav[i]:+.4f} Nm")
    print(
        "\nCompare: τ_output = gear_ratio * η * Kt * (I_mA / 1000)"
        "\nAdjust Kt and η until τ_output ≈ τ_grav at multiple configurations."
    )