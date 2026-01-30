# Example Configuration Files

This directory contains pre-configured YAML files for different experimental scenarios.

## Quick Start

Copy the appropriate config to your main configs directory and adjust parameters:

```bash
# For system validation (Section 4.2)
cp configs/examples/gravity_comp_validation.yaml configs/my_validation.yaml

# For contact comparison (Section 4.3.3)
cp configs/examples/position_mode_test.yaml configs/my_position.yaml
cp configs/examples/impedance_mode_test.yaml configs/my_impedance.yaml

# For force feedback tests (Section 4.3.2, 4.3.4)
cp configs/examples/with_force_feedback.yaml configs/my_with_fb.yaml
cp configs/examples/without_force_feedback.yaml configs/my_no_fb.yaml
```

## Configuration Files

### gravity_comp_validation.yaml

**Purpose:** Friction identification, dynamics validation, gravity hold test

**Key settings:**
- `teleop.enable: false` - No follower robot needed
- `controller.frequency: 500` - High sample rate
- `logging.enable: true` - Data logging active

**Used for:**
- Plot 1: Friction Identification (Section 4.2.1)
- Plot 2: Dynamics Model Validation (Section 4.2.1)
- Plot 3: Filter Comparison (Section 4.2.2)
- Plot 4: Gravity Compensation Validation (Section 4.3.1)

---

### position_mode_test.yaml

**Purpose:** Contact behavior test with rigid position control

**Key settings:**
- `teleop.enable: true`
- `teleop.impedance.enable: false` - Position control
- `controller.torque_feedback.enable: false`

**Used for:**
- Plot 6: Contact Behavior Comparison (Section 4.3.3, Test A)

---

### impedance_mode_test.yaml

**Purpose:** Contact behavior test with compliant impedance control

**Key settings:**
- `teleop.enable: true`
- `teleop.impedance.enable: true` - Impedance control
- `teleop.impedance.kp: 150.0` - Moderate stiffness
- `controller.torque_feedback.enable: false`

**Used for:**
- Plot 6: Contact Behavior Comparison (Section 4.3.3, Test B)

---

### with_force_feedback.yaml

**Purpose:** Teleoperation with haptic force feedback

**Key settings:**
- `teleop.enable: true`
- `teleop.impedance.enable: true`
- `controller.torque_feedback.enable: true` - Force feedback ON
- `controller.torque_feedback.gain: -2.0` - Resistive feedback

**Used for:**
- Plot 5: Force-Torque Correlation (Section 4.3.2)
- Plot 7: Teleoperation Statistics (Section 4.3.4, With Feedback)

---

### without_force_feedback.yaml

**Purpose:** Teleoperation without haptic force feedback (baseline)

**Key settings:**
- `teleop.enable: true`
- `teleop.impedance.enable: true`
- `controller.torque_feedback.enable: false` - Force feedback OFF

**Used for:**
- Plot 7: Teleoperation Statistics (Section 4.3.4, Without Feedback)

---

## Customization Checklist

Before using these configs, verify/adjust:

### Hardware-specific settings:

1. **Dynamixel port:**
   ```yaml
   dynamixel:
     dynamixel_port: "/dev/ttyDXL_gello"  # Check with: ls /dev/serial/by-id/
   ```

2. **Servo types:**
   ```yaml
   servo_types: [
     "XC330_T288_T", "XM430_W350_T", "XM430_W350_T",
     "XC330_T288_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T"
   ]
   ```

3. **Robot IP (for teleop configs):**
   ```yaml
   teleop:
     robot:
       robot_ip: "192.168.1.11"  # Your UR5e IP address
   ```

### Calibrated settings:

4. **Gravity compensation gains:**
   ```yaml
   controller:
     gravity_comp:
       gain: 0.9  # Global multiplier
       gain_per_joint: [1.0, 1.3, 1.0, 1.5, 3.0, 2.5]  # Per-joint tuning
   ```

5. **Friction compensation:**
   ```yaml
   controller:
     static_friction_comp:
       friction_feedforward: [0.0, 0.002, 0.008, 0.002, 0.002, 0.001]
       viscous_friction: [0.003, 0.02, 0.001, 0.0, 0.0, 0.0]
   ```

6. **Teleop mapping:**
   ```yaml
   teleop:
     mapping:
       offsets: [4.7124, 4.7124, 3.1416, 1.4835, 3.1416, -0.1745]
       auto_align: true  # Recommended for easier startup
   ```

7. **Force feedback scaling:**
   ```yaml
   controller:
     torque_feedback:
       gain: -2.0  # Tune for comfortable haptic strength
       motor_scalar: 94.5652173913  # Motor gear ratio
   ```

## Validation

Test your config before experiments:

```bash
# Syntax check
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# Quick test (10 seconds)
timeout 10 python gello/factr/gravity_compensation.py -c configs/my_config.yaml --log

# Inspect logged data
python scripts/visualization/inspect_log.py logs/gello_data_*.csv
```

## Troubleshooting

**"Port not found":**
- Check: `ls /dev/serial/by-id/`
- Update `dynamixel_port` in config

**"Robot connection failed":**
- Ping robot: `ping 192.168.1.11`
- Check UR5e is in remote control mode
- Verify IP address in config

**Arm drifts during hold test:**
- Increase `controller.gravity_comp.gain` (try 0.95)
- Tune `gain_per_joint` for problematic joints
- Add `velocity_damping` (0.01-0.05)

**Force feedback too weak/strong:**
- Adjust `controller.torque_feedback.gain` (-1.0 to -4.0)
- Lower is weaker, higher (more negative) is stronger

**High frequency noise:**
- Check `robot.filter_type: "one_euro"`
- Tune `filter_min_cutoff` (lower = smoother)
- Tune `filter_beta` (lower = less lag)
