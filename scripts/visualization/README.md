# GELLO Data Visualization Scripts

This directory contains scientific visualization scripts for analyzing GELLO teleoperation data. All scripts generate publication-quality plots in SVG format suitable for LaTeX documents.

## Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Configuration Files

Pre-configured YAML files for each experiment are available in `configs/examples/`:

- `gravity_comp_validation.yaml` - For Section 4.2 tests (no follower robot)
- `position_mode_test.yaml` - Position control (Section 4.3.3, Test A)
- `impedance_mode_test.yaml` - Impedance control (Section 4.3.3, Test B)
- `with_force_feedback.yaml` - Force feedback enabled (Section 4.3.2, 4.3.4)
- `without_force_feedback.yaml` - No force feedback (Section 4.3.4 baseline)

See `configs/examples/README.md` for customization instructions.

## Quick Reference

For a copy-paste friendly command reference, run:

```bash
./scripts/visualization/quick_reference.sh
```

This displays all commands needed for each experiment in order.

## Experiment Overview

| Section | Plot | Config File | Duration | Follower | Key Metric |
|---------|------|-------------|----------|----------|------------|
| 4.2.1 | Friction Identification | `gravity_comp_validation.yaml` | 60-120s | No | Friction model fit |
| 4.2.1 | Dynamics Validation | `gravity_comp_validation.yaml` | 60-90s | No | RMSE < 0.3 Nm |
| 4.2.2 | Filter Comparison | (reuse friction data) | - | No | Latency vs. noise |
| 4.3.1 | Gravity Hold Test | `gravity_comp_validation.yaml` | 30-60s | No | Drift < 1° |
| 4.3.2 | Force-Torque Correlation | `with_force_feedback.yaml` | 20-40s | Yes | Correlation > 0.8 |
| 4.3.3 | Contact Behavior | `position_mode_test.yaml`<br>`impedance_mode_test.yaml` | 10-20s each | Yes | Force reduction 60-80% |
| 4.3.4 | Teleoperation Stats | `without_force_feedback.yaml`<br>`with_force_feedback.yaml` | 10 trials each | Yes | p-value < 0.05 |

## Data Logging

Enable data logging when running gravity compensation:

```bash
# With config file setting
python gello/factr/gravity_compensation.py --config configs/your_config.yaml --log

# Data is saved to logs/gello_data_YYYYMMDD_HHMMSS.csv
```

## Visualization Scripts

### 1. Friction Identification (Section 4.2.1)

Plots the friction characteristic curve showing Coulomb + viscous friction model.

```bash
python plot_friction_identification.py logs/gello_data_*.csv \
    --joint 2 \
    --output friction_identification.svg
```

**What to expect:**
- Scatter plot showing velocity vs. friction torque
- Clear hysteresis loop at zero velocity (stiction)
- Fitted model curve overlaid in red

### 2. Dynamics Model Validation (Section 4.2.1)

Validates URDF inertial parameters by showing residual errors.

```bash
python plot_dynamics_validation.py logs/gello_data_*.csv \
    --output dynamics_validation.svg
```

**What to expect:**
- 6 subplots (one per joint)
- Residual oscillating around zero indicates good model
- Constant offset indicates mass/CoM error in URDF

### 3. Filter Comparison (Section 4.2.2)

Demonstrates 1€ Filter advantages over EMA filtering.

```bash
python plot_filter_comparison.py logs/gello_data_*.csv \
    --joint 4 \
    --ema-alpha 0.1 \
    --output filter_comparison.svg
```

**What to expect:**
- Upper plot: Full comparison of raw, EMA, and 1€ filtered signals
- Lower plot: Zoomed view of motion onset
- 1€ Filter shows lower latency than EMA

### 4. Gravity Compensation Validation (Section 4.3.1)

Shows "weightless arm" behavior - positions should remain constant.

```bash
python plot_gravity_compensation.py logs/gello_data_*.csv \
    --joints 1 2 3 \
    --output gravity_comp_validation.svg
```

**What to expect:**
- Nearly horizontal lines (< 1° drift over 5 seconds)
- Excessive drift indicates insufficient compensation gain

### 5. Force-Torque Correlation (Section 4.3.2)

Demonstrates causal relationship between follower forces and leader feedback torques.

```bash
python plot_force_torque_correlation.py logs/gello_data_*.csv \
    --joint 2 \
    --tcp-axis 2 \
    --output force_torque_correlation.svg
```

**What to expect:**
- Dual y-axis plot (force in blue, torque in red)
- Curves should be similar in shape
- Minimal lag (< 10ms) indicates good feedback

### 6. Contact Behavior Comparison (Section 4.3.3)

Compares position control vs. impedance control safety.

```bash
python plot_contact_behavior.py \
    logs/position_mode.csv \
    logs/impedance_mode.csv \
    --output contact_behavior.svg
```

**What to expect:**
- Upper plot: Impedance yields on contact, position fights
- Lower plot: Impedance has much lower peak forces (safety!)
- Typical reduction: 60-80%

### 7. Teleoperation Statistics (Section 4.3.4)

Statistical comparison of task performance with/without force feedback.

```bash
# Option 1: From pre-computed summary CSV
python plot_teleoperation_statistics.py \
    --trials experiment_results.csv \
    --metric max_force \
    --output teleoperation_statistics.svg

# Option 2: From individual log files
python plot_teleoperation_statistics.py \
    --logs trial1_no_fb.csv trial1_with_fb.csv trial2_no_fb.csv trial2_with_fb.csv ... \
    --output teleoperation_statistics.svg
```

**What to expect:**
- Boxplot comparing two conditions
- With feedback should show:
  - Lower median force
  - Smaller variance (more consistent)
  - Statistical significance (p < 0.05)

## Typical Workflow

### For Section 4.2 (System Validation):

#### 4.2.1a: Friction Identification

**Goal:** Record slow, continuous motion to identify friction parameters.

**Config requirements:**
```yaml
controller:
  frequency: 500  # Higher frequency for better resolution
  gravity_comp:
    enable: true
    gain: 0.9
  static_friction_comp:
    enable_speed: 0.08  # Keep friction comp on during recording
teleop:
  enable: false  # No follower needed
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. Start with arm in neutral position
2. Run: `python gello/factr/gravity_compensation.py -c configs/your_config.yaml --log`
3. **Slowly** move each joint through its full range (one at a time):
   - Joint 2 (Shoulder): Move up and down, ~30 seconds
   - Joint 3 (Elbow): Extend and retract, ~30 seconds
   - Vary velocity: very slow → medium → very slow
4. Stop program (Ctrl+C)
5. Rename log: `mv logs/gello_data_*.csv logs/friction_data.csv`

**Expected log characteristics:**
- Duration: 60-120 seconds
- Sample rate: ~500 Hz
- Key columns: `q_dot_leader_*`, `tau_total_*`, `tau_gravity_*`
- Velocities should range from -0.5 to +0.5 rad/s

**Generate plot:**
```bash
python scripts/visualization/plot_friction_identification.py logs/friction_data.csv \
    --joint 2 \
    --filter-mode gravity_comp \
    --output plots/friction_identification.svg
```

---

#### 4.2.1b: Dynamics Model Validation

**Goal:** Validate URDF inertial parameters with quasi-static motion.

**Config requirements:**
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
    gain_per_joint: [1.0, 1.3, 1.0, 1.5, 3.0, 2.5]  # Use your tuned gains
  static_friction_comp:
    enable_speed: 0.08
teleop:
  enable: false
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. Start with arm in home position
2. Run: `python gello/factr/gravity_compensation.py -c configs/your_config.yaml --log`
3. **Smoothly** guide arm through workspace:
   - Move to 5-6 different poses (corners of workspace)
   - Hold each pose for ~2 seconds
   - Transition slowly between poses (~5 seconds per transition)
   - Cover full range of joint 2 and 3 (highest loads)
   - Avoid sudden accelerations (keep dq < 0.3 rad/s)
4. Total duration: ~60 seconds
5. Stop and rename: `mv logs/gello_data_*.csv logs/dynamics_validation.csv`

**Expected log characteristics:**
- Duration: 60-90 seconds
- Low velocities (max ~0.3 rad/s)
- All 6 joints should move through significant range
- `tau_total` ≈ `tau_gravity` (friction should be small)

**Generate plot:**
```bash
python scripts/visualization/plot_dynamics_validation.py logs/dynamics_validation.csv \
    --filter-mode gravity_comp \
    --output plots/dynamics_validation.svg
```

**Interpreting results:**
- RMSE < 0.1 Nm: Excellent model
- RMSE < 0.3 Nm: Good model
- Constant offset: Check mass/CoM in URDF
- Oscillating residual: Normal (unmodeled friction/dynamics)

---

#### 4.2.2: Filter Comparison

**Note:** Can reuse `friction_data.csv` from 4.2.1a.

**Generate plot:**
```bash
python scripts/visualization/plot_filter_comparison.py logs/friction_data.csv \
    --joint 4 \
    --ema-alpha 0.1 \
    --output plots/filter_comparison.svg
```

This plot is generated synthetically by applying different filters to raw velocity data.

---

### For Section 4.3 (Teleoperation Performance):

#### 4.3.1: Gravity Compensation Hold Test

**Goal:** Demonstrate "weightless arm" - position should not drift when released.

**Config requirements:**
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9  # Should be well-tuned from previous calibration
    gain_per_joint: [1.0, 1.3, 1.0, 1.5, 3.0, 2.5]
    velocity_damping: 0.01  # Small damping helps stability
  static_friction_comp:
    enable_speed: 0.08
    friction_feedforward: [0.0, 0.002, 0.008, 0.002, 0.002, 0.001]
    viscous_friction: [0.003, 0.02, 0.001, 0.0, 0.0, 0.0]
teleop:
  enable: false  # Pure gravity compensation test
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. Start logging: `python gello/factr/gravity_compensation.py -c configs/your_config.yaml --log`
2. Move arm to test pose (e.g., shoulder at 45°, elbow extended)
3. **Gently release arm** (don't push or pull)
4. **Do not touch arm for 5-10 seconds**
5. Repeat with 2-3 different poses (especially test Joint 2, 3)
6. Stop: Ctrl+C
7. Rename: `mv logs/gello_data_*.csv logs/gravity_hold.csv`

**Expected log characteristics:**
- Duration: 30-60 seconds (multiple hold tests)
- Velocity should be near zero during holds
- Position drift should be < 1° over 5 seconds

**Generate plot:**
```bash
python scripts/visualization/plot_gravity_compensation.py logs/gravity_hold.csv \
    --joints 1 2 3 \
    --time-window 10.0 15.0 \
    --output plots/gravity_compensation.svg
```

**Troubleshooting:**
- Arm sinks down: Increase `controller.gravity_comp.gain` or adjust `gain_per_joint`
- Arm floats up: Decrease gain
- Oscillations: Add `velocity_damping` (0.01-0.05)

---

#### 4.3.2: Force-Torque Correlation Test

**Goal:** Show that TCP forces at follower create haptic feedback at leader.

**Config requirements:**
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
  torque_feedback:
    enable: true
    gain: -2.0  # Negative for resistive feedback
    damping: 0.0
    motor_scalar: 94.5652173913
teleop:
  enable: true
  use_direct_rtde: true
  hz: 250
  impedance:
    enable: true  # Use impedance for compliant contact
    kp: 150.0
    kd: 12.0
  robot:
    _target_: gello.robots.ur5e.URRobot
    robot_ip: "192.168.1.11"
    filter_type: "one_euro"
    filter_min_cutoff: 0.5
    filter_beta: 0.007
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. **Prepare:** Place a digital scale or force sensor under UR5e end-effector
2. Start: `python gello/factr/gravity_compensation.py -c configs/your_config.yaml --log`
3. Wait 2-3 seconds (no force)
4. **Slowly** press down on scale with UR5e using GELLO:
   - Ramp up force gradually over 3-4 seconds
   - Hold maximum force for 2-3 seconds
   - Release gradually over 3-4 seconds
5. **You should feel resistance in GELLO leader as force increases**
6. Repeat 2-3 times
7. Stop: Ctrl+C
8. Rename: `mv logs/gello_data_*.csv logs/force_feedback.csv`

**Expected log characteristics:**
- Duration: 20-40 seconds
- `tcp_force_2` (Fz) should show clear ramp up/hold/down pattern
- `tau_feedback_2` should mirror force pattern (negative correlation)
- Peak force: 10-50 N (safe range)

**Generate plot:**
```bash
python scripts/visualization/plot_force_torque_correlation.py logs/force_feedback.csv \
    --joint 2 \
    --tcp-axis 2 \
    --filter-mode impedance_teleop \
    --output plots/force_torque_correlation.svg
```

**Expected results:**
- Cross-correlation > 0.8 indicates good coupling
- Lag < 20ms indicates low latency
- Similar curve shapes confirm causal relationship

---

#### 4.3.3: Contact Behavior Comparison

**Goal:** Demonstrate safety advantage of impedance over position control.

**Test A: Position Control (Rigid)**

Config for position mode:
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
  torque_feedback:
    enable: false  # Disable feedback for clearer comparison
teleop:
  enable: true
  use_direct_rtde: true
  hz: 250
  impedance:
    enable: false  # POSITION MODE
  robot:
    _target_: gello.robots.ur5e.URRobot
    robot_ip: "192.168.1.11"
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. Start: `python gello/factr/gravity_compensation.py -c configs/position_mode.yaml --log`
2. Place hard obstacle (wooden block) on table below UR5e
3. Using GELLO, command UR5e to move **straight down** toward obstacle
4. **Continue pushing command for 1-2 seconds after contact**
5. Release and move away
6. Stop: Ctrl+C
7. Rename: `mv logs/gello_data_*.csv logs/position_contact.csv`

**Test B: Impedance Control (Compliant)**

Config for impedance mode:
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
  torque_feedback:
    enable: false
teleop:
  enable: true
  use_direct_rtde: true
  hz: 250
  impedance:
    enable: true  # IMPEDANCE MODE
    kp: 150.0  # Moderate stiffness
    kd: 12.0
    use_leader_velocity: true
  robot:
    _target_: gello.robots.ur5e.URRobot
    robot_ip: "192.168.1.11"
logging:
  enable: true
  log_dir: "logs"
```

**Procedure:**
1. Start: `python gello/factr/gravity_compensation.py -c configs/impedance_mode.yaml --log`
2. **Same obstacle, same motion as Test A**
3. Command UR5e straight down toward obstacle
4. Continue pushing for 1-2 seconds after contact
5. Stop: Ctrl+C
6. Rename: `mv logs/gello_data_*.csv logs/impedance_contact.csv`

**Generate plot:**
```bash
python scripts/visualization/plot_contact_behavior.py \
    logs/position_contact.csv \
    logs/impedance_contact.csv \
    --tcp-axis 2 \
    --follower-joint 2 \
    --output plots/contact_behavior.svg
```

**Expected results:**
- **Position mode:** High force spike (50-150 N), robot fights obstacle
- **Impedance mode:** Lower plateau force (10-40 N), robot yields
- Typical force reduction: 60-80%
- Position continues moving into obstacle, impedance stops/reverses

---

#### 4.3.4: Teleoperation Task Statistics

**Goal:** Statistical proof that force feedback improves task performance.

**Task suggestion:** "Peg-in-hole" or "vertical surface contact"

**Setup:**
- Prepare task (e.g., vertical peg, foam block)
- Define "trial" as one complete insertion/contact
- Need 10 trials WITHOUT feedback, 10 trials WITH feedback

**Config A: Without Force Feedback**
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
  torque_feedback:
    enable: false  # NO FEEDBACK
teleop:
  enable: true
  impedance:
    enable: true
    kp: 150.0
    kd: 12.0
  robot:
    _target_: gello.robots.ur5e.URRobot
    robot_ip: "192.168.1.11"
logging:
  enable: true
  log_dir: "logs"
```

**Procedure (Without Feedback):**
1. For each trial (i=1 to 10):
   - Start: `python gello/factr/gravity_compensation.py -c configs/no_feedback.yaml --log`
   - Perform task (e.g., insert peg)
   - Stop: Ctrl+C
   - Rename: `mv logs/gello_data_*.csv logs/trial_no_fb_${i}.csv`

**Config B: With Force Feedback**
```yaml
controller:
  frequency: 500
  gravity_comp:
    enable: true
    gain: 0.9
  torque_feedback:
    enable: true  # WITH FEEDBACK
    gain: -2.0
    damping: 0.0
    motor_scalar: 94.5652173913
teleop:
  enable: true
  impedance:
    enable: true
    kp: 150.0
    kd: 12.0
  robot:
    _target_: gello.robots.ur5e.URRobot
    robot_ip: "192.168.1.11"
logging:
  enable: true
  log_dir: "logs"
```

**Procedure (With Feedback):**
1. For each trial (i=1 to 10):
   - Start: `python gello/factr/gravity_compensation.py -c configs/with_feedback.yaml --log`
   - Perform same task
   - Stop: Ctrl+C
   - Rename: `mv logs/gello_data_*.csv logs/trial_with_fb_${i}.csv`

**Extract statistics:**
```bash
# Option 1: Use plotting script directly with log files
python scripts/visualization/plot_teleoperation_statistics.py \
    --logs logs/trial_no_fb_*.csv logs/trial_with_fb_*.csv \
    --metric max_force \
    --tcp-axis 2 \
    --output plots/teleoperation_statistics.svg

# Option 2: Create summary CSV manually
# Create logs/trials.csv:
# condition,max_force
# without_feedback,15.3
# without_feedback,16.8
# ...
# with_feedback,8.2
# with_feedback,9.1
# ...

python scripts/visualization/plot_teleoperation_statistics.py \
    --trials logs/trials.csv \
    --metric max_force \
    --output plots/teleoperation_statistics.svg
```

**Expected results:**
- With feedback: Lower median, smaller IQR (more consistent)
- Statistical significance: p < 0.05 (Mann-Whitney U test)
- Typical improvement: 30-50% reduction in peak forces

---

### Quick Validation Checklist

Before running experiments, verify your config:

```bash
# Check config syntax
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"

# Inspect existing log
python scripts/visualization/inspect_log.py logs/test.csv

# Test logging (10 seconds)
timeout 10 python gello/factr/gravity_compensation.py -c configs/your_config.yaml --log
python scripts/visualization/inspect_log.py logs/gello_data_*.csv
```

---

## LaTeX Integration

All plots are saved as SVG (vector graphics). Include in LaTeX:

```latex
\usepackage{svg}

\begin{figure}[htbp]
    \centering
    \includesvg[width=0.8\textwidth]{friction_identification.svg}
    \caption{Friction characteristic curve for Joint 2 (Shoulder). 
             The fitted model shows Coulomb friction of 0.452 Nm and 
             viscous coefficient of 0.086 Nm·s/rad.}
    \label{fig:friction_id}
\end{figure}
```

Alternatively, use the PNG preview for quick drafts:

```latex
\usepackage{graphicx}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{friction_identification.png}
    \caption{...}
\end{figure}
```

## Troubleshooting

**"No data for mode 'gravity_comp'"**
- Check that `control_mode` column exists in CSV
- Try removing `--filter-mode` argument

**"Could not extract force from log"**
- Verify that teleop was active during recording
- Check that `tcp_force_X` columns exist

**Plot looks noisy/messy**
- Use `--time-window` to zoom on specific region
- Increase filter window size in filter_comparison.py

**LaTeX doesn't compile with SVG**
- Ensure `\usepackage{svg}` is in preamble
- May need to install Inkscape for SVG conversion
- Fallback: Use PNG version

## Custom Analysis

All scripts can be modified for custom analysis. Key entry points:

- Modify `sns.set_context()` for font sizes
- Change colors via matplotlib color codes
- Add subplots for additional metrics
- Adjust statistical tests in teleoperation_statistics.py

## Data Format

All scripts expect CSV with these columns:

```
timestamp,control_mode,
q_leader_0,...,q_leader_5,
q_dot_leader_0,...,q_dot_leader_5,
tau_gravity_0,...,tau_gravity_5,
tau_friction_0,...,tau_friction_5,
tau_damping_0,...,tau_damping_5,
tau_null_0,...,tau_null_5,
tau_limit_0,...,tau_limit_5,
tau_feedback_0,...,tau_feedback_5,
tau_total_0,...,tau_total_5,
tau_external_0,...,tau_external_5,
gripper_pos_leader,gripper_vel_leader,
q_follower_0,...,q_follower_5,
q_dot_follower_0,...,q_dot_follower_5,
gripper_pos_follower,
tcp_force_0,...,tcp_force_5
```

(Automatically generated by HighFrequencyDataLogger)
