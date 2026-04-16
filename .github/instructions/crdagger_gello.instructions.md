---
description: System instructions and project context for the Compliant Human-in-the-Loop Imitation Learning project (GELLO + UR5e).
applyTo: '*.py, *.yaml'
---

# Project Context: Compliant Human-in-the-Loop Imitation Learning (GELLO + UR5e)

This project implements a Compliant Human-in-the-Loop Imitation Learning framework based on the CR-DAgger paradigm. The system uses a low-cost, teleoperated leader arm (GELLO) to control an industrial follower robot (UR5e) for contact-rich manipulation tasks.

The core objective is to enable a human operator to seamlessly intervene and correct an autonomously running Deep Learning policy (mainly Action Chunking with Transformer (ACT)) via physical, compliant pushes on the GELLO leader, without using a physical clutch button. These interventions are recorded as spatial deviations (`delta_q`) and trained into a high-frequency Residual Policy.

## Hardware Environment & Physical Constraints
When writing or reviewing code, you MUST respect the following physical hardware realities:
* **Leader (GELLO):** Kinematically scaled 6-DoF arm of the UR5e using Dynamixel XM430-W350 and XC330-T288 servos.
    * *Constraint 1 (The Friction Wall):* Gear ratio is 353.5:1. Stiction is massive. 
    * *Constraint 2 (Efficiency Flip):* The efficiency factor `eta` flips drastically depending on the direction of motion. Quasi-static observers (like Shi) are highly noisy during velocity zero-crossings and should not be trusted blindly without energy/power thresholding.
* **Follower (UR5e):** Controlled via RTDE at 500 Hz using `torqueControl()`.
* **Sensors:** Internal F/T Sensor at the UR5e wrist.

## Coding Guidelines & Architectural Rules (DO NOT VIOLATE)
1. **Multi-Rate IPC:** Python's GIL will block hardware loops. The Base Policy (1-10 Hz, GPU) MUST run in a separate process from the Hardware Loop (330 Hz, CPU). Communication must occur via lock-free Shared Memory (`SharedTrajectoryBuffer`). Never use `threading` for policy inference.
2. **Timing Discipline:** Never use `time.time()` for control loops or trajectory interpolation. Always use `time.monotonic()` or `time.perf_counter()` to prevent NTP clock adjustment jumps.
3. **Image Handling:** Do not encode JPEGs inside the 330 Hz control loop. Use raw `numpy.copyto()` for shared memory. Offload any compression or resizing to the low-frequency policy process.
4. **Constant Admittance:** Use constant Stiffness (`K`) and Damping (`D`) for the Admittance controller. Do not implement variable stiffness dynamically, as it corrupts the physical meaning of `delta_q` for the Machine Learning pipeline.

## System Operating Modes
Ensure any control logic respects these three lifecycles:
* **Phase A (Demonstration Collection):** GELLO in Current Mode (Gravity Comp). UR5e in Impedance Mode. Human demonstrates the task.
* **Phase B (CR-DAgger Interventions & Controller Evaluation):** We evaluate two leader control strategies:
    1. **Impedance Baseline (Primary):** Base policy drives a soft PD controller (Current Mode). Human pushes physically against the tracking spring. `delta_q` is the physical deviation.
    2. **Admittance Controller (Comparison):** Base policy drives virtual mass-spring-damper (Position Mode).
    In both cases, `EnergyInjectionDetector` (or force threshold) registers intervention. Save `delta_q` and `tau_ext` to dataset.
* **Phase C (Autonomous Deployment):** $q_{ref\_total} = q_{base\_policy} + \Delta q_{residual\_policy}$. Human can still intervene.

## Step-by-Step Development Roadmap
When assisting with implementation, follow this chronological sequence to isolate bugs:

### Step 1: Observer & Baseline Validation
* Implement `observer_weights.py` / `observer_holding_weights.py` to document Shi Observer noise and the `eta`-flip issue.
* Validate Momentum Observer in Current Mode.

### Step 2: Leader Control Strategy (Impedance vs. Admittance)
* Implement `trajectory_controller.py` (Impedance): Soft Python PD controller driving Current Mode. (Likely the final choice due to hardware constraints).
* Implement `trajectory_controller_admittance.py` (Admittance): Virtual mass-spring-damper math. Document the tracking vs. compliance trade-off.
* Implement Intervention Logic based on force/energy thresholds to trigger data recording.
* Test with a hardcoded dummy policy (sine wave).

### Step 3: Multi-Rate Inter-Process Communication (IPC)
* Implement `SharedTrajectoryBuffer` (lock-free double buffering via `multiprocessing.shared_memory`).
* Implement `SharedObservationSnapshot` for sensory data.
* Stress-test IPC timing bounds.

### Step 4: Data Recording & ML Pipeline
* Format output using the **LeRobot v3.0** standard.
* Apply Latency Compensation for correction labels: `delta_q[t] = q[t] - q_ref[t - dt]`.
* Ensure feature synchronicity (`images`, `proprioception`, `tau_ext`, `wrench`, `delta_q`).