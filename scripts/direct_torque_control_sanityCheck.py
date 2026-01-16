"""
Impedance Control Test Script für UR5e

Testet verschiedene Steifigkeiten und Dämpfungen.
Der Roboter sollte sich wie eine Feder zur Startposition verhalten.

WARNUNG: Starte mit niedrigen Kp/Kd Werten!
"""

import rtde_control
import sys
import rtde_receive
import numpy as np
import time

ROBOT_IP = "192.168.1.11"

def test_impedance_hold():
    """Test: Roboter hält Position mit einstellbarer Steifigkeit."""
    
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    
    # Startposition merken
    q_target = np.array(rtde_r.getActualQ())
    print(f"Target position (hold): {[f'{np.rad2deg(x):.1f}°' for x in q_target]}")
    
    # Impedance Parameter - STARTE NIEDRIG!
    kp = 150.0   # Nm/rad - niedrig = weich
    kd = 15.0    # Nm*s/rad
    
    # Safety limits
    tau_max = np.array([30.0, 30.0, 20.0, 5.0, 5.0, 5.0])
    
    print(f"\nImpedance Control: Kp={kp}, Kd={kd}")
    print("Der Roboter sollte sich jetzt wie eine weiche Feder verhalten.")
    print("Drücke gegen den Roboter - er sollte nachgeben und zurückkehren.")
    print("Ctrl+C zum Stoppen\n")
    
    dt = 1.0 / 500.0
    
    try:
        while True:
            t0 = time.time()
            
            # Aktuelle Zustände
            q = np.array(rtde_r.getActualQ())
            qd = np.array(rtde_r.getActualQd())
            
            # Impedance Law
            q_error = q_target - q
            tau_cmd = kp * q_error - kd * qd
            
            # Clamp
            tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
            
            # Debug alle 500 Iterationen (~1 Sekunde)
            if int(time.time() * 500) % 500 == 0:
                print(f"q_error (deg): {[f'{np.rad2deg(x):+.2f}' for x in q_error]}")
                print(f"tau_cmd (Nm):  {[f'{x:+.2f}' for x in tau_cmd]}")
            
            # Senden
            rtde_c.directTorque(tau_cmd.tolist())
            
            # Timing
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        rtde_c.stopJ(2.0)
        print("Done.")


def test_impedance_follow_sine():
    """Test: Roboter folgt einer Sinus-Trajektorie mit Impedance Control."""
    
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    
    q_start = np.array(rtde_r.getActualQ())
    
    # Parameter
    kp = 150.0
    kd = 15.0
    amplitude = np.deg2rad(10)  # 10° Amplitude
    frequency = 0.2  # Hz
    joint_to_move = 0  # Base joint
    
    tau_max = np.array([30.0, 30.0, 20.0, 5.0, 5.0, 5.0])
    
    print(f"Sinus-Trajektorie auf Joint {joint_to_move}")
    print(f"Amplitude: {np.rad2deg(amplitude):.1f}°, Frequenz: {frequency} Hz")
    print("Ctrl+C zum Stoppen\n")
    
    dt = 1.0 / 500.0
    t_start = time.time()
    
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            
            # Ziel-Trajektorie
            q_target = q_start.copy()
            q_target[joint_to_move] += amplitude * np.sin(2 * np.pi * frequency * t)
            
            qd_target = np.zeros(6)
            qd_target[joint_to_move] = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
            
            # Aktuelle Zustände
            q = np.array(rtde_r.getActualQ())
            qd = np.array(rtde_r.getActualQd())
            
            # Impedance Law
            q_error = q_target - q
            qd_error = qd_target - qd
            tau_cmd = kp * q_error + kd * qd_error
            
            tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)
            
            rtde_c.directTorque(tau_cmd.tolist())
            
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        rtde_c.stopJ(2.0)


def test_read_write_simultaneity():
    """
    Test: Simultaner Lese- und Schreibzugriff auf rtde_c (STRESS TEST).
    
    Optimierungs-Versuch:
    Wir führen erst den kritischen Schreibzugriff (directTorque) aus, 
    und danach den langsamen Lesezugriff (getJointTorques).
    """

    print("Initialisiere Interfaces...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    q_target = np.array(rtde_r.getActualQ())
    print(f"Target position: {[f'{np.rad2deg(x):.1f}°' for x in q_target]}")

    # Weiche Regelung
    kp = 20.0
    kd = 3.0
    tau_max = np.array([30.0, 30.0, 20.0, 5.0, 5.0, 5.0])

    dt = 1.0 / 500.0  # 2ms Target Cycle Time

    print("\n--- Start Latency Test (Explizit rtde_c conflict) ---")
    print("Strategy: WRITE (Critical) -> READ (Optional/Slow)")

    latencies = []
    overruns = 0
    steps = 0
    max_loop_time = 0.0

    try:
        while True:
            t0 = time.time()

            # --- 1. STATE READ (rtde_r is safe/fast) ---
            q = np.array(rtde_r.getActualQ())
            qd = np.array(rtde_r.getActualQd())

            # --- 2. CONTROL CALC ---
            q_error = q_target - q
            tau_cmd = kp * q_error - kd * qd
            tau_cmd = np.clip(tau_cmd, -tau_max, tau_max)

            # --- 3. WRITE (CRITICAL PRIORITY) ---
            # Send command immediately to keep robot happy
            rtde_c.directTorque(tau_cmd.tolist())

            # --- 4. READ (THE BOTTLENECK) ---
            t_read_start = time.perf_counter()
            
            # Explicitly using the control interface as requested
            # usage of rtde_c for reading IS the stress test
            meas_torques = rtde_c.getJointTorques()
            
            t_read = time.perf_counter() - t_read_start

            # --- Timing Analyse ---
            t_loop = time.time() - t0

            # Statistiken
            if len(latencies) > 1000: latencies.pop(0)
            latencies.append(t_read * 1000.0)

            if t_loop > max_loop_time:
                max_loop_time = t_loop

            if t_loop > dt * 1.05:
                overruns += 1

            steps += 1

            # Logging: Nur alle 500 Steps
            if steps % 500 == 0:
                avg_read_ms = sum(latencies) / len(latencies)
                sys.stdout.write(f"\rStep {steps}: Read Latency={avg_read_ms:.3f}ms | Max Loop={max_loop_time*1000:.3f}ms | Overruns={overruns}   ")
                sys.stdout.flush()
                
                overruns = 0
                max_loop_time = 0.0

            # Wait for next cycle
            if t_loop < dt:
                time.sleep(dt - t_loop)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        rtde_c.stopJ(2.0)
        print("Done.")


if __name__ == "__main__":
    print("=== UR5e Impedance Control Tests ===")
    print("1: Hold position (push against robot)")
    print("2: Follow sine trajectory")
    print("3: Test Read/Write Simultaneity (Latency Check)")
    choice = input("Auswahl (1/2/3): ")
    
    if choice == "1":
        test_impedance_hold()
    elif choice == "2":
        test_impedance_follow_sine()
    elif choice == "3":
        test_read_write_simultaneity()
    else:
        print("Ungültige Auswahl")