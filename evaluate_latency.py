import numpy as np

file_path = '/media/internal/nvme/jstranghoener/data/cr_dagger_npz/phaseB_smoke_whiteboard_whiping/crdagger_phaseB_lerobot_act_phaseB_smoke_whiteboard_whiping_1744143_ep0000.npz'
d = np.load(file_path)

q_cmd_ur5e = d['q_cmd_ur5e']
q_follower = d.get('q_follower', d.get('q_actual'))

q_cmd_leader = d['q_cmd_leader']
q_leader = d['q_leader']
timestamps = d['timestamps']

# Calculate average dt
dt = np.mean(np.diff(timestamps))
print(f"=== Recording Metriken ===")
print(f"Average frame dt: {dt:.4f} s ({1/dt:.1f} Hz)")

def estimate_latency_mse(target, actual, max_shift=40):
    # We assume 'actual' is delayed relative to 'target'.
    # -> actual[i + shift] aligns best with target[i]
    best_shift = 0
    min_error = float('inf')
    
    for shift in range(max_shift):
        if shift == 0:
            err = np.mean((target - actual)**2)
        else:
            err = np.mean((target[:-shift] - actual[shift:])**2)
            
        if err < min_error:
            min_error = err
            best_shift = shift
            
    return best_shift

print("\n=== Latenz Analyse (Mean Squared Error Minimierung) ===")
f_shift = estimate_latency_mse(q_cmd_ur5e, q_follower)
print(f"Follower (UR5e) Gesamt-Delay : {f_shift} frames -> ~{f_shift * dt * 1000:.1f} ms")

l_shift = estimate_latency_mse(q_cmd_leader, q_leader)
print(f"Leader (GELLO) Gesamt-Delay  : {l_shift} frames -> ~{l_shift * dt * 1000:.1f} ms")

print("\n=== Follower Latenz pro Gelenk ===")
for i in range(6):
    shift = estimate_latency_mse(q_cmd_ur5e[:, i], q_follower[:, i])
    print(f"  Joint {i+1}: {shift:2d} frames (~{shift * dt * 1000:5.1f} ms)")

print("\n=== Leader Latenz pro Gelenk ===")
for i in range(6):
    shift = estimate_latency_mse(q_cmd_leader[:, i], q_leader[:, i])
    print(f"  Joint {i+1}: {shift:2d} frames (~{shift * dt * 1000:5.1f} ms)")

dq_ref = d.get('dq_ref', np.zeros_like(q_cmd_ur5e))
np.set_printoptions(precision=4, suppress=True)
print(f"\n=== Policy Outputs Übersicht ===")
print(f"Max kommandierte Geschw.       : {np.max(np.abs(dq_ref), axis=0)} rad/s")
print(f"Durchschn. kommandierte Geschw.: {np.mean(np.abs(dq_ref), axis=0)} rad/s")

