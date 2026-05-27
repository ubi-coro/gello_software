import numpy as np

file_path = '/media/internal/nvme/jstranghoener/data/cr_dagger_npz/phaseB_smoke_whiteboard_whiping/crdagger_phaseB_lerobot_act_phaseB_smoke_whiteboard_whiping_1744143_ep0000.npz'
d = np.load(file_path)

eps_f = d['epsilon_ur5e']
eps_l = d['epsilon_leader']
delta = d['delta_leader']

np.set_printoptions(precision=4, suppress=True)

print("=== Follower Epsilon (Abweichung Hardware zu Command) ===")
print(f"Mean (abs) : {np.mean(np.abs(eps_f), axis=0)} rad")
print(f"Max  (abs) : {np.max(np.abs(eps_f), axis=0)} rad")

print("\n=== Leader Epsilon (Abweichung GELLO Hardware zu Command) ===")
print(f"Mean (abs) : {np.mean(np.abs(eps_l), axis=0)} rad")
print(f"Max  (abs) : {np.max(np.abs(eps_l), axis=0)} rad")

print("\n=== Leader Delta (Interventions-Offset) ===")
print(f"Mean (abs) : {np.mean(np.abs(delta), axis=0)} rad")
print(f"Max  (abs) : {np.max(np.abs(delta), axis=0)} rad")

