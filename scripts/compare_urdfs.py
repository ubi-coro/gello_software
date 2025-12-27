import os
import numpy as np
import sys

try:
    import pinocchio as pin
except ImportError:
    print("Error: Could not import 'pinocchio'. Please install it (e.g., 'conda install pinocchio -c conda-forge').")
    sys.exit(1)

def analyze_urdf(urdf_path):
    print(f"\n{'='*20}")
    print(f"Analyzing: {os.path.basename(urdf_path)}")
    print(f"Path: {urdf_path}")
    
    if not os.path.exists(urdf_path):
        print(f"Error: File not found at {urdf_path}")
        return None

    try:
        # Try to load the model
        if hasattr(pin, 'buildModelFromUrdf'):
            model = pin.buildModelFromUrdf(urdf_path)
        elif hasattr(pin, 'buildModelsFromUrdf'):
             model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
        else:
            print("Error: 'pinocchio' module has no 'buildModelFromUrdf' or 'buildModelsFromUrdf'.")
            return None

        # Check and set gravity
        print(f"Model gravity (linear): {model.gravity.linear}")
        if np.linalg.norm(model.gravity.linear) == 0:
            print("Gravity is zero. Setting to standard Earth gravity [0, 0, -9.81].")
            model.gravity.linear = np.array([0., 0., -9.81])

        data = model.createData()
        
        info = {
            "name": model.name,
            "nq": model.nq,
            "nv": model.nv,
            "nbodies": model.nbodies,
            "joints": [],
            "limits": [],
            "gravity_q0": None,
            "mass_diag_q0": None,
            "com_q0": None
        }
        
        print(f"Model name: {model.name}")
        print(f"Number of joints (nq): {model.nq}")
        print(f"Number of DoF (nv): {model.nv}")
        print(f"Number of bodies: {model.nbodies}")
        
        # Compute dynamics at q=0
        q0 = pin.neutral(model)
        
        # Forward Kinematics & Gravity
        pin.forwardKinematics(model, data, q0)
        pin.computeGeneralizedGravity(model, data, q0)
        
        print(f"\nGravity vector (g(q=0)):")
        print(data.g)
        info["gravity_q0"] = data.g
        
        # Mass Matrix
        pin.crba(model, data, q0)
        M = data.M
        print(f"\nMass Matrix diagonal (M(q=0) diag):")
        print(np.diag(M))
        info["mass_diag_q0"] = np.diag(M)
        
        # Center of Mass
        com = pin.centerOfMass(model, data, q0)
        print(f"\nTotal COM (q=0): {com}")
        info["com_q0"] = com
        
        return info

    except Exception as e:
        print(f"Error loading/processing URDF: {e}")
        return None

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    urdfs_to_check = [
        "gello/factr/urdf/franka_emika/factr_teleop_franka.urdf",
        "gello/factr/urdf/yam_active_gello/robot.urdf",
        "gello/factr/urdf/GELLO_Assembly_URDF_V4/GELLO_Assembly_URDF_V4.urdf"
    ]
    
    results = {}
    
    for rel_path in urdfs_to_check:
        full_path = os.path.join(base_path, rel_path)
        info = analyze_urdf(full_path)
        if info:
            results[os.path.basename(rel_path)] = info

    # Comparison Summary
    if len(results) > 1:
        print(f"\n{'='*20}")
        print("Comparison Summary")
        print(f"{'Property':<20} | {' | '.join([name[:15] for name in results.keys()])}")
        print("-" * (20 + 18 * len(results)))
        
        props = ["nq", "nv", "nbodies"]
        for p in props:
            row = f"{p:<20} | " + " | ".join([str(results[name][p]) for name in results.keys()])
            print(row)
            
        # Compare Gravity Norm
        print(f"{'Gravity Norm':<20} | " + " | ".join([f"{np.linalg.norm(results[name]['gravity_q0']):.4f}" for name in results.keys()]))

if __name__ == "__main__":
    main()
