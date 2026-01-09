import os
import numpy as np
import sys

try:
    import pinocchio as pin
except ImportError:
    print("Error: Could not import 'pinocchio'. Please install it (e.g., 'conda install pinocchio -c conda-forge').")
    sys.exit(1)


def _try_get_mass(inertia_obj):
    if inertia_obj is None:
        return None
    m = getattr(inertia_obj, "mass", None)
    if m is None:
        return None
    try:
        return float(m() if callable(m) else m)
    except Exception:
        return None


def _try_get_inertia_matrix(inertia_obj):
    if inertia_obj is None:
        return None
    I = getattr(inertia_obj, "inertia", None)
    if I is None:
        return None
    try:
        return np.array(I)
    except Exception:
        return None


def _summarize_inertials(model, top_n=8, zero_mass_eps=1e-8):
    names = list(getattr(model, "names", []))
    inertias = list(getattr(model, "inertias", []))
    njoints = getattr(model, "njoints", len(names))
    if njoints is None or njoints == 0:
        njoints = len(names)

    joint_rows = []
    for jid in range(min(njoints, len(inertias))):
        name = names[jid] if jid < len(names) else f"joint_{jid}"
        inertia_obj = inertias[jid] if jid < len(inertias) else None
        mass = _try_get_mass(inertia_obj)
        I = _try_get_inertia_matrix(inertia_obj)

        if mass is None:
            mass = 0.0
        inertia_is_zero = True
        if I is not None and I.size != 0:
            inertia_is_zero = bool(np.all(np.abs(I) <= 0.0))

        joint_rows.append(
            {
                "jid": jid,
                "name": name,
                "mass": float(mass),
                "inertia_is_zero": inertia_is_zero,
            }
        )

    # Skip universe (jid=0) for meaningful mass accounting
    moving_rows = [r for r in joint_rows if r["jid"] != 0]

    masses = np.array([r["mass"] for r in moving_rows], dtype=float) if moving_rows else np.array([], dtype=float)
    total_mass = float(np.sum(masses)) if masses.size else 0.0

    zero_mass = [r for r in moving_rows if abs(r["mass"]) <= zero_mass_eps]
    zero_mass_names = [r["name"] for r in zero_mass]

    inertia_zero = [r for r in moving_rows if r["inertia_is_zero"]]
    inertia_zero_names = [r["name"] for r in inertia_zero]

    top = sorted(moving_rows, key=lambda r: r["mass"], reverse=True)[:top_n]
    top_masses = [(r["name"], float(r["mass"])) for r in top]

    return {
        "njoints": int(njoints),
        "total_mass": total_mass,
        "num_moving_joints": int(len(moving_rows)),
        "num_zero_mass": int(len(zero_mass_names)),
        "zero_mass_names": zero_mass_names,
        "num_zero_inertia": int(len(inertia_zero_names)),
        "zero_inertia_names": inertia_zero_names,
        "top_masses": top_masses,
    }

def analyze_urdf(urdf_path):
    print(f"\n{'='*80}")
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
        if np.linalg.norm(model.gravity.linear) == 0:
            model.gravity.linear = np.array([0., 0., -9.81])

        data = model.createData()
        
        print(f"Model name: {model.name}")
        print(f"nq: {model.nq}, nv: {model.nv}, nbodies: {model.nbodies}")

        inertial_summary = _summarize_inertials(model)
        print("\nInertial summary")
        print(f"Total mass (sum link inertias): {inertial_summary['total_mass']:.6f} kg")
        print(
            f"Zero-mass links: {inertial_summary['num_zero_mass']}/{inertial_summary['num_moving_joints']}"
        )
        if inertial_summary["num_zero_mass"]:
            preview = ", ".join(inertial_summary["zero_mass_names"][:10])
            suffix = " ..." if len(inertial_summary["zero_mass_names"]) > 10 else ""
            print(f"  e.g.: {preview}{suffix}")
        print(
            f"Zero-inertia-matrix links: {inertial_summary['num_zero_inertia']}/{inertial_summary['num_moving_joints']}"
        )
        if inertial_summary["num_zero_inertia"]:
            preview = ", ".join(inertial_summary["zero_inertia_names"][:10])
            suffix = " ..." if len(inertial_summary["zero_inertia_names"]) > 10 else ""
            print(f"  e.g.: {preview}{suffix}")
        print("Top masses:")
        for n, m in inertial_summary["top_masses"]:
            print(f"  - {n}: {m:.6f} kg")

        # Simple plausibility warnings (heuristics)
        warnings = []
        if inertial_summary["total_mass"] < 1.0:
            warnings.append("Total mass < 1 kg (very likely wrong units or missing inertials)")
        elif inertial_summary["total_mass"] < 5.0:
            warnings.append("Total mass < 5 kg (suspicious for full arm+gripper)")
        if inertial_summary["num_moving_joints"] > 0:
            frac_zero_mass = inertial_summary["num_zero_mass"] / inertial_summary["num_moving_joints"]
            if frac_zero_mass > 0.5:
                warnings.append(
                    f">50% links have ~0 mass ({inertial_summary['num_zero_mass']}/{inertial_summary['num_moving_joints']})"
                )
        # last joint massless check (often gripper)
        if inertial_summary["num_moving_joints"] > 0 and getattr(model, "njoints", 0):
            last_jid = getattr(model, "njoints", len(getattr(model, "names", []))) - 1
            if 0 <= last_jid < len(getattr(model, "inertias", [])):
                last_mass = _try_get_mass(model.inertias[last_jid])
                if last_mass is not None and abs(last_mass) <= 1e-8:
                    last_name = model.names[last_jid] if last_jid < len(model.names) else f"joint_{last_jid}"
                    warnings.append(f"Last joint inertia mass is ~0 ({last_name})")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  - {w}")

        # Define Poses to check
        poses = {}
        
        # 1. Zero / Neutral Pose
        q_neutral = pin.neutral(model)
        poses["Neutral (q=0)"] = q_neutral

        # 2. Random Pose 1 (Fixed Seed for reproducibility per model structure)
        # Note: randomConfiguration depends on joint limits in URDF.
        np.random.seed(42) 
        q_rand1 = pin.randomConfiguration(model)
        poses["Random Pose 1"] = q_rand1

        # 3. Random Pose 2
        q_rand2 = pin.randomConfiguration(model)
        poses["Random Pose 2"] = q_rand2
        
        # 4. Fixed Pose (e.g. all 0.5 rad, if within limits, or just force it)
        # We force it to see dynamics at a specific configuration regardless of limits
        # Only if nq == nv (fixed base usually)
        if model.nq == model.nv:
             q_fixed = np.full(model.nq, 0.5)
             poses["Fixed (all 0.5)"] = q_fixed

        results = {
            "name": model.name,
            "nq": model.nq,
            "nv": model.nv,
            "nbodies": model.nbodies,
            "inertial_summary": inertial_summary,
            "pose_data": {}
        }

        for pose_name, q in poses.items():
            print(f"\n--- {pose_name} ---")
            # print(f"q: {np.array2string(q, precision=3, suppress_small=True)}")
            
            # Forward Kinematics & Gravity
            pin.forwardKinematics(model, data, q)
            pin.computeGeneralizedGravity(model, data, q)
            
            # Mass Matrix
            pin.crba(model, data, q)
            M_diag = np.diag(data.M)
            
            # Center of Mass
            com = pin.centerOfMass(model, data, q)
            
            print(f"Gravity (g): {np.array2string(data.g, precision=4, suppress_small=True)}")
            print(f"Mass Matrix Diag: {np.array2string(M_diag, precision=4, suppress_small=True)}")
            print(f"CoM Position: {np.array2string(com, precision=4, suppress_small=True)}")
            
            results["pose_data"][pose_name] = {
                "g": data.g.copy(),
                "g_norm": float(np.linalg.norm(data.g)),
                "M_diag": M_diag.copy(),
                "com": com.copy()
            }
            
        return results

    except Exception as e:
        print(f"Error loading/processing URDF: {e}")
        import traceback
        traceback.print_exc()
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

    # Comparison Summary Table for Neutral Pose
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON (Neutral Pose q=0)")
        print(f"{'Property':<25} | {' | '.join([name[:20] for name in results.keys()])}")
        print("-" * (25 + 23 * len(results)))
        
        # Basic props
        for p in ["nq", "nv", "nbodies"]:
            row = f"{p:<25} | " + " | ".join([str(results[name][p]) for name in results.keys()])
            print(row)
            
        # Gravity Norm at Neutral
        row_g = f"{'Gravity Norm (Neutral)':<25} | "
        vals = []
        for name in results.keys():
            g = results[name]["pose_data"]["Neutral (q=0)"]["g"]
            vals.append(f"{np.linalg.norm(g):.4f}")
        print(row_g + " | ".join(vals))

        # Mass Matrix Trace at Neutral
        row_m = f"{'Mass Matrix Trace':<25} | "
        vals = []
        for name in results.keys():
            m = results[name]["pose_data"]["Neutral (q=0)"]["M_diag"]
            vals.append(f"{np.sum(m):.4f}")
        print(row_m + " | ".join(vals))

if __name__ == "__main__":
    main()
