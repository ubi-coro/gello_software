import numpy as np

def parallel_axis_theorem(I_cm, mass, vector_to_new_point):
    """
    Verschiebt einen Trägheitstensor I_cm (definiert im Schwerpunkt)
    um den Vektor 'vector_to_new_point' zu einem neuen Punkt.
    """
    d = np.array(vector_to_new_point) / 1000.0 # Umrechnung mm in m
    d_norm_sq = np.dot(d, d)
    E = np.eye(3)
    
    # Steiner Term: m * (d^2 * E - d * d.T)
    I_steiner = mass * (d_norm_sq * E - np.outer(d, d))
    
    return I_cm + I_steiner

def get_motor_inertia(motor_type="XC330"):
    # Datenblatt Werte in kg*m^2 (umgerechnet von g*mm^2 mit Faktor 1e-9)
    if motor_type == "XC330":
        # XC330 (aus deinem Chatverlauf)
        return np.array([
            [3.528e-6, -2.327e-8, -2.524e-8],
            [-2.327e-8, 1.801e-6, -4.571e-7],
            [-2.524e-8, -4.571e-7, 2.977e-6]
        ])
    elif motor_type == "XM430":
        # XM430 (aus deinem Chatverlauf)
        return np.array([
            [2.374e-5, -1.298e-7, -7.061e-8],
            [-1.298e-7, 1.302e-5, -1.828e-6],
            [-7.061e-8, -1.828e-6, 2.119e-5]
        ])
    return np.zeros((3,3))

# ==========================================
# INPUTS: HIER DEINE CAD-WERTE EINTRAGEN
# ==========================================

# Beispiel für Link 4 (Forearm)
# 1. Daten des PLASTIK-TEILS (aus CAD Inertial Properties)
m_L4_plastic = 0.0179  # kg (Link 4 Masse ohne Motor)
r_L4_plastic = np.array([2.98, 0.032, -18.817]) # mm (Schwerpunkt Plastik)

# Trägheitstensor des Plastiks (am Schwerpunkt des Plastiks!)
# Bitte Werte aus CAD hier einsetzen (in kg*m^2 !!)
# Falls CAD g*mm^2 liefert, nimm * 1e-9
I_L4_plastic = np.array([
    [8.049e-6, -3.49e-9, 1.90e-6],  # Ixx, Ixy, Ixz
    [-3.49e-9, 8.75e-6, -6.68e-9],  # Iyx, Iyy, Iyz
    [1.90e-6, -6.68e-9, 5.20e-6]    # Izx, Izy, Izz
]) 

# 2. Daten des MOTORS (XC330 in Link 4)
m_motor = 0.0232 # kg
r_L4_motor = np.array([-34.946, 0.957, -4.234]) # mm (Position Motor CoM)
I_motor = get_motor_inertia("XC330")

# 3. Zuvor berechneter NEUER GESAMT-SCHWERPUNKT (aus Turn 4)
r_L4_neu = np.array([-27.92, 0.55, -1.09]) # mm

# ==========================================
# BERECHNUNG
# ==========================================

# Vektoren berechnen (Verschiebung vom jew. Schwerpunkt zum neuen Gesamt-Schwerpunkt)
dist_plastic = r_L4_neu - r_L4_plastic
dist_motor   = r_L4_neu - r_L4_motor

# Steiner anwenden
I_plastic_new = parallel_axis_theorem(I_L4_plastic, m_L4_plastic, dist_plastic)
I_motor_new   = parallel_axis_theorem(I_motor, m_motor, dist_motor)

# Addieren
I_total = I_plastic_new + I_motor_new

print("Neuer Inertia Tensor für Link 4 (kg*m^2):")
print(f"ixx=\"{I_total[0,0]:.2e}\" ixy=\"{I_total[0,1]:.2e}\" ixz=\"{I_total[0,2]:.2e}\"")
print(f"iyy=\"{I_total[1,1]:.2e}\" iyz=\"{I_total[1,2]:.2e}\"")
print(f"izz=\"{I_total[2,2]:.2e}\"")