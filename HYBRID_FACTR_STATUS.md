# Hybrid-Ansatz (URDF + Identifikation) – Stand & Integration in FACTR

## Zielbild

Ein hybrides Dynamikmodell für den GELLO-Leader, das:

- **URDF (CAD + Satz von Steiner)** als Basis nutzt (Geometrie, Massen, Inertien).
- **Experimentell identifizierte Parameter** ergänzt (primär low-frequency: Reibung, Bias; optional Gravity/CoM-Korrekturen).
- In **FACTR** (gravity compensation / dynamikbasierte Regelung) **zur Laufzeit** verwendet wird, ohne URDF-Dateien zwangsläufig umzuschreiben.

Der Fokus ist Teleoperation im **niederfrequenten Bereich**, in dem **Gravitation + Reibung** dominieren.

---

## Stand im Code

### 1) Leader Disturbance Observer (ohne F/T-Sensor)

- Datei: gello/4Ch_Bilat/gello_ur5e_observer.py
- Inhalt: ein erster, **selbstenthaltener** Observer zur Schätzung von
  - Gelenkgeschwindigkeit (dtheta_hat)
  - externen Gelenkmomenten (tau_ext_hat)
- Pinocchio wird verwendet für Modellterme (M, C, g).
- Die Integration in FACTR/Agents ist bewusst noch nicht erfolgt (Prototyp/First shot).

### 2) Parameteridentifikation (offline)

- Datei: gello/4Ch_Bilat/computeJointTorqueRegressor.py

Enthält aktuell zwei Pfade:

1. **identify_friction_only(...)** (empfohlen für „so einfach wie nötig“)
   - Identifiziert Reibung (viskos + Coulomb) und optional Bias pro Gelenk.
   - Nutzt **URDF-Gravitation** als bekannten Term.
   - Benötigt kein ddq und ist robust für handgeführte, low-frequency Datensätze.

   Modellannahme:

   \[
   \tau_{meas}(q, \dot q) \approx g_{URDF}(q) + F_v \dot q + F_c \mathrm{sign}(\dot q) + b
   \]

2. **identify_dynamics(...)** (Full rigid-body + Reibung)
   - Baut den vollen Pinocchio-Regressor (computeJointTorqueRegressor) und ergänzt Reibung.
   - Kann theoretisch auch Inertien mitidentifizieren.
   - Für saubere Inertial-Identifikation wären gezielt anregende Trajektorien und aufwendigere Constraints/Validierung nötig.

---

## Empfohlenes Vorgehen (Hybrid, FACTR-kompatibel)

### Phase A (MVP): URDF + Reibung/Bias identifizieren und online addieren

1) **Datenerfassung (Leader, handgeführt)**

- Logge pro Sample:
  - q (Position)
  - dq (Velocity)  
  - tau_meas (Gelenkmoment aus Motorstrom / Current->Torque Umrechnung)
  - dt (optional, für friction-only nicht zwingend)

Praxis-Tipps:
- Viele Richtungswechsel pro Gelenk (für Coulomb-Term).
- Langsam, aber nicht nur dq≈0 (sonst dominiert Bias).
- Unterschiedliche Konfigurationen q (damit g(q) „ausgemittelt“ wird).

2) **Offline-Identifikation**

- Verwende identify_friction_only(...)
- Nutze ridge_lambda > 0, wenn Parameter instabil/zu groß werden.

Output:
- fv (nq,), fc (nq,), optional bias (nq,)

3) **Online-Modell in FACTR verwenden (ohne URDF zu ändern)**

Im FACTR-Loop wird typischerweise bereits g(q) aus Pinocchio berechnet.
Ergänze das Modell um Reibung/Bias:

\[
\tau_{hybrid}(q, \dot q) = g_{URDF}(q) + \tau_{fric}(\dot q) + b
\]

mit

\[
\tau_{fric}(\dot q) = F_v \dot q + F_c \mathrm{sign}(\dot q)
\]

**Wichtig:** Das ist kein „URDF-Update“, sondern eine **Modellerweiterung zur Laufzeit**.

4) **Gravity Compensation / Transparenz**

- Gravity compensation nutzt tau_hybrid statt nur g(q)
- Ergebnis: weniger „sticky/klebrig“, bessere Haptik bei langsamen Bewegungen


### Phase B (optional): Korrekturen an Gravitation/CoM (weiter low-frequency)

Wenn die URDF-Gravitation merklich danebenliegt (z.B. systematischer Offset oder falsche Massenverteilung), kann man zusätzlich sehr wenige Parameter fitten:

- Joint-spezifische gravity-scaling Faktoren oder
- wenige CoM-Offsets

Das ist deutlich einfacher als volle Inertial-Identifikation und bleibt im low-frequency Scope.


### Phase C (aufwändig): Trägheiten/Inertialparameter identifizieren

Für eine sinnvolle Identifikation von Inertien braucht man i.d.R.:

- gezielt designte Trajektorien (persistente Anregung, Multi-Sinus/Chirp)
- ausreichend ddq-Anteile
- saubere ddq-Schätzung (oft über gefiltertes q)
- Constraints/Regularisierung für physikalische Plausibilität (SPD-Inertias etc.)

Das ist eher ein eigenes Teilprojekt.

---

## Integration in FACTR (konzeptionell, minimalinvasiv)

### Wo sitzt das im Stack?

- URDF-basierte Terme (g, C, M) kommen aus Pinocchio in der FACTR-Implementierung.
- Identifizierte Parameter werden als **zusätzlicher Modellterm** in die Torque-Berechnung addiert.

Minimaler Integrationspunkt (später):

- In der Torque-Berechnung für den Leader (FACTR loop) wird
  - g(q) berechnet
  - tau_fric(dq) berechnet
  - ggf. bias addiert
  - resultierendes tau_cmd an Dynamixel gesendet

### Empfehlung zur Ablage der identifizierten Parameter

- Speichere fv/fc/bias als `npz` (oder yaml), z.B. unter `gello/factr/identified_params/`.
- Lade sie beim Start der FACTR-Session.

---

## Nächste konkrete Schritte

1) Datenerfassungs-Skript/Logger (q, dq, tau_meas) für den Leader
2) friction-only Fit durchführen und Parameter speichern
3) (Optional) Offline-Validierung: tau_pred = g(q) + fric(dq) + bias vs tau_meas
4) Danach erst: minimaler Hook in FACTR-Torque-Berechnung

