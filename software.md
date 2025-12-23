
# Softwarestruktur (GELLO Software) – Architekturüberblick

Dieses Dokument beschreibt die Softwarestruktur des Projekts und den grundlegenden Teleoperations-Flow (Leader → Follower). Anschließend wird erklärt, wie **FACTR** (Force-Augmented Teleoperation / Gravity Compensation) diese Grundstruktur erweitert.

Hinweis: **UR5e-spezifische Erweiterungen** (z.B. Force-Feedback-Varianten für UR5e, zusätzliche Observer, etc.) stammen aus deiner Arbeit und werden hier als **Custom Extensions** gekennzeichnet. Der Kern der Architektur (Agent/Robot/Env/Loop, MuJoCo-Sim) folgt dem ursprünglichen Projektaufbau.

---

## 1. Projektstruktur (Ordner/Module)

Die Codebase ist in wenige Kernbereiche gegliedert:

- `gello/agents/`
	- Agenten repräsentieren das „Eingabegerät“ (Leader): z.B. GELLO (Dynamixel), SpaceMouse, Quest.
	- Wichtige Idee: **Agent liefert Aktionen** (typisch Joint-Targets) aus Beobachtungen.

- `gello/robots/`
	- Roboter repräsentieren das „Ausgabegerät“ (Follower): reale Roboter (UR, etc.) oder Simulation.
	- Wichtige Idee: **Robot nimmt Aktionen entgegen** (z.B. joint_state) und liefert Beobachtungen zurück.

- `gello/env.py`
	- `RobotEnv` ist der „Kleber“: verbindet Agent und Robot, hält Rate/Timing, stellt `step()` und `get_obs()` bereit.

- `gello/utils/`
	- Enthält u.a. den zentralen Control-Loop (`control_utils.py`), Launch/Helper-Funktionen.

- `gello/dynamixel/`
	- Low-Level Ansteuerung und Abstraktionen für Dynamixel-Servos.

- `gello/factr/`
	- FACTR-spezifische Komponenten (insb. Gravity Compensation auf dem Leader) und URDF-Assets.

- `gello/dm_control_tasks/` und `gello/robots/sim_robot.py`
	- Simulation mit **MuJoCo** (über `dm_control`/MJCF und/or direkte MuJoCo APIs).
	- Enthält Arm-Modelle (z.B. UR5e) und Task-/Arena-Struktur.

- `experiments/`
	- Entry-Points / Launch-Skripte (z.B. YAML-basierte Launches, schnelle Testläufe).

- `configs/`
	- YAML-Konfigurationen, die definieren: welche Agenten/Robots starten, Mapping, Gains, Ports, URDF-Pfade.

---

## 2. Grundarchitektur: Leader-Follower Teleoperation

### 2.1 Kern-Interfaces (konzeptionell)

Das Projekt folgt einem klaren Muster:

- **Agent (Leader)**
	- liest Eingaben (z.B. Joint-Positionen des GELLO Leaders)
	- berechnet daraus eine **Action** (typisch Ziel-Gelenkpositionen)

- **Robot (Follower)**
	- empfängt die Action (z.B. Joint-Targets)
	- setzt diese auf dem realen Roboter oder in der Simulation um
	- liefert **Observations** zurück (z.B. Joint States, TCP Pose)

- **Environment (`RobotEnv`)**
	- verbindet Agent und Robot
	- steuert Timing/Rate und die Abfolge „obs → act → step“

### 2.2 Control Loop

Der typische Ablauf ist:

1. `obs = env.get_obs()`
2. `action = agent.act(obs)`
3. `env.step(action)` → ruft intern `robot.command_joint_state(action)`
4. Wiederholen mit definierter Rate

Dieser Flow wird in Utilities/Experiment-Skripten gestartet.

---

## 3. Skizze: Datenfluss ohne FACTR

```text
					(Leader Input)
			GELLO (Dynamixel) / SpaceMouse / Quest
									 |
									 v
						+--------------+
						|    Agent     |
						| act(obs)->u  |
						+--------------+
									 |
							 action u
									 |
									 v
						+--------------+
						|   RobotEnv   |
						| step(u)      |
						+--------------+
									 |
					command_joint_state(u)
									 |
									 v
		 +--------------------------------+
		 |            Robot               |
		 |  - Real UR / anderer Robot     |
		 |  - oder MuJoCo Sim             |
		 +--------------------------------+
									 |
						 get_observations()
									 |
									 v
								obs
```

---

## 4. Simulation: MuJoCo / dm_control

Für Simulation wird MuJoCo verwendet:

- UR5e als MJCF-Modell aus der MuJoCo Menagerie (siehe `gello/dm_control_tasks/arms/ur5e.py`).
- Simulation-Server/Robot-Wrapper sitzt in `gello/robots/sim_robot.py`.
- Die Sim kann über YAML-Konfigs wie ein normaler „Robot“ gestartet werden.

Damit ist es möglich, **deine Leader-Hardware** (GELLO + Dynamixel) bereits gegen einen **simulierten UR5e** zu testen.

---

## 5. FACTR: Erweiterung der Grundarchitektur

### 5.1 Was FACTR im Projekt bedeutet

FACTR erweitert primär den **Leader**:

- Der Leader (GELLO) wird nicht nur „ausgelesen“, sondern aktiv mit **Drehmomenten** angesteuert.
- Ziel ist eine **dynamische Maskierung** der internen Leader-Dynamik (insb. Gravitation), um den Arm „leicht/transparent“ zu machen.
- Das basiert auf einem **Pinocchio-Modell** des Leaders (URDF), aus dem in Echtzeit inverse Dynamik / Gravitation berechnet wird.

Typisches Modellkonzept:

- `g(q)` aus URDF/Pinocchio
- optional weitere Terme (Reibung, Barrier-Forces, Nullspace-Regulation)

### 5.2 Skizze: Datenfluss mit FACTR (Gravity Compensation)

```text
				Leader (GELLO + Dynamixel)
									 |
					read q, dq, ...
									 |
									 v
				+-----------------------+
				| FACTR (Leader Loop)   |
				| - Pinocchio: g(q)     |
				| - (optional) friction |
				| - torque_cmd -> DX    |
				+-----------------------+
									 |
				 compensated leader state
									 |
									 v
						+--------------+
						|    Agent     |
						| act(obs)->u  |
						+--------------+
									 |
							 action u
									 v
							 RobotEnv -> Follower
```

Wichtig: FACTR läuft typischerweise mit höherer Frequenz (z.B. 500 Hz), während Teleop/Env oft mit geringerer Rate (z.B. 30 Hz) läuft.

---

## 6. Konfiguration (YAML)

Das Projekt wird stark über YAML konfiguriert:

- Auswahl Robot/Agent über `_target_` (Python Import-Pfad)
- Hardwareports, Robot IPs
- Mapping: `index_map`, `signs`, `offsets`, `auto_align`
- FACTR-Parameter: `gravity_comp.gain`, `controller.frequency`, etc.

Beispiel: URDF-Pfad für den Leader (GELLO):

- `configs/ur5e_gello_factr_hw.yaml` enthält z.B.
	- `arm_teleop.leader_urdf: "gello/factr/urdf/.../GELLO_Assembly_URDF_V3.urdf"`

---

## 7. Custom Extensions (deine Erweiterungen für UR5e)

Folgende Punkte gehören **nicht** zum ursprünglichen Kern, sondern sind als Erweiterung für deine UR5e-FACTR Zielsetzung zu verstehen:

- UR5e-spezifische FACTR/Force-Feedback-Erweiterungen (z.B. Nutzung von Wrench/Jacobian, oder neue UR-API für joint torques)
- Erweiterte Observer für Leader (z.B. Disturbance Observer/4-Channel Bilateral Control)
- Zusätzliche Identifikations-/Hybrid-Modellierung (Reibung/Bias-Identifikation) als Ergänzung zum CAD/Steiner-URDF

Diese Erweiterungen bauen jedoch sauber auf der bestehenden Architektur auf (Agent/Robot/Env/Loop + Pinocchio + YAML).

---

## 8. Einordnung: Hybrid-Modell in FACTR (URDF + Identifikation)

Für einen hybriden Ansatz wird die URDF **nicht zwingend geändert**.
Stattdessen wird zur Laufzeit ein Modellterm ergänzt:

\[
	au_{hybrid}(q,\dot q) = g_{URDF}(q) + \tau_{fric}(\dot q) + b
\]

Das passt sehr gut zur FACTR-Idee: URDF liefert zuverlässige Basis (Geometrie, Inertien), Identifikation verbessert low-frequency Transparenz (Reibung/Bias).

Weiterführend (optional) können Gravity/CoM-Korrekturen oder vollständige Inertial-Identifikation ergänzt werden, was allerdings deutlich mehr Aufwand für Trajektorien-Design, Constraints und Validierung bedeutet.

---

## 9. Einordnung zur Doku-Passage (Code-Referenzen, Stand vs. offene Punkte)

Dieser Abschnitt ordnet die im Text beschriebenen Konzepte direkt der aktuellen Codebase zu und markiert, was bereits umgesetzt ist bzw. was für die vollständige Umsetzung noch fehlt.

### 9.1 „URDF als Basis“ (CAD → URDF → Pinocchio)

**Im Text:** Das URDF beschreibt kinematische Kette + dynamische Parameter (Masse, CoM, Trägheit). Korrekturen der Inertial-Tags erfolgen offline (CAD + Satz von Steiner) und das URDF wird anschließend in Pinocchio geladen.

**In der Codebase (umgesetzt):**

- Das URDF wird als Pfad in YAML konfiguriert und für die Modellrechnung geladen.
	- Beispielkonfig (Leader URDF): [configs/ur5e_gello_factr_hw.yaml](configs/ur5e_gello_factr_hw.yaml)
	- Loader & Modellnutzung: [gello/factr/gravity_compensation.py](gello/factr/gravity_compensation.py)

**In der Codebase (nicht enthalten / extern):**

- Der CAD-Workflow (Fusion 360, ACDC4Robot) und die Steiner-Korrektur sind **nicht** Teil dieses Repos. Die Software setzt voraus, dass das URDF bereits physikalisch plausibel ist.

### 9.2 „Interne vs. externe Kräfte unterscheiden“

**Im Text:** Für haptische Transparenz muss das System interne Dynamik (z.B. Gravitation/Reibung) von Umweltinteraktion (extern) trennen.

**In der Codebase (teilweise umgesetzt):**

- **Interne Terme (Leader):**
	- Gravitation / inverse Dynamik via Pinocchio RNEA ist implementiert:
		- [gello/factr/gravity_compensation.py](gello/factr/gravity_compensation.py)

- **Externe Terme (Follower → Leader Force Feedback):**
	- Eine Force-Feedback-Struktur existiert: Follower-Joint-Torques können abgefragt und als Feedback-Torque auf den Leader gemappt werden.
		- Follower torque acquisition + Feedback-Gesetz: [gello/factr/gravity_compensation.py](gello/factr/gravity_compensation.py)
	- Voraussetzung: der Follower-Robot muss Joint-Torques liefern (z.B. UR über RTDE „ActualJointTorques“ oder eigene Methode).

**Offen / abhängig von Hardware/API (noch zu vervollständigen):**

- Falls der Follower nur einen TCP-Wrench liefert (6D), fehlt im Kern noch die standardisierte Pipeline „Wrench → Joint Torque“ (Jacobian-Transpose bzw. wrench-to-torque mapping). Das gehört zu deinen UR5e-Extensions.

### 9.3 Gravitationsterm und Reibungsterm (low-frequency Fokus)

**Im Text:** Low-frequency: \(\tau \approx g(q) + \tau_{fric}(\dot q, t)\). Reibung wird in kinetisch (viskos + Coulomb) und statisch (Dithering) aufgeteilt.

**In der Codebase (umgesetzt):**

- **Gravitation (und mehr):**
	- RNEA-basierte Gravity Compensation ist implementiert:
		- [gello/factr/gravity_compensation.py](gello/factr/gravity_compensation.py)

- **Statische Reibung / Dithering (Haftreibungskompensation):**
	- Ein stiction/dither Ansatz ist implementiert und in den FACTR-Loop integrierbar:
		- [gello/factr/gravity_compensation.py](gello/factr/gravity_compensation.py)
	- Zusätzlich existieren Standalone-Skripte, die das Verhalten demonstrieren:
		- [gello/factr/run_UR5e_gello_gravity_comp.py](gello/factr/run_UR5e_gello_gravity_comp.py)

**In der Codebase (noch nicht als „kinetisches Reibungsmodell“ im Controller integriert):**

- Ein explizites Modell \(F_v \dot q + F_c\,\mathrm{sgn}(\dot q) + b\) ist als Offline-Identifikation vorbereitet, aber noch nicht in den FACTR-Laufzeitcontroller eingebunden.
	- Identifikation (offline): [gello/4Ch_Bilat/computeJointTorqueRegressor.py](gello/4Ch_Bilat/computeJointTorqueRegressor.py)

### 9.4 Parameteridentifikation (Ridge Regression) und „hybrides Modell“

**Im Text:** Reibung + Bias werden datengetrieben identifiziert, nicht ins URDF zurückgeschrieben, sondern als Laufzeit-Erweiterung im Controller genutzt.

**In der Codebase (umgesetzt / vorbereitet):**

- Ridge/Least-Squares-basierte Identifikation ist implementiert:
	- friction-only (empfohlen für low-frequency): `identify_friction_only(...)` in
		- [gello/4Ch_Bilat/computeJointTorqueRegressor.py](gello/4Ch_Bilat/computeJointTorqueRegressor.py)
	- full regressor (für weiterführende Forschung): `identify_dynamics(...)` in
		- [gello/4Ch_Bilat/computeJointTorqueRegressor.py](gello/4Ch_Bilat/computeJointTorqueRegressor.py)

- Das Dokumentationsartefakt zum Hybrid-Ansatz liegt als Standnotiz vor:
	- [HYBRID_FACTR_STATUS.md](HYBRID_FACTR_STATUS.md)

**Offen (noch zu implementieren, minimalinvasiv möglich):**

- Logger/Datenerfassung für (q, dq, tau_meas) als reproduzierbarer Workflow.
- Speichern/Laden der identifizierten Parameter (z.B. `.npz`) und Anwendung im FACTR-Loop.
- Optional: Validierungs-Tooling (tau_pred vs. tau_meas) für Fit-Qualität.

### 9.5 Disturbance Observer (Leader ohne F/T Sensor)

**Im Text:** Trennung intern/extern kann über Modell + Beobachter erfolgen.

**In der Codebase (Custom Extension, prototypisch umgesetzt):**

- Ein erster Leader-Disturbance-Observer existiert als eigenständiges Modul:
	- [gello/4Ch_Bilat/gello_ur5e_observer.py](gello/4Ch_Bilat/gello_ur5e_observer.py)

**Offen:**

- Integration in den FACTR-Loop (z.B. Observer nutzt tau_cmd und geschätzte Modellterme; liefert tau_ext_hat zurück).
- Abstimmung mit tatsächlichen Messgrößen des Leaders (Torque aus Strommessung, Motor-Konstanten, Getriebe, etc.).

### 9.6 UR5e-bezogene FACTR-Erweiterungen (Custom)

Für deine UR5e-Arbeit existieren bereits agent-nahe Komponenten, die als „nicht Original“ zu sehen sind.
Beispiel:

- UR5e-spezifischer FACTR-Agent: [gello/factr/ur5e_gello_factr_agent.py](gello/factr/ur5e_gello_factr_agent.py)

Diese Teile sollten in der Doku als Erweiterung eingeordnet werden (und ggf. im Zuge der Stabilisierung/Tests bereinigt), da die ursprüngliche Architektur generischer Agent/Robot/Env-basiert ist.


