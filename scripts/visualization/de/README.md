# GELLO Visualisierung – Deutsche Plots für Masterarbeit

Dieses Verzeichnis enthält wissenschaftliche Visualisierungsskripte für die GELLO-Teleoperation
mit deutschen Beschriftungen. Alle Skripte erzeugen publikationsreife Plots im SVG-Format
für die Einbettung in LaTeX-Dokumente.

## Voraussetzungen

```bash
pip install numpy pandas matplotlib scipy
```

## Verfügbare Plots

### 1. Reibungskompensation (plot_reibung_kompensation.py)
Zeigt die Reibungscharakteristik (Geschwindigkeit vs. Drehmoment) mit angepasstem Coulomb+viskos Modell.

```bash
python scripts/visualization/de/plot_reibung_kompensation.py logs/gravity_comp.csv --joint 2 --output reibung.svg
```

### 2. Filtervergleich (plot_filter_vergleich.py)
Vergleicht Rohsignal, EMA-Filter und 1€-Filter für externe Drehmomente.

```bash
python scripts/visualization/de/plot_filter_vergleich.py logs/teleop_data.csv --joint 2 --output filter.svg
```

### 3. Driftverhalten / Gravitationskompensation (plot_drift_verhalten.py)
Zeigt das "schwerelose" Armverhalten mit minimaler Drift im Gelenk- und kartesischen Raum.

```bash
python scripts/visualization/de/plot_drift_verhalten.py logs/gravity_hold.csv --kartesisch --output drift.svg
```

### 4. Reglervergleich (plot_regler_vergleich.py)
Vergleicht Positionsregelung vs. Impedanzregelung bei Kontakt mit einer Wand.

```bash
python scripts/visualization/de/plot_regler_vergleich.py logs/position_mode.csv logs/impedance_mode.csv --output regler.svg
```

### 5. Kraftrückkopplung (plot_kraft_rueckkopplung.py)
Zeigt den kausalen Zusammenhang zwischen TCP-Kräften und Leader-Drehmomenten.

```bash
# Einzelne Logdatei mit allen Achsen
python scripts/visualization/de/plot_kraft_rueckkopplung.py logs/force_feedback.csv --output kraft.svg

# Separate Logdateien für X, Y, Z
python scripts/visualization/de/plot_kraft_rueckkopplung.py logs/force_x.csv logs/force_y.csv logs/force_z.csv --output kraft_xyz.svg
```

### 6. Teleoperationsverhalten (plot_teleoperation_verhalten.py)
Zeigt das Tracking zwischen Leader und Follower im Gelenk- und kartesischen Raum.

```bash
python scripts/visualization/de/plot_teleoperation_verhalten.py logs/teleop_data.csv --mode beide --output teleop.svg
```

---

## Datenaufnahme-Plan

Folgende Experimente müssen durchgeführt werden, um alle Plots zu erzeugen:

### Experiment 1: Gravitationskompensation ohne Follower
**Zweck:** Reibungsidentifikation, Filtervergleich, Driftverhalten
**Konfiguration:** `configs/examples/gravity_comp_validation.yaml`
**Dauer:** 60-90 Sekunden
**Durchführung:**
1. Arm in Kalibrierungsposition bringen
2. Logging starten: `--log`
3. 30s still halten (Drift-Test)
4. 30-60s langsam durch Arbeitsraum bewegen (Reibungsidentifikation)
5. Logging beenden

```bash
python gello/factr/gravity_compensation.py --config configs/examples/gravity_comp_validation.yaml --log
```

**Erzeugt Daten für:**
- `plot_reibung_kompensation.py`
- `plot_filter_vergleich.py`  
- `plot_drift_verhalten.py`

---

### Experiment 2: Teleoperation mit Positionsregelung
**Zweck:** Baseline für Reglervergleich
**Konfiguration:** `configs/examples/position_mode_test.yaml`
**Dauer:** 10-20 Sekunden
**Durchführung:**
1. Follower-Roboter starten
2. Teleoperation mit Positionsregelung aktivieren
3. Langsam gegen eine harte Wand fahren
4. Logging beenden

```bash
python gello/factr/gravity_compensation.py --config configs/examples/position_mode_test.yaml --log
```

---

### Experiment 3: Teleoperation mit Impedanzregelung
**Zweck:** Vergleich mit Positionsregelung
**Konfiguration:** `configs/examples/impedance_mode_test.yaml`
**Dauer:** 10-20 Sekunden
**Durchführung:** Wie Experiment 2, aber mit Impedanzregelung

```bash
python gello/factr/gravity_compensation.py --config configs/examples/impedance_mode_test.yaml --log
```

**Erzeugt Daten für:**
- `plot_regler_vergleich.py` (zusammen mit Exp. 2)

---

### Experiment 4: Kraftrückkopplung X-Achse
**Zweck:** Kraftrückkopplungs-Validierung in X-Richtung
**Konfiguration:** `configs/examples/with_force_feedback.yaml`
**Dauer:** 20-40 Sekunden
**Durchführung:**
1. Kraftrückkopplung aktivieren
2. Follower gegen Hindernis in X-Richtung drücken
3. Kurz halten, dann lösen
4. Wiederholen

```bash
python gello/factr/gravity_compensation.py --config configs/examples/with_force_feedback.yaml --log
# Ausgabe umbenennen: mv logs/gello_data_*.csv logs/forceFeedback_x.csv
```

### Experiment 5 & 6: Kraftrückkopplung Y- und Z-Achse
Wie Experiment 4, aber in Y- und Z-Richtung.

**Erzeugt Daten für:**
- `plot_kraft_rueckkopplung.py`

---

### Experiment 7: Teleoperations-Tracking
**Zweck:** Leader-Follower Tracking-Verhalten
**Konfiguration:** `configs/examples/impedance_mode_test.yaml`
**Dauer:** 30-60 Sekunden
**Durchführung:**
1. Teleoperation starten
2. Verschiedene Bewegungen im Arbeitsraum ausführen
3. Langsame und schnelle Bewegungen mischen

**Erzeugt Daten für:**
- `plot_teleoperation_verhalten.py`

---

## Neue Spalten im Log

Die folgenden Spalten wurden zum Logger hinzugefügt:

| Spalte | Beschreibung | Einheit |
|--------|--------------|---------|
| `tcp_x_leader`, `tcp_y_leader`, `tcp_z_leader` | TCP-Position Leader (FK) | m |
| `tcp_x_follower`, `tcp_y_follower`, `tcp_z_follower` | TCP-Position Follower | m |
| `tau_external_raw_{i}` | Ungefilterte externe Drehmomente | Nm |
| `tau_external_ema_{i}` | EMA-gefilterte externe Drehmomente | Nm |
| `tau_external_oneeuro_{i}` | 1€-gefilterte externe Drehmomente | Nm |

---

## Plot-Übersicht für Masterarbeit

| Abschnitt | Plot | Datei | Experiment |
|-----------|------|-------|------------|
| 4.2.1 | Reibungsidentifikation | `reibung_kompensation.svg` | Exp. 1 |
| 4.2.2 | Filtervergleich | `filter_vergleich.svg` | Exp. 1 |
| 4.3.1 | Gravitationskompensation | `drift_verhalten.svg` | Exp. 1 |
| 4.3.2 | Kraftrückkopplung | `kraft_rueckkopplung.svg` | Exp. 4-6 |
| 4.3.3 | Reglervergleich | `regler_vergleich.svg` | Exp. 2-3 |
| 4.3.4 | Teleoperation | `teleop_verhalten.svg` | Exp. 7 |

---

## Tipps für gute Datenqualität

1. **Reibungsidentifikation:** Gleichmäßige, langsame Bewegungen in beide Richtungen
2. **Drift-Test:** Absolut stillhalten für mindestens 30 Sekunden
3. **Kontaktkräfte:** Langsam gegen Hindernis fahren, nicht ruckartig
4. **Tracking:** Variation von schnellen und langsamen Bewegungen
5. **Kraftrückkopplung:** Deutliche Kontakte mit verschiedenen Kräften

## Beispiel-Workflow

```bash
# 1. Alle Experimente durchführen und Logs erzeugen
# (siehe Experiment-Beschreibungen oben)

# 2. Logs in das Visualisierungsverzeichnis kopieren
mkdir -p scripts/visualization/de/logs
cp gello/factr/logs/*.csv scripts/visualization/de/logs/

# 3. Alle Plots erzeugen
cd scripts/visualization/de
python plot_reibung_kompensation.py logs/gravity_comp.csv --output ../plots/reibung.svg
python plot_filter_vergleich.py logs/gravity_comp.csv --output ../plots/filter.svg
python plot_drift_verhalten.py logs/gravity_comp.csv --kartesisch --output ../plots/drift.svg
python plot_regler_vergleich.py logs/position_mode.csv logs/impedance_mode.csv --output ../plots/regler.svg
python plot_kraft_rueckkopplung.py logs/force_x.csv logs/force_y.csv logs/force_z.csv --output ../plots/kraft.svg
python plot_teleoperation_verhalten.py logs/teleop_data.csv --output ../plots/teleop.svg
```
