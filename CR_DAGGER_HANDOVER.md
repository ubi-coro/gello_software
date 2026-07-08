# CR-DAgger Hardware-Uebergabe

Dieses Dokument beschreibt den aktuellen GELLO + UR5e CR-DAgger Aufbau fuer die Uebergabe an einen neuen Operator. Der Schwerpunkt liegt auf sicherem Startup, den Konfigurationsdateien von `gello/cr_dagger/scripts/run_cr_dagger_collection.py` und den Hardware-/URDF-Dateien, die zusammenpassen muessen.

## Aktive Architektur

Der validierte Phase-B-Pfad ist BOTA-SE(3)-Residual-Control:

```text
q_cmd_ur5e = q_ref_policy + delta_human
```

Der GELLO ist in Phase B nicht die direkte UR5e-Befehlsquelle. Er dient als haptisches Mirror-Device und als mechanischer Traeger fuer den MiniOne-Sensor. Die Policy laeuft asynchron ueber Shared Memory; der schnelle Hardwarepfad darf nicht auf Inferenz warten.

## Sicherer Startup

1. UR5e in die definierte Referenz-Startpose fahren.
2. GELLO in `arm_teleop.initialization.calibration_joint_pos` der verwendeten YAML halten.
3. Beachten: Das letzte Handgelenk kann in dieser Pose knapp ueber oder knapp unter `0 rad` stehen. Das ist normal; nicht manuell eine andere Umdrehungs-Branch erzwingen.
4. Nach Umbauten, Reboots oder unklarem Startzustand zuerst den read-only Disambiguation-Check ausfuehren:

```bash
python scripts/turn_disambiguation.py read --config configs/ur5e_gello_factr_hw_V3_PhaseB.yaml
```

5. Beim Phase-B-Startup den GELLO am letzten Gelenk ueber dem BOTA MiniOne abstuetzen. Das Skript sammelt BOTA-Warmup und Software-Bias, bevor der Leader in Position-Hold geht. In diesem Fenster darf der Sensor nicht belastet werden und der GELLO darf nicht auf den Sensor absacken.
6. Collection erst starten, wenn im Terminal gemeldet wird, dass der Phase-B BOTA-SE3 Loop armed ist.

`legacy_auto_align` nicht als schnelle Loesung fuer einen Startup-Mismatch verwenden. Die kalibrierten Offsets sind absichtlich gesetzt; der sichere Pfad ist Offset-/Turn-Pruefung mit `turn_disambiguation.py`.

## Wichtige Config-Dateien

`configs/ur5e_gello_factr_hw_V3_PhaseA.yaml`

Phase-A-Demonstrationsconfig. Nutzt `gello/factr/urdf/gello_urdf_minione/robot.urdf`, niedrige Leader-Gravcomp und direkte UR5e-Impedance-Teleop fuer menschlich gefuehrte Demonstrationen.

`configs/ur5e_gello_factr_hw_V3_PhaseB.yaml`

Phase-B-Interventionsconfig. Nutzt `gello/factr/urdf/gello_urdf_minione_sensorFrame/robot.urdf`, weil `cr_dagger_defaults.yaml` den BOTA-Frame `bota_sensor_frame` auswaehlt. Gravcomp muss fuer Policy-Tracking hoch genug bleiben.

`gello/cr_dagger/config/cr_dagger_defaults.yaml`

CR-DAgger Runtime-Defaults, geladen ueber `--cr-dagger-defaults`. Wichtige `phase_b`-Bloecke:

- `follower`: UR5e-Impedance-Gains und Arming-Rampe.
- `leader_position`: Dynamixel-Position-Hold-Handoff nach BOTA-Biasing.
- `bota`: MiniOne-Treiberpfad, Warmup, Bias, Kalibrierung, Frame, Admittance und Contact-Scaling.
- `residual`: Limits fuer `delta_human`.
- `intervention`: geloggte Contact-/Delta-Schwellen.

`gello/cr_dagger/config/bota_minione_wrench_calibration_identity.yaml`

Statische orientierungsabhaengige BOTA-Wrench-Kompensation. Sie wird aus No-Contact-Pose-Sweeps erzeugt und geladen, wenn `phase_b.bota.wrench_calibration_enable` true ist.

## Wichtige Einstellparameter

Die meisten Stellgroessen fuer Phase B liegen in `gello/cr_dagger/config/cr_dagger_defaults.yaml` unter `phase_b`. Werte immer nur einzeln aendern und danach mit kurzer, beobachtbarer Validierung testen.

### Sensor-Masking und Wrench-Aufbereitung

`phase_b.bota.axis_mask`

Maskiert die 6D-SE(3)-Admittance-Achsen in der Reihenfolge `[x, y, z, rx, ry, rz]`. Aktuell `[1, 1, 1, 0, 0, 0]`: nur translatorische Korrekturen sind aktiv, Rotationen sind gesperrt. Fuer erste Tests Masking konservativ lassen. Rotationsachsen erst aktivieren, wenn Sensorframe, URDF und Vorzeichen sicher validiert sind.

`phase_b.bota.deadband`

Totband fuer den konditionierten Wrench in Base-Koordinaten, Reihenfolge `[Fx, Fy, Fz, Tx, Ty, Tz]`. Hoehere Werte ignorieren kleine Bias-/Rauschanteile, machen kleine menschliche Korrekturen aber unsensibler. Zu niedrige Werte fuehren zu Drift oder ungewollter Residualbewegung.

`phase_b.bota.saturation`

Hartes Limit fuer konditionierte Kraefte und Momente. Das ist ein Sicherheitslimit gegen Peaks. Nicht als normalen Gain-Ersatz verwenden.

`phase_b.bota.filter_cutoff_hz` und `filter_alpha`

Filtern den Base-Wrench. `filter_cutoff_hz` ist die normale Einstellung; `filter_alpha: 0.0` bedeutet, dass die cutoff-basierte Filterung genutzt wird. Niedriger Cutoff beruhigt das Signal, erhoeht aber Latenz.

`phase_b.bota.base_axis_map`, `base_axis_signs`, `wrench_sign`

Achsen- und Vorzeichenkorrektur nach Rotation in die Base. Bei falscher Richtung nicht sofort Gains aendern, sondern zuerst diese Zuordnung und das Sensorframe pruefen. Fuer den aktuellen sensorFrame-URDF-Stand bleiben Map und Signs identity.

`phase_b.bota.wrench_calibration_enable` und `wrench_calibration`

Aktiviert die statische MiniOne-Wrench-Kompensation. Wenn die Sensorhalterung, der Sensor oder das URDF-Frame geaendert wurden, muss die Kalibrierung neu validiert oder neu gefittet werden.

### SE(3)-Admittance

`phase_b.bota.cart_mass`, `cart_damp`, `cart_stiff`

Virtuelle kartesische Dynamik fuer die aus dem BOTA-Wrench erzeugte Task-Space-Korrektur. Hoehere Daempfung reduziert Schwingen. Hoehere Steifigkeit zieht den Offset staerker zurueck. Niedrigere Steifigkeit haelt Korrekturen laenger. Nicht mit UR5e-Impedance-Gains verwechseln.

`phase_b.bota.cart_max`

Maximaler kartesischer Offset pro Achse. Aktuell sind nur Translationen bis `0.035 m` erlaubt, Rotationen stehen auf `0.0`. Das ist ein zentrales Sicherheitslimit.

`phase_b.bota.dls_damping`

Damping fuer die damped-least-squares Rueckprojektion von Task-Space-Delta nach Joint-Delta. Hoeher ist konservativer und reduziert starke Gelenkbewegungen nahe schlechter Konditionierung, macht Korrekturen aber weicher.

`phase_b.bota.contact_force_scale` und `contact_torque_scale`

Skalieren die Contact-Probability aus Kraft/Moment. Diese Werte beeinflussen vor allem Intervention-Logging und Release-Verhalten, nicht direkt den Wrench selbst.

### Residual-Limits

`phase_b.residual.delta_max`

Maximale menschliche Joint-Korrektur `delta_human` pro Gelenk nach Limiter. Das begrenzt direkt `q_cmd_ur5e = q_ref_policy + delta_human`. Bei zu kleinem Wert kann der Mensch nicht genug korrigieren; bei zu grossem Wert steigen Sprung- und Label-Risiken.

`phase_b.residual.delta_rate_max`

Maximale Aufbaugeschwindigkeit der Korrektur. Dieser Parameter verhindert harte Beschleunigungsspitzen am UR5e.

`phase_b.residual.delta_release_rate_max` und `delta_release_tau_s`

Bestimmen, wie schnell eine Korrektur nach Kontaktende wieder auf null abklingt. Zu schnelles Release erzeugt Rueckspruenge; zu langsames Release laesst die Policy laenger als gewollt versetzt fahren.

`phase_b.residual.delta_release_contact_threshold`

Contact-Probability-Schwelle fuer Release-Logik. Darunter darf der Residualanteil abgebaut werden.

### Follower- und Leader-Handoff

`phase_b.follower.kp`, `kd`

UR5e Direct-Torque-Impedance-Gains. Erst Gravcomp und Startup-Alignment validieren, dann diese Werte anfassen. Ein Startup-Mismatch darf nicht durch weiche oder harte Gains kaschiert werden.

`phase_b.follower.handoff_ramp_s`

Rampe vom gemessenen UR5e-Zustand zum ersten Ziel. Schuetzt gegen kleine Arming-Schritte, ist aber keine Rettung fuer eine falsche Startpose.

`phase_b.leader_position.current_limit`, `goal_current`, `position_*_gains`, `velocity_*_gains`

Dynamixel-Position-Hold-Konfiguration fuer den GELLO in Phase B. Das haelt den Leader/Mirror nach der BOTA-Bias-Phase. Aenderungen hier koennen Sag, Ruckeln oder Sensorbelastung verursachen.

`phase_b.leader_position.recover_pre_handoff`, `recover_s`, `recover_max_delta`

Kompensiert kleinen synthetischen Handoff-Sag. Wenn `recover_max_delta` ueberschritten wird, wird Recovery uebersprungen. Das ist ein Warnsignal fuer mechanisches Halten, Startup-Pose oder zu grosse Bewegung waehrend Bias/Handoff.

### Intervention-Logging

`phase_b.intervention.contact_threshold`

Schwelle fuer das geloggte Contact-Intervention-Bit aus BOTA-Kontakt.

`phase_b.intervention.delta_threshold`

Schwelle fuer das geloggte Delta-Intervention-Bit aus `delta_human`. Diese Werte beeinflussen Labels/Metadaten; sie ersetzen nicht die physikalischen Residual-Limits.

### Hardware-YAML-Parameter

`controller.gravity_comp.gain` und `gain_per_joint` in `configs/ur5e_gello_factr_hw_V3_PhaseB.yaml`

Fuer Phase B hoch halten. Wenn der Leader sein Gewicht nicht sauber kompensiert, trackt die Impedance nicht nur Fehler, sondern auch Gravitation; das kann Schwingen erzeugen.

`teleop.mapping.offsets`, `alignment_mode`, `turn_disambiguate_offsets`, `turn_safe_targets`

Kalibrierte Leader->Follower-Zuordnung. `alignment_mode: turn_disambiguate` behalten. Bei Mismatch `scripts/turn_disambiguation.py read` nutzen, nicht `legacy_auto_align` aktivieren.

`arm_teleop.initialization.joint_offsets`, `calibration_joint_pos`

Permanent-Offsets und Startup-Anker fuer den GELLO. Diese Werte sind sicherheitskritisch, weil sie die Multi-Turn-Decodierung bestimmen.

## Runtime-Beispiele

Phase A Demonstrationssammlung:

```bash
python gello/cr_dagger/scripts/run_cr_dagger_collection.py ^
  --config configs/ur5e_gello_factr_hw_V3_PhaseA.yaml ^
  --lerobot-repo operator/cr_dagger_phase_a
```

Phase B Interventionssammlung:

```bash
python gello/cr_dagger/scripts/run_cr_dagger_collection.py ^
  --interventions ^
  --config configs/ur5e_gello_factr_hw_V3_PhaseB.yaml ^
  --cr-dagger-defaults gello/cr_dagger/config/cr_dagger_defaults.yaml ^
  --lerobot-repo operator/cr_dagger_phase_b
```

Unter Linux-Shells `^` durch `\` ersetzen.

## URDFs und Hardware-Dateien

- `gello/factr/urdf/gello_urdf_minione/robot.urdf`: GELLO-Modell ohne expliziten BOTA-Sensorframe, verwendet fuer Phase A.
- `gello/factr/urdf/gello_urdf_minione_sensorFrame/robot.urdf`: GELLO-Modell mit `bota_sensor_frame`, erforderlich fuer Phase B.
- `configs/bota_binary.json`: BOTA-Treiberconfig aus den Defaults. Diese Datei muss auf der Runtime-Maschine vorhanden sein.
- `scripts/turn_disambiguation.py`: read-only Pruefung fuer permanente Offsets, per-boot Turn-Branches und Leader->Follower-Mapping.
- `scripts/gello_get_offset.py`: Helper zum Neuberechnen permanenter Leader-Offsets nach Hardwareaenderungen.
- `scripts/bota_minione.py` und `scripts/fit_bota_wrench_calibration.py`: Aufzeichnen und Fitten der BOTA-Wrench-Kalibrierung.

## Fehlersuche

- GELLO und UR5e liegen auf einem Gelenk um ca. `2*pi` auseinander: `turn_disambiguation.py read` ausfuehren; Signs/Offsets nicht blind aendern.
- Phase B fuehlt sich schwer an oder oszilliert: zuerst Leader-Gravcomp validieren, danach Impedance-Gains.
- BOTA-Korrekturen driften direkt nach Startup: Startup wiederholen und Sensor waehrend `warmup_s` und `bias_s` unbelastet halten.
- `bota_binary.json` fehlt: BOTA-Treiberconfig auf der Runtime-Maschine bereitstellen, bevor Phase B gestartet wird.
- UR5e springt beim Arming: stoppen und pruefen, ob UR5e und GELLO vor dem Start wirklich in der konfigurierten Startpose standen.
