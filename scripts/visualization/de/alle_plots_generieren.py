#!/usr/bin/env python3
"""
Alle deutschen Plots für die Masterarbeit generieren.

Verwendet die Logdateien im logs/ Verzeichnis und erzeugt
alle SVG-Plots im plots/ Verzeichnis.

Verwendung:
    python alle_plots_generieren.py [--log-dir LOGS] [--output-dir PLOTS]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Alle deutschen Plots generieren")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Verzeichnis mit Logdateien")
    parser.add_argument("--output-dir", type=str, default="../plots",
                        help="Ausgabeverzeichnis für Plots")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot-Konfiguration: (Script, Argumente, Ausgabedatei)
    plots = []
    
    # 1. Reibungskompensation
    reibung_log = log_dir / "gravity_comp.csv"
    if not reibung_log.exists():
        # Fallback auf andere Namen
        for name in ["gravity_hold.csv", "friction_data.csv"]:
            if (log_dir / name).exists():
                reibung_log = log_dir / name
                break
    
    if reibung_log.exists():
        plots.append((
            "plot_reibung_kompensation.py",
            [str(reibung_log), "--joint", "2"],
            output_dir / "reibung_kompensation.svg"
        ))
        
        # 2. Filtervergleich (gleiche Daten)
        plots.append((
            "plot_filter_vergleich.py",
            [str(reibung_log), "--joint", "2"],
            output_dir / "filter_vergleich.svg"
        ))
        
        # 3. Driftverhalten
        plots.append((
            "plot_drift_verhalten.py",
            [str(reibung_log), "--joints", "1", "2", "3", "--kartesisch"],
            output_dir / "drift_verhalten.svg"
        ))
    else:
        print(f"⚠ Keine Gravitationskompensations-Logdatei gefunden in {log_dir}")
    
    # 4. Reglervergleich
    pos_log = log_dir / "position_mode.csv"
    imp_log = log_dir / "impedance_mode.csv"
    
    # Alternative Namen prüfen
    if not pos_log.exists():
        for name in ["teleop_movement_pos.csv", "position_teleop.csv"]:
            if (log_dir / name).exists():
                pos_log = log_dir / name
                break
    
    if not imp_log.exists():
        for name in ["teleop_movement_imp.csv", "impedance_teleop.csv"]:
            if (log_dir / name).exists():
                imp_log = log_dir / name
                break
    
    if pos_log.exists() and imp_log.exists():
        plots.append((
            "plot_regler_vergleich.py",
            [str(pos_log), str(imp_log), "--tcp-axis", "2"],
            output_dir / "regler_vergleich.svg"
        ))
    else:
        print(f"⚠ Regler-Vergleichsdaten nicht gefunden (position_mode.csv, impedance_mode.csv)")
    
    # 5. Kraftrückkopplung
    force_x = log_dir / "forceFeedback_x.csv"
    force_y = log_dir / "forceFeedback_y.csv"
    force_z = log_dir / "forceFeedback_z.csv"
    
    # Alternative Namen
    if not force_x.exists():
        for pattern in ["forceFeedback_pos_x.csv", "forceFeedback_imp_x.csv"]:
            if (log_dir / pattern).exists():
                force_x = log_dir / pattern
                break
    if not force_y.exists():
        for pattern in ["forceFeedback_pos_y.csv", "forceFeedback_imp_y.csv"]:
            if (log_dir / pattern).exists():
                force_y = log_dir / pattern
                break
    if not force_z.exists():
        for pattern in ["forceFeedback_pos_z.csv", "forceFeedback_imp_z.csv"]:
            if (log_dir / pattern).exists():
                force_z = log_dir / pattern
                break
    
    if force_x.exists() and force_y.exists() and force_z.exists():
        plots.append((
            "plot_kraft_rueckkopplung.py",
            [str(force_x), str(force_y), str(force_z)],
            output_dir / "kraft_rueckkopplung.svg"
        ))
    elif force_x.exists():
        # Nur eine Achse verfügbar
        plots.append((
            "plot_kraft_rueckkopplung.py",
            [str(force_x)],
            output_dir / "kraft_rueckkopplung.svg"
        ))
    else:
        print(f"⚠ Kraftrückkopplungs-Daten nicht gefunden")
    
    # 6. Teleoperationsverhalten
    teleop_log = log_dir / "teleop_data.csv"
    if not teleop_log.exists():
        # Versuche andere Namen
        for name in ["teleop_movement_imp.csv", "impedance_mode.csv", "gello_data_*.csv"]:
            matches = list(log_dir.glob(name))
            if matches:
                teleop_log = matches[0]
                break
    
    if teleop_log.exists():
        plots.append((
            "plot_teleoperation_verhalten.py",
            [str(teleop_log), "--mode", "beide"],
            output_dir / "teleoperation_verhalten.svg"
        ))
    else:
        print(f"⚠ Teleoperation-Logdatei nicht gefunden")
    
    # Plots generieren
    print(f"\n{'='*60}")
    print(f"Generiere {len(plots)} Plots...")
    print(f"{'='*60}\n")
    
    erfolge = 0
    fehler = 0
    
    for script, args_list, output in plots:
        print(f"📊 {script} → {output.name}")
        cmd = [sys.executable, str(script_dir / script)] + args_list + ["--output", str(output)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"   ✓ Erfolgreich")
                erfolge += 1
            else:
                print(f"   ✗ Fehler: {result.stderr[:200]}")
                fehler += 1
        except subprocess.TimeoutExpired:
            print(f"   ✗ Timeout")
            fehler += 1
        except Exception as e:
            print(f"   ✗ Ausnahme: {e}")
            fehler += 1
    
    print(f"\n{'='*60}")
    print(f"Fertig: {erfolge} erfolgreich, {fehler} fehlgeschlagen")
    print(f"Plots in: {output_dir.absolute()}")
    print(f"{'='*60}")
    
    return 0 if fehler == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
