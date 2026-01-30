#!/usr/bin/env python3
"""
Gravitationskompensation – Driftverhalten (Abschnitt 4.3.1)

Zeigt das "schwerelose" Armverhalten: Bei aktivierter Gravitationskompensation
sollte der Arm in jeder Position stabil bleiben (minimale Drift).

Verwendung:
    python plot_drift_verhalten.py <logfile.csv> --output drift_verhalten.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Deutsche Schrifteinstellungen
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'svg'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.axisbelow'] = True

# Deutsche Gelenknamen
GELENK_NAMEN = {
    0: "Gelenk 1 (Basis)",
    1: "Gelenk 2 (Schulter)",
    2: "Gelenk 3 (Ellbogen)",
    3: "Gelenk 4 (Handgelenk 1)",
    4: "Gelenk 5 (Handgelenk 2)",
    5: "Gelenk 6 (Handgelenk 3)",
}


def main():
    parser = argparse.ArgumentParser(description="Driftverhalten der Gravitationskompensation")
    parser.add_argument("logfile", type=str, help="Pfad zur CSV-Logdatei")
    parser.add_argument("--output", type=str, default="drift_verhalten.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Zeitfenster [Start, Ende] in Sekunden")
    parser.add_argument("--joints", type=int, nargs='+', default=[1, 2, 3],
                        help="Gelenke für Plot (Standard: 1 2 3 = Schulter, Ellbogen, Handgelenk1)")
    parser.add_argument("--kartesisch", action="store_true",
                        help="TCP-Position im kartesischen Raum anzeigen")
    args = parser.parse_args()

    # Daten laden
    print(f"Lade Daten aus {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Zeitstempel in relative Zeit umwandeln
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Zeitfenster anwenden
    if args.time_window:
        t_start, t_end = args.time_window
        df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
    
    print(f"Analysiere {len(df)} Messpunkte über {df['time'].iloc[-1]:.1f} Sekunden")
    
    farben = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    if args.kartesisch and 'tcp_x_leader' in df.columns:
        # Kartesische Darstellung (2x1 Layout)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        zeit = df['time'].values
        
        # Oberer Plot: Gelenkpositionen
        for idx, gelenk_idx in enumerate(args.joints):
            if gelenk_idx > 5:
                continue
            
            position_rad = df[f'q_leader_{gelenk_idx}'].values
            position_grad = np.rad2deg(position_rad)
            
            # Drift berechnen
            start_pos = position_grad[0]
            drift = position_grad - start_pos
            max_drift = np.max(np.abs(drift))
            
            ax1.plot(zeit, position_grad, linewidth=2.0, alpha=0.85,
                     color=farben[idx % len(farben)],
                     label=f'{GELENK_NAMEN.get(gelenk_idx, f"Gelenk {gelenk_idx}")} (Drift: {max_drift:.2f}°)')
            
            # Horizontale Linie bei Startposition
            ax1.axhline(start_pos, color=farben[idx % len(farben)],
                        linewidth=0.8, linestyle='--', alpha=0.4)
        
        ax1.set_ylabel('Gelenkposition [°]', fontsize=14)
        ax1.set_title('Gravitationskompensation: Driftverhalten im Gelenkraum',
                      fontsize=16, pad=15, fontweight='bold')
        ax1.legend(loc='best', fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        
        # Unterer Plot: Kartesische TCP-Position
        tcp_x = df['tcp_x_leader'].values * 1000  # in mm
        tcp_y = df['tcp_y_leader'].values * 1000
        tcp_z = df['tcp_z_leader'].values * 1000
        
        # Drift vom Startpunkt berechnen
        drift_x = tcp_x - tcp_x[0]
        drift_y = tcp_y - tcp_y[0]
        drift_z = tcp_z - tcp_z[0]
        
        ax2.plot(zeit, drift_x, linewidth=2.0, alpha=0.85, color='#e74c3c',
                 label=f'X (max: {np.max(np.abs(drift_x)):.2f} mm)')
        ax2.plot(zeit, drift_y, linewidth=2.0, alpha=0.85, color='#3498db',
                 label=f'Y (max: {np.max(np.abs(drift_y)):.2f} mm)')
        ax2.plot(zeit, drift_z, linewidth=2.0, alpha=0.85, color='#2ecc71',
                 label=f'Z (max: {np.max(np.abs(drift_z)):.2f} mm)')
        
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Zeit [s]', fontsize=14)
        ax2.set_ylabel('TCP-Drift [mm]', fontsize=14)
        ax2.set_title('Kartesische Drift der TCP-Position', fontsize=14, pad=10)
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        ax2.grid(True, alpha=0.3)
        
        # Gesamtdrift berechnen
        total_drift = np.sqrt(drift_x**2 + drift_y**2 + drift_z**2)
        max_total_drift = np.max(total_drift)
        
        # Textbox mit Zusammenfassung
        zusammenfassung = (
            f'Maximale Drift:\n'
            f'  X: {np.max(np.abs(drift_x)):.2f} mm\n'
            f'  Y: {np.max(np.abs(drift_y)):.2f} mm\n'
            f'  Z: {np.max(np.abs(drift_z)):.2f} mm\n'
            f'  Gesamt: {max_total_drift:.2f} mm'
        )
        ax2.text(0.98, 0.02, zusammenfassung, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
    else:
        # Nur Gelenkraum-Darstellung
        fig, ax = plt.subplots(figsize=(12, 7))
        
        zeit = df['time'].values
        
        for idx, gelenk_idx in enumerate(args.joints):
            if gelenk_idx > 5:
                print(f"Warnung: Gelenk {gelenk_idx} außerhalb des Bereichs, überspringe")
                continue
            
            position_rad = df[f'q_leader_{gelenk_idx}'].values
            position_grad = np.rad2deg(position_rad)
            
            # Drift berechnen
            start_pos = position_grad[0]
            drift = position_grad - start_pos
            max_drift = np.max(np.abs(drift))
            
            ax.plot(zeit, position_grad, linewidth=2.0, alpha=0.85,
                    color=farben[idx % len(farben)],
                    label=f'{GELENK_NAMEN.get(gelenk_idx, f"Gelenk {gelenk_idx}")} (Drift: {max_drift:.2f}°)')
            
            # Horizontale Linie bei Startposition
            ax.axhline(start_pos, color=farben[idx % len(farben)],
                       linewidth=0.8, linestyle='--', alpha=0.4)
        
        ax.set_xlabel('Zeit [s]', fontsize=14)
        ax.set_ylabel('Gelenkposition [°]', fontsize=14)
        ax.set_title('Gravitationskompensation: Positionshaltetest',
                     fontsize=16, pad=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # Erklärungsbox
        erklaerung = (
            'Ideales Verhalten: Horizontale Linien (keine Drift)\n'
            'Übermäßige Drift deutet auf unzureichende\n'
            'Kompensation oder Modellungenauigkeiten hin'
        )
        ax.text(0.02, 0.98, erklaerung, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Speichern
    ausgabe_pfad = Path(args.output)
    plt.savefig(ausgabe_pfad)
    print(f"Plot gespeichert: {ausgabe_pfad}")
    
    png_pfad = ausgabe_pfad.with_suffix('.png')
    plt.savefig(png_pfad, dpi=150)
    print(f"Vorschau gespeichert: {png_pfad}")
    
    # Drift-Statistiken ausgeben
    print("\n=== Drift-Statistiken (Maximale Absolute Abweichung) ===")
    for gelenk_idx in args.joints:
        if gelenk_idx <= 5:
            position_rad = df[f'q_leader_{gelenk_idx}'].values
            position_grad = np.rad2deg(position_rad)
            drift = position_grad - position_grad[0]
            max_drift = np.max(np.abs(drift))
            print(f"{GELENK_NAMEN.get(gelenk_idx, f'Gelenk {gelenk_idx}'):20s}: {max_drift:.3f}°")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
