#!/usr/bin/env python3
"""
Teleoperationsverhalten – Visualisierung (Abschnitt 4.3)

Zeigt das Tracking-Verhalten zwischen Leader und Follower während der
Teleoperation sowohl im Gelenkraum als auch im kartesischen Raum.

Verwendung:
    python plot_teleoperation_verhalten.py <logfile.csv> --output teleop_verhalten.svg
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

GELENK_NAMEN = {
    0: "Gelenk 1 (Basis)",
    1: "Gelenk 2 (Schulter)",
    2: "Gelenk 3 (Ellbogen)",
    3: "Gelenk 4 (Handgelenk 1)",
    4: "Gelenk 5 (Handgelenk 2)",
    5: "Gelenk 6 (Handgelenk 3)",
}


def main():
    parser = argparse.ArgumentParser(description="Teleoperationsverhalten visualisieren")
    parser.add_argument("logfile", type=str, help="Pfad zur CSV-Logdatei")
    parser.add_argument("--output", type=str, default="teleop_verhalten.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Zeitfenster [Start, Ende] in Sekunden")
    parser.add_argument("--joints", type=int, nargs='+', default=[1, 2, 3],
                        help="Gelenke für Gelenkraum-Plot (Standard: 1 2 3)")
    parser.add_argument("--mode", type=str, choices=['kartesisch', 'gelenk', 'beide'],
                        default='beide', help="Darstellungsmodus")
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
    
    # Prüfen ob kartesische Daten verfügbar
    hat_kartesisch = 'tcp_x_leader' in df.columns and 'tcp_x_follower' in df.columns
    
    if args.mode == 'kartesisch' and not hat_kartesisch:
        print("Warnung: Kartesische Daten nicht verfügbar, verwende Gelenkraum")
        args.mode = 'gelenk'
    
    zeit = df['time'].values
    farben_leader = ['#e74c3c', '#f39c12', '#9b59b6']  # Rot, Orange, Violett
    farben_follower = ['#3498db', '#2ecc71', '#1abc9c']  # Blau, Grün, Türkis
    
    if args.mode == 'beide' and hat_kartesisch:
        # Layout: 2 Zeilen (Kartesisch oben, Gelenkraum unten)
        fig, (ax_kart, ax_gelenk) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        # === Oberer Plot: Kartesischer Raum (X, Y, Z) ===
        # Leader
        tcp_x_leader = df['tcp_x_leader'].values * 1000  # in mm
        tcp_y_leader = df['tcp_y_leader'].values * 1000
        tcp_z_leader = df['tcp_z_leader'].values * 1000
        
        # Follower
        tcp_x_follower = df['tcp_x_follower'].values * 1000
        tcp_y_follower = df['tcp_y_follower'].values * 1000
        tcp_z_follower = df['tcp_z_follower'].values * 1000
        
        # X-Achse
        ax_kart.plot(zeit, tcp_x_leader, linewidth=2.0, alpha=0.85, color=farben_leader[0],
                     linestyle='-', label='Leader X')
        ax_kart.plot(zeit, tcp_x_follower, linewidth=2.0, alpha=0.85, color=farben_follower[0],
                     linestyle='--', label='Follower X')
        
        # Y-Achse
        ax_kart.plot(zeit, tcp_y_leader, linewidth=2.0, alpha=0.85, color=farben_leader[1],
                     linestyle='-', label='Leader Y')
        ax_kart.plot(zeit, tcp_y_follower, linewidth=2.0, alpha=0.85, color=farben_follower[1],
                     linestyle='--', label='Follower Y')
        
        # Z-Achse
        ax_kart.plot(zeit, tcp_z_leader, linewidth=2.0, alpha=0.85, color=farben_leader[2],
                     linestyle='-', label='Leader Z')
        ax_kart.plot(zeit, tcp_z_follower, linewidth=2.0, alpha=0.85, color=farben_follower[2],
                     linestyle='--', label='Follower Z')
        
        ax_kart.set_ylabel('TCP-Position [mm]', fontsize=14)
        ax_kart.set_title('Teleoperationsverhalten: Kartesischer Raum',
                          fontsize=16, pad=15, fontweight='bold')
        ax_kart.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.95)
        ax_kart.grid(True, alpha=0.3)
        
        # Tracking-Fehler berechnen und anzeigen
        fehler_x = np.mean(np.abs(tcp_x_leader - tcp_x_follower))
        fehler_y = np.mean(np.abs(tcp_y_leader - tcp_y_follower))
        fehler_z = np.mean(np.abs(tcp_z_leader - tcp_z_follower))
        
        stats_text = (
            f'Mittlerer Tracking-Fehler:\n'
            f'  ΔX: {fehler_x:.2f} mm\n'
            f'  ΔY: {fehler_y:.2f} mm\n'
            f'  ΔZ: {fehler_z:.2f} mm'
        )
        ax_kart.text(0.02, 0.98, stats_text, transform=ax_kart.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # === Unterer Plot: Gelenkraum ===
        for idx, gelenk in enumerate(args.joints[:3]):  # Maximal 3 Gelenke
            # Leader
            q_leader = np.rad2deg(df[f'q_leader_{gelenk}'].values)
            ax_gelenk.plot(zeit, q_leader, linewidth=2.0, alpha=0.85,
                           color=farben_leader[idx % len(farben_leader)],
                           linestyle='-', label=f'Leader {GELENK_NAMEN.get(gelenk, f"G{gelenk}")}')
            
            # Follower
            if f'q_follower_{gelenk}' in df.columns:
                q_follower = np.rad2deg(df[f'q_follower_{gelenk}'].values)
                ax_gelenk.plot(zeit, q_follower, linewidth=2.0, alpha=0.85,
                               color=farben_follower[idx % len(farben_follower)],
                               linestyle='--', label=f'Follower {GELENK_NAMEN.get(gelenk, f"G{gelenk}")}')
        
        ax_gelenk.set_xlabel('Zeit [s]', fontsize=14)
        ax_gelenk.set_ylabel('Gelenkposition [°]', fontsize=14)
        ax_gelenk.set_title('Teleoperationsverhalten: Gelenkraum', fontsize=14, pad=10)
        ax_gelenk.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.95)
        ax_gelenk.grid(True, alpha=0.3)
        
    elif args.mode == 'kartesisch' and hat_kartesisch:
        # Nur kartesisch: 3 Subplots (X, Y, Z)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        achsen = ['x', 'y', 'z']
        achsen_labels = ['X', 'Y', 'Z']
        
        for idx, (achse, label) in enumerate(zip(achsen, achsen_labels)):
            leader = df[f'tcp_{achse}_leader'].values * 1000
            follower = df[f'tcp_{achse}_follower'].values * 1000
            
            axes[idx].plot(zeit, leader, linewidth=2.0, alpha=0.85, color='#e74c3c',
                           label='Leader')
            axes[idx].plot(zeit, follower, linewidth=2.0, alpha=0.85, color='#3498db',
                           linestyle='--', label='Follower')
            
            axes[idx].set_ylabel(f'{label}-Position [mm]', fontsize=13)
            axes[idx].legend(loc='upper right', fontsize=11, framealpha=0.95)
            axes[idx].grid(True, alpha=0.3)
            
            # Tracking-Fehler
            fehler = np.mean(np.abs(leader - follower))
            axes[idx].text(0.02, 0.95, f'Mittlerer Fehler: {fehler:.2f} mm',
                           transform=axes[idx].transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[0].set_title('Teleoperationsverhalten: Kartesischer Raum',
                          fontsize=16, pad=15, fontweight='bold')
        axes[-1].set_xlabel('Zeit [s]', fontsize=14)
        
    else:
        # Nur Gelenkraum
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for idx, gelenk in enumerate(args.joints):
            # Leader
            q_leader = np.rad2deg(df[f'q_leader_{gelenk}'].values)
            ax.plot(zeit, q_leader, linewidth=2.0, alpha=0.85,
                    color=farben_leader[idx % len(farben_leader)],
                    linestyle='-', label=f'Leader {GELENK_NAMEN.get(gelenk, f"G{gelenk}")}')
            
            # Follower
            if f'q_follower_{gelenk}' in df.columns:
                q_follower = np.rad2deg(df[f'q_follower_{gelenk}'].values)
                ax.plot(zeit, q_follower, linewidth=2.0, alpha=0.85,
                        color=farben_follower[idx % len(farben_follower)],
                        linestyle='--', label=f'Follower {GELENK_NAMEN.get(gelenk, f"G{gelenk}")}')
        
        ax.set_xlabel('Zeit [s]', fontsize=14)
        ax.set_ylabel('Gelenkposition [°]', fontsize=14)
        ax.set_title('Teleoperationsverhalten: Gelenkraum',
                     fontsize=16, pad=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11, ncol=2, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    ausgabe_pfad = Path(args.output)
    plt.savefig(ausgabe_pfad)
    print(f"Plot gespeichert: {ausgabe_pfad}")
    
    png_pfad = ausgabe_pfad.with_suffix('.png')
    plt.savefig(png_pfad, dpi=150)
    print(f"Vorschau gespeichert: {png_pfad}")
    
    # Tracking-Statistiken ausgeben
    print("\n=== Tracking-Statistiken ===")
    for gelenk in args.joints:
        if f'q_follower_{gelenk}' in df.columns:
            q_leader = df[f'q_leader_{gelenk}'].values
            q_follower = df[f'q_follower_{gelenk}'].values
            fehler = np.abs(q_leader - q_follower)
            print(f"{GELENK_NAMEN.get(gelenk, f'Gelenk {gelenk}'):25s}: "
                  f"Mittel={np.rad2deg(np.mean(fehler)):.2f}°, "
                  f"Max={np.rad2deg(np.max(fehler)):.2f}°")
    
    if hat_kartesisch:
        print("\nKartesischer Raum:")
        for achse in ['x', 'y', 'z']:
            leader = df[f'tcp_{achse}_leader'].values
            follower = df[f'tcp_{achse}_follower'].values
            fehler = np.abs(leader - follower) * 1000  # in mm
            print(f"  {achse.upper()}: Mittel={np.mean(fehler):.2f} mm, Max={np.max(fehler):.2f} mm")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
