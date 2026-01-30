#!/usr/bin/env python3
"""
Reglervergleich – Positions- vs. Impedanzregelung (Abschnitt 4.3.3)

Vergleicht das Kontaktverhalten zwischen Positionsregelung und
Impedanzregelung. Zeigt, dass Impedanzregelung bei Kontakt nachgibt,
während Positionsregelung hohe Kraftspitzen erzeugt.

Verwendung:
    python plot_regler_vergleich.py <position_log.csv> <impedanz_log.csv> --output regler_vergleich.svg
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


def lade_und_filtere(logdatei, modus):
    """Daten laden und nach Steuerungsmodus filtern."""
    df = pd.read_csv(logdatei)
    
    if 'control_mode' in df.columns:
        df = df[df['control_mode'] == modus]
    
    if len(df) > 0:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Vergleich Positions- vs. Impedanzregelung")
    parser.add_argument("position_log", type=str, help="Logdatei Positionsregelung")
    parser.add_argument("impedanz_log", type=str, help="Logdatei Impedanzregelung")
    parser.add_argument("--output", type=str, default="regler_vergleich.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--tcp-axis", type=int, default=2,
                        help="TCP-Kraftachse (0-2 für Fx,Fy,Fz; Standard: 2=Fz)")
    parser.add_argument("--follower-joint", type=int, default=2,
                        help="Follower-Gelenk für Positionsplot (Standard: 2)")
    args = parser.parse_args()

    print("Lade Positionsregelungsdaten...")
    df_pos = lade_und_filtere(args.position_log, "position_teleop")
    
    # Fallback falls Modusfilterung nicht funktioniert
    if len(df_pos) == 0:
        df_pos = pd.read_csv(args.position_log)
        if len(df_pos) > 0:
            df_pos['time'] = df_pos['timestamp'] - df_pos['timestamp'].iloc[0]
    
    print("Lade Impedanzregelungsdaten...")
    df_imp = lade_und_filtere(args.impedanz_log, "impedance_teleop")
    
    if len(df_imp) == 0:
        df_imp = pd.read_csv(args.impedanz_log)
        if len(df_imp) > 0:
            df_imp['time'] = df_imp['timestamp'] - df_imp['timestamp'].iloc[0]
    
    if len(df_pos) == 0 or len(df_imp) == 0:
        print("Fehler: Konnte Daten nicht aus beiden Dateien laden")
        return 1
    
    print(f"Positionsregelung: {len(df_pos)} Messpunkte, {df_pos['time'].iloc[-1]:.1f}s")
    print(f"Impedanzregelung: {len(df_imp)} Messpunkte, {df_imp['time'].iloc[-1]:.1f}s")
    
    # Achsenbeschriftungen
    tcp_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    tcp_einheiten = ["N", "N", "N", "Nm", "Nm", "Nm"]
    
    # Subplots erstellen (2 Zeilen: Position, Kraft)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    # === Oberer Plot: Follower-Position ===
    gelenk = args.follower_joint
    
    # Positionsregelung (blau)
    if f'q_follower_{gelenk}' in df_pos.columns:
        pos_position = np.rad2deg(df_pos[f'q_follower_{gelenk}'].values)
        pos_zeit = df_pos['time'].values
        ax1.plot(pos_zeit, pos_position, linewidth=2.0, alpha=0.85, color='#3498db',
                 label='Positionsregelung')
    
    # Impedanzregelung (rot)
    if f'q_follower_{gelenk}' in df_imp.columns:
        imp_position = np.rad2deg(df_imp[f'q_follower_{gelenk}'].values)
        imp_zeit = df_imp['time'].values
        ax1.plot(imp_zeit, imp_position, linewidth=2.0, alpha=0.85, color='#e74c3c',
                 label='Impedanzregelung')
    
    ax1.set_ylabel(f'Follower Gelenk {gelenk} Position [°]', fontsize=13)
    ax1.set_title('Kontaktverhalten: Positionsregelung vs. Impedanzregelung',
                  fontsize=15, pad=10, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Annotation
    ax1.text(0.02, 0.98,
             'Impedanzregelung: Gibt bei Kontakt nach\nPositionsregelung: Kämpft gegen Hindernis',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
    
    # === Unterer Plot: TCP-Kraft ===
    max_kraft_pos = 0
    max_kraft_imp = 0
    
    # Positionsregelung (blau)
    if f'tcp_force_{args.tcp_axis}' in df_pos.columns:
        pos_kraft = df_pos[f'tcp_force_{args.tcp_axis}'].values
        pos_zeit = df_pos['time'].values
        ax2.plot(pos_zeit, pos_kraft, linewidth=2.0, alpha=0.85, color='#3498db',
                 label='Positionsregelung')
        
        # Maximum markieren
        max_kraft_pos = np.max(np.abs(pos_kraft))
        max_idx_pos = np.argmax(np.abs(pos_kraft))
        ax2.plot(pos_zeit[max_idx_pos], pos_kraft[max_idx_pos], 'D',
                 markersize=10, color='#3498db', markeredgecolor='black', markeredgewidth=1.5,
                 label=f'Max: {max_kraft_pos:.1f} {tcp_einheiten[args.tcp_axis]}')
    
    # Impedanzregelung (rot)
    if f'tcp_force_{args.tcp_axis}' in df_imp.columns:
        imp_kraft = df_imp[f'tcp_force_{args.tcp_axis}'].values
        imp_zeit = df_imp['time'].values
        ax2.plot(imp_zeit, imp_kraft, linewidth=2.0, alpha=0.85, color='#e74c3c',
                 label='Impedanzregelung')
        
        # Maximum markieren
        max_kraft_imp = np.max(np.abs(imp_kraft))
        max_idx_imp = np.argmax(np.abs(imp_kraft))
        ax2.plot(imp_zeit[max_idx_imp], imp_kraft[max_idx_imp], 'D',
                 markersize=10, color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5,
                 label=f'Max: {max_kraft_imp:.1f} {tcp_einheiten[args.tcp_axis]}')
        
        # Vergleich ausgeben
        print(f"\n=== Kraftvergleich ===")
        print(f"Positionsregelung Max-Kraft: {max_kraft_pos:.2f} {tcp_einheiten[args.tcp_axis]}")
        print(f"Impedanzregelung Max-Kraft: {max_kraft_imp:.2f} {tcp_einheiten[args.tcp_axis]}")
        if max_kraft_pos > 0:
            print(f"Reduktion: {(1 - max_kraft_imp/max_kraft_pos)*100:.1f}%")
    
    ax2.set_xlabel('Zeit [s]', fontsize=13)
    ax2.set_ylabel(f'TCP-Kraft {tcp_labels[args.tcp_axis]} [{tcp_einheiten[args.tcp_axis]}]', fontsize=13)
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Annotation
    if max_kraft_pos > 0 and max_kraft_imp > 0:
        reduktion = (1 - max_kraft_imp/max_kraft_pos)*100
        ax2.text(0.02, 0.98,
                 f'Impedanzregelung: Niedrigere Spitzenkräfte (sicher!)\n'
                 f'Kraftreduktion: {reduktion:.0f}%',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
    
    plt.tight_layout()
    
    # Speichern
    ausgabe_pfad = Path(args.output)
    plt.savefig(ausgabe_pfad)
    print(f"Plot gespeichert: {ausgabe_pfad}")
    
    png_pfad = ausgabe_pfad.with_suffix('.png')
    plt.savefig(png_pfad, dpi=150)
    print(f"Vorschau gespeichert: {png_pfad}")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
