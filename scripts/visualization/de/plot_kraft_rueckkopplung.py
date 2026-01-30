#!/usr/bin/env python3
"""
Kraftrückkopplung – Visualisierung (Abschnitt 4.3.2)

Zeigt den kausalen Zusammenhang zwischen externen Kräften am Follower-Roboter
und Rückkopplungsdrehmomenten am Leader-Roboter. Demonstriert, dass die
haptische Rückkopplung korrekt übertragen wird.

Verwendung:
    python plot_kraft_rueckkopplung.py <logfile.csv> --output kraft_rueckkopplung.svg

Für mehrere Achsen:
    python plot_kraft_rueckkopplung.py logs/force_x.csv logs/force_y.csv logs/force_z.csv --output kraft_xyz.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import signal

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

# Deutsche Beschriftungen
TCP_LABELS = {0: "Fx", 1: "Fy", 2: "Fz", 3: "Tx", 4: "Ty", 5: "Tz"}
TCP_EINHEITEN = {0: "N", 1: "N", 2: "N", 3: "Nm", 4: "Nm", 5: "Nm"}
GELENK_NAMEN = {
    0: "Gelenk 1", 1: "Gelenk 2", 2: "Gelenk 3",
    3: "Gelenk 4", 4: "Gelenk 5", 5: "Gelenk 6",
}


def berechne_korrelation(signal1, signal2):
    """Berechnet Kreuzkorrelation und Verzögerung."""
    # Normalisieren
    s1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-9)
    s2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-9)
    
    # Kreuzkorrelation
    korrelation = np.correlate(s1_norm, s2_norm, mode='full')
    lags = np.arange(-len(s1_norm) + 1, len(s2_norm))
    
    # Peak finden
    peak_idx = np.argmax(np.abs(korrelation))
    peak_lag = lags[peak_idx]
    peak_korr = korrelation[peak_idx] / len(signal1)
    
    return peak_korr, peak_lag


def main():
    parser = argparse.ArgumentParser(description="Kraftrückkopplung visualisieren")
    parser.add_argument("logfiles", type=str, nargs='+', help="Pfad(e) zur CSV-Logdatei(en)")
    parser.add_argument("--output", type=str, default="kraft_rueckkopplung.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Zeitfenster [Start, Ende] in Sekunden")
    parser.add_argument("--joint", type=int, default=2,
                        help="Gelenk für Drehmomentanalyse (Standard: 2 = Schulter)")
    parser.add_argument("--tcp-axes", type=int, nargs='+', default=[0, 1, 2],
                        help="TCP-Kraftachsen (0=Fx, 1=Fy, 2=Fz; Standard: 0 1 2)")
    parser.add_argument("--filter-mode", type=str, default=None,
                        help="Steuerungsmodus filtern (z.B. 'impedance_teleop')")
    args = parser.parse_args()

    # Anzahl der Subplots bestimmen
    if len(args.logfiles) > 1:
        # Mehrere Logdateien = mehrere Achsen
        n_plots = len(args.logfiles)
    else:
        # Eine Logdatei = mehrere TCP-Achsen
        n_plots = len(args.tcp_axes)
    
    # Layout bestimmen (3x1 für XYZ)
    if n_plots <= 3:
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
    else:
        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4*rows))
        axes = axes.flatten()
    
    korrelationen = []
    
    if len(args.logfiles) > 1:
        # Mehrere Logdateien - jede für eine Achse
        for idx, logdatei in enumerate(args.logfiles):
            print(f"\nLade {logdatei}...")
            df = pd.read_csv(logdatei)
            
            # Nach Modus filtern
            if args.filter_mode and 'control_mode' in df.columns:
                df = df[df['control_mode'] == args.filter_mode]
            
            if len(df) == 0:
                print(f"Warnung: Keine Daten in {logdatei}")
                continue
            
            df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
            
            if args.time_window:
                t_start, t_end = args.time_window
                df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
            
            zeit = df['time'].values
            tcp_achse = args.tcp_axes[idx] if idx < len(args.tcp_axes) else 2
            
            tcp_kraft = df[f'tcp_force_{tcp_achse}'].values
            tau_feedback = df[f'tau_feedback_{args.joint}'].values
            
            # Glätten für bessere Visualisierung
            window = min(51, len(tcp_kraft) // 10)
            if window % 2 == 0:
                window += 1
            if window > 3 and len(tcp_kraft) > window:
                tcp_kraft_smooth = signal.savgol_filter(tcp_kraft, window, 3)
            else:
                tcp_kraft_smooth = tcp_kraft
            
            # Korrelation berechnen
            if len(tcp_kraft) > 100:
                korr, lag = berechne_korrelation(tcp_kraft, tau_feedback)
                dt = np.mean(np.diff(zeit))
                lag_ms = lag * dt * 1000
                korrelationen.append((TCP_LABELS[tcp_achse], korr, lag_ms))
            else:
                korr, lag_ms = 0, 0
            
            # Plot mit Dual-Y-Achse
            ax1 = axes[idx]
            
            farbe1 = '#3498db'
            ax1.set_ylabel(f'TCP-Kraft {TCP_LABELS[tcp_achse]} [{TCP_EINHEITEN[tcp_achse]}]',
                           fontsize=12, color=farbe1)
            line1 = ax1.plot(zeit, tcp_kraft_smooth, linewidth=2.0, alpha=0.85, color=farbe1,
                             label=f'Follower {TCP_LABELS[tcp_achse]}')
            ax1.tick_params(axis='y', labelcolor=farbe1)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            farbe2 = '#e74c3c'
            ax2.set_ylabel(f'Leader Drehmoment {GELENK_NAMEN[args.joint]} [Nm]',
                           fontsize=12, color=farbe2)
            line2 = ax2.plot(zeit, tau_feedback, linewidth=2.0, alpha=0.85, color=farbe2,
                             label=f'Leader τ{args.joint}')
            ax2.tick_params(axis='y', labelcolor=farbe2)
            
            # Titel mit Korrelation
            ax1.set_title(f'Kraftrückkopplung {TCP_LABELS[tcp_achse]}-Achse '
                          f'(Korrelation: {korr:.3f}, Verzögerung: {lag_ms:.1f} ms)',
                          fontsize=13, pad=8)
            
            # Legende
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.95)
            
            if idx == n_plots - 1 or idx == len(args.logfiles) - 1:
                ax1.set_xlabel('Zeit [s]', fontsize=13)
    
    else:
        # Eine Logdatei - mehrere Achsen
        print(f"Lade {args.logfiles[0]}...")
        df = pd.read_csv(args.logfiles[0])
        
        if args.filter_mode and 'control_mode' in df.columns:
            df = df[df['control_mode'] == args.filter_mode]
        
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        if args.time_window:
            t_start, t_end = args.time_window
            df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
        
        zeit = df['time'].values
        
        for idx, tcp_achse in enumerate(args.tcp_axes):
            tcp_kraft = df[f'tcp_force_{tcp_achse}'].values
            tau_feedback = df[f'tau_feedback_{args.joint}'].values
            
            # Korrelation berechnen
            if len(tcp_kraft) > 100:
                korr, lag = berechne_korrelation(tcp_kraft, tau_feedback)
                dt = np.mean(np.diff(zeit))
                lag_ms = lag * dt * 1000
                korrelationen.append((TCP_LABELS[tcp_achse], korr, lag_ms))
            else:
                korr, lag_ms = 0, 0
            
            # Plot
            ax1 = axes[idx]
            
            farbe1 = '#3498db'
            line1 = ax1.plot(zeit, tcp_kraft, linewidth=2.0, alpha=0.85, color=farbe1,
                             label=f'Follower {TCP_LABELS[tcp_achse]}')
            ax1.set_ylabel(f'{TCP_LABELS[tcp_achse]} [{TCP_EINHEITEN[tcp_achse]}]',
                           fontsize=12, color=farbe1)
            ax1.tick_params(axis='y', labelcolor=farbe1)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            farbe2 = '#e74c3c'
            line2 = ax2.plot(zeit, tau_feedback, linewidth=2.0, alpha=0.85, color=farbe2,
                             label=f'Leader τ')
            ax2.set_ylabel(f'τ{args.joint} [Nm]', fontsize=12, color=farbe2)
            ax2.tick_params(axis='y', labelcolor=farbe2)
            
            ax1.set_title(f'{TCP_LABELS[tcp_achse]}-Achse (ρ={korr:.3f}, Δt={lag_ms:.1f}ms)',
                          fontsize=13, pad=8)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=10)
            
            if idx == len(args.tcp_axes) - 1:
                ax1.set_xlabel('Zeit [s]', fontsize=13)
    
    # Haupttitel
    fig.suptitle('Kraftrückkopplung: Haptische Rückmeldung Validierung',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Speichern
    ausgabe_pfad = Path(args.output)
    plt.savefig(ausgabe_pfad, bbox_inches='tight')
    print(f"\nPlot gespeichert: {ausgabe_pfad}")
    
    png_pfad = ausgabe_pfad.with_suffix('.png')
    plt.savefig(png_pfad, dpi=150, bbox_inches='tight')
    print(f"Vorschau gespeichert: {png_pfad}")
    
    # Korrelationsstatistiken ausgeben
    if korrelationen:
        print("\n=== Korrelationsanalyse ===")
        for achse, korr, lag in korrelationen:
            print(f"  {achse}: Korrelation = {korr:.3f}, Verzögerung = {lag:.1f} ms")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
