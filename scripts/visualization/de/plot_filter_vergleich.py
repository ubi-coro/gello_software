#!/usr/bin/env python3
"""
Filtervergleich – Visualisierung (Abschnitt 4.2.2)

Vergleicht Rohsignal, EMA-gefiltert und 1€-gefiltert zur Demonstration
des Kompromisses zwischen Rauschunterdrückung und Latenz.

Verwendung:
    python plot_filter_vergleich.py <logfile.csv> --joint 2 --output filter_vergleich.svg
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
    parser = argparse.ArgumentParser(description="Filtervergleich für externe Drehmomente")
    parser.add_argument("logfile", type=str, help="Pfad zur CSV-Logdatei")
    parser.add_argument("--joint", type=int, default=2,
                        help="Gelenkindex (0-5), Standard: 2 (Ellbogen)")
    parser.add_argument("--output", type=str, default="filter_vergleich.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--time-window", type=float, nargs=2, default=None,
                        help="Zeitfenster [Start, Ende] in Sekunden")
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
    
    gelenk_idx = args.joint
    print(f"Analysiere {GELENK_NAMEN.get(gelenk_idx, f'Gelenk {gelenk_idx}')} ({len(df)} Messpunkte)")
    
    # Daten extrahieren
    zeit = df['time'].values
    
    # Gefilterte Signale aus Log laden (falls vorhanden)
    if f'tau_external_raw_{gelenk_idx}' in df.columns:
        tau_roh = df[f'tau_external_raw_{gelenk_idx}'].values
        tau_ema = df[f'tau_external_ema_{gelenk_idx}'].values
        tau_oneeuro = df[f'tau_external_oneeuro_{gelenk_idx}'].values
        print("Verwende vorberechnete Filterdaten aus Log")
    else:
        # Fallback: nur externes Drehmoment verwenden
        tau_roh = df[f'tau_external_{gelenk_idx}'].values
        
        # EMA nachträglich anwenden
        ema_alpha = 0.1
        tau_ema = np.zeros_like(tau_roh)
        tau_ema[0] = tau_roh[0]
        for i in range(1, len(tau_roh)):
            tau_ema[i] = ema_alpha * tau_roh[i] + (1 - ema_alpha) * tau_ema[i-1]
        
        # 1€-Filter nachträglich anwenden
        min_cutoff = 1.0
        beta = 0.007
        d_cutoff = 1.0
        tau_oneeuro = np.zeros_like(tau_roh)
        x_prev = tau_roh[0]
        dx_prev = 0.0
        t_prev = zeit[0]
        
        for i in range(len(tau_roh)):
            x = tau_roh[i]
            dt = zeit[i] - t_prev if i > 0 else 0.002
            if dt <= 0:
                dt = 0.002
            
            dx = (x - x_prev) / dt
            tau_d = 1.0 / (2 * np.pi * d_cutoff)
            alpha_d = 1.0 / (1.0 + tau_d / dt)
            dx_smooth = alpha_d * dx + (1 - alpha_d) * dx_prev
            
            cutoff = min_cutoff + beta * abs(dx_smooth)
            tau = 1.0 / (2 * np.pi * cutoff)
            alpha = 1.0 / (1.0 + tau / dt)
            
            x_filtered = alpha * x + (1 - alpha) * x_prev
            tau_oneeuro[i] = x_filtered
            
            x_prev = x_filtered
            dx_prev = dx_smooth
            t_prev = zeit[i]
        
        print("Filter nachträglich angewendet")
    
    # Plot mit zwei Subplots erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # === Oberer Plot: Vollständiger Vergleich ===
    ax1.plot(zeit, tau_roh, linewidth=0.6, alpha=0.5, color='#95a5a6',
             label='Rohsignal', rasterized=True)
    ax1.plot(zeit, tau_ema, linewidth=1.8, alpha=0.9, color='#3498db',
             label='EMA (α=0,1)')
    ax1.plot(zeit, tau_oneeuro, linewidth=1.8, alpha=0.9, color='#e74c3c',
             label='1€-Filter')
    
    ax1.set_ylabel('Externes Drehmoment [Nm]', fontsize=13)
    ax1.set_title(f'Filtervergleich – {GELENK_NAMEN.get(gelenk_idx, f"Gelenk {gelenk_idx}")}',
                  fontsize=15, pad=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # === Unterer Plot: Detailansicht bei Bewegungsbeginn ===
    # Region mit schneller Änderung finden
    tau_abs = np.abs(tau_roh)
    schwelle = np.percentile(tau_abs, 75)
    hoch_idx = np.where(tau_abs > schwelle)[0]
    
    if len(hoch_idx) > 100:
        # Übergang finden
        uebergang_idx = hoch_idx[0]
        for idx in hoch_idx:
            if idx > 50:
                uebergang_idx = idx
                break
        
        # Zoom-Fenster: ±1 Sekunde um Übergang
        zoom_start = max(0, uebergang_idx - 500)
        zoom_end = min(len(zeit), uebergang_idx + 500)
        
        ax2.plot(zeit[zoom_start:zoom_end], tau_roh[zoom_start:zoom_end],
                 linewidth=0.8, alpha=0.5, color='#95a5a6', label='Rohsignal')
        ax2.plot(zeit[zoom_start:zoom_end], tau_ema[zoom_start:zoom_end],
                 linewidth=2.0, alpha=0.9, color='#3498db', label='EMA')
        ax2.plot(zeit[zoom_start:zoom_end], tau_oneeuro[zoom_start:zoom_end],
                 linewidth=2.0, alpha=0.9, color='#e74c3c', label='1€-Filter')
        
        # Übergangspunkt markieren
        ax2.axvline(zeit[uebergang_idx], color='k', linewidth=1.5, linestyle='--',
                    alpha=0.6, label='Kontaktbeginn')
    else:
        # Keine klare Änderung gefunden - mittleren Abschnitt zoomen
        mitte = len(zeit) // 2
        zoom_start = max(0, mitte - 500)
        zoom_end = min(len(zeit), mitte + 500)
        
        ax2.plot(zeit[zoom_start:zoom_end], tau_roh[zoom_start:zoom_end],
                 linewidth=0.8, alpha=0.5, color='#95a5a6', label='Rohsignal')
        ax2.plot(zeit[zoom_start:zoom_end], tau_ema[zoom_start:zoom_end],
                 linewidth=2.0, alpha=0.9, color='#3498db', label='EMA')
        ax2.plot(zeit[zoom_start:zoom_end], tau_oneeuro[zoom_start:zoom_end],
                 linewidth=2.0, alpha=0.9, color='#e74c3c', label='1€-Filter')
    
    ax2.set_xlabel('Zeit [s]', fontsize=13)
    ax2.set_ylabel('Externes Drehmoment [Nm]', fontsize=13)
    ax2.set_title('Detailansicht: Filterverhalten bei Signalsprung', fontsize=13, pad=8)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Textbox mit Erklärung
    erklaerung = (
        '1€-Filter: Adaptive Grenzfrequenz\n'
        '  → Geringe Latenz bei schnellen Bewegungen\n'
        '  → Starke Glättung bei langsamen Signalen\n\n'
        'EMA: Feste Glättung\n'
        '  → Konstante Latenz unabhängig von Bewegung'
    )
    ax1.text(0.02, 0.02, erklaerung, transform=ax1.transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    
    plt.tight_layout()
    
    # Speichern
    ausgabe_pfad = Path(args.output)
    plt.savefig(ausgabe_pfad)
    print(f"Plot gespeichert: {ausgabe_pfad}")
    
    png_pfad = ausgabe_pfad.with_suffix('.png')
    plt.savefig(png_pfad, dpi=150)
    print(f"Vorschau gespeichert: {png_pfad}")
    
    # Statistiken ausgeben
    print("\n=== Filterstatistiken ===")
    print(f"Rohsignal - Std: {np.std(tau_roh):.4f} Nm")
    print(f"EMA       - Std: {np.std(tau_ema):.4f} Nm (Reduktion: {(1-np.std(tau_ema)/np.std(tau_roh))*100:.1f}%)")
    print(f"1€-Filter - Std: {np.std(tau_oneeuro):.4f} Nm (Reduktion: {(1-np.std(tau_oneeuro)/np.std(tau_roh))*100:.1f}%)")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
