#!/usr/bin/env python3
"""
Reibungskompensation - Visualisierung (Abschnitt 4.2.1)

Vergleicht die Reibungscharakteristik mit und ohne Kompensation.
Zeigt die Coulomb- + viskose Reibung im Gelenkraum.

Verwendung:
    python plot_reibung_kompensation.py <logfile.csv> --joint 2 --output reibung_kompensation.svg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.optimize import curve_fit

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


def coulomb_viskos_modell(geschwindigkeit, tau_coulomb, tau_viskos, totzone=0.01):
    """Coulomb + viskoses Reibungsmodell."""
    tau = np.zeros_like(geschwindigkeit)
    for i, v in enumerate(geschwindigkeit):
        if abs(v) < totzone:
            tau[i] = 0.0
        else:
            tau[i] = tau_coulomb * np.sign(v) + tau_viskos * v
    return tau


def main():
    parser = argparse.ArgumentParser(description="Reibungskompensation visualisieren")
    parser.add_argument("logfile", type=str, help="Pfad zur CSV-Logdatei")
    parser.add_argument("--joint", type=int, default=2, 
                        help="Gelenkindex (0-5), Standard: 2 (Schulter)")
    parser.add_argument("--output", type=str, default="reibung_kompensation.svg",
                        help="Ausgabedateiname")
    parser.add_argument("--filter-mode", type=str, default="gravity_comp",
                        help="Nur Daten aus diesem Steuerungsmodus verwenden")
    args = parser.parse_args()

    # Daten laden
    print(f"Lade Daten aus {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Nach Steuerungsmodus filtern
    if args.filter_mode and 'control_mode' in df.columns:
        df = df[df['control_mode'] == args.filter_mode]
    
    if len(df) == 0:
        print(f"Fehler: Keine Daten für Modus '{args.filter_mode}' gefunden")
        return 1
    
    gelenk_idx = args.joint
    print(f"Analysiere {GELENK_NAMEN.get(gelenk_idx, f'Gelenk {gelenk_idx}')} ({len(df)} Messpunkte)")
    
    # Geschwindigkeit und Reibungsdrehmoment extrahieren
    geschwindigkeit = df[f'q_dot_leader_{gelenk_idx}'].values
    
    # Reibungsdrehmoment aus Log verwenden
    if f'tau_friction_{gelenk_idx}' in df.columns:
        tau_reibung = df[f'tau_friction_{gelenk_idx}'].values
        print("Verwende geloggte tau_friction Spalte")
    else:
        # Fallback: aus Gesamt - Gravitation berechnen
        tau_gravitation = df[f'tau_gravity_{gelenk_idx}'].values
        tau_gesamt = df[f'tau_total_{gelenk_idx}'].values
        tau_reibung = tau_gesamt - tau_gravitation
        print("Warnung: Berechne Reibung als Gesamt - Gravitation")
    
    # Ausreißer filtern (über 99. Perzentil)
    gueltig_maske = np.abs(tau_reibung) < np.percentile(np.abs(tau_reibung), 99)
    geschwindigkeit = geschwindigkeit[gueltig_maske]
    tau_reibung = tau_reibung[gueltig_maske]
    
    # Nahe-Null-Geschwindigkeit filtern (Totzonenrauschen)
    bewegungs_maske = np.abs(geschwindigkeit) > 0.005
    geschwindigkeit = geschwindigkeit[bewegungs_maske]
    tau_reibung = tau_reibung[bewegungs_maske]
    
    print(f"Nach Filterung: {len(geschwindigkeit)} Messpunkte")
    print(f"Geschwindigkeitsbereich: [{geschwindigkeit.min():.4f}, {geschwindigkeit.max():.4f}] rad/s")
    print(f"Reibungsbereich: [{tau_reibung.min():.4f}, {tau_reibung.max():.4f}] Nm")
    
    # Reibungsmodell anpassen
    try:
        if np.std(tau_reibung) < 0.001:
            print(f"Warnung: Sehr geringe Reibungsvariation (Std={np.std(tau_reibung):.6f})")
            raise ValueError("Unzureichende Datenvariation für Anpassung")
        
        tau_bereich = np.abs(tau_reibung).max()
        vel_bereich = np.abs(geschwindigkeit).max()
        p0_coulomb = tau_bereich * 0.3
        p0_viskos = (tau_bereich * 0.7) / vel_bereich if vel_bereich > 0 else 0.1
        
        popt, _ = curve_fit(
            coulomb_viskos_modell,
            geschwindigkeit,
            tau_reibung,
            p0=[p0_coulomb, p0_viskos],
            bounds=([0, 0], [5.0, 2.0]),
            maxfev=10000
        )
        tau_coulomb_fit, tau_viskos_fit = popt
        
        print(f"\n✓ Angepasste Parameter:")
        print(f"  Coulomb-Reibung: {tau_coulomb_fit:.4f} Nm")
        print(f"  Viskose Reibung: {tau_viskos_fit:.4f} Nm·s/rad")
        
        # Glatte Modellkurve generieren
        vel_modell = np.linspace(geschwindigkeit.min(), geschwindigkeit.max(), 500)
        tau_modell = coulomb_viskos_modell(vel_modell, tau_coulomb_fit, tau_viskos_fit)
    except Exception as e:
        print(f"Warnung: Modellanpassung fehlgeschlagen: {e}")
        tau_coulomb_fit, tau_viskos_fit = None, None
        vel_modell, tau_modell = None, None
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Streudiagramm der Messungen
    ax.scatter(geschwindigkeit, tau_reibung,
               s=3, alpha=0.4, c='#3498db', label='Messdaten', rasterized=True)
    
    # Angepasstes Modell plotten
    if tau_coulomb_fit is not None:
        ax.plot(vel_modell, tau_modell, 'r-', linewidth=2.5,
                label=f'Modell ($\\tau_C$={tau_coulomb_fit:.3f} Nm, $\\tau_v$={tau_viskos_fit:.3f} Nm·s/rad)')
    
    # Beschriftung
    ax.set_xlabel('Gelenkgeschwindigkeit [rad/s]', fontsize=14)
    ax.set_ylabel('Reibungsdrehmoment [Nm]', fontsize=14)
    ax.set_title(f'Reibungscharakteristik – {GELENK_NAMEN.get(gelenk_idx, f"Gelenk {gelenk_idx}")}',
                 fontsize=16, pad=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Textbox mit Erklärung
    erklaerung = (
        'Reibungsmodell: $\\tau = \\tau_C \\cdot \\mathrm{sign}(\\dot{q}) + \\tau_v \\cdot \\dot{q}$\n'
        'Coulomb: richtungsabhängig, konstant\n'
        'Viskos: proportional zur Geschwindigkeit'
    )
    ax.text(0.98, 0.02, erklaerung, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
