#!/usr/bin/env python3
"""
Analyse-Tool für Haltemoment-Messungen
=======================================

Dieses Skript analysiert und visualisiert die Ergebnisse der Haltemoment-Messungen.

Verwendung:
    python scripts/analyze_holding_torques.py recordings/holding_torques_20260102_150000.json
    python scripts/analyze_holding_torques.py recordings/holding_torques_*.json --compare
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[INFO] Matplotlib nicht verfügbar - keine Plots")


def load_measurement(filepath: str) -> Dict:
    """Lädt eine Messung aus JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_measurement_info(data: Dict):
    """Gibt grundlegende Informationen über die Messung aus"""
    meta = data['metadata']
    has_rnea = 'rnea_torque_nm' in data['measurements'][0] if data['measurements'] else False
    
    print("\n" + "="*70)
    print("MESSUNG-INFORMATIONEN")
    print("="*70)
    print(f"Zeitstempel:      {meta['timestamp']}")
    print(f"Anzahl Gelenke:   {meta['num_joints']}")
    print(f"Servo-Typen:      {', '.join(meta['servo_types'])}")
    print(f"Anzahl Messungen: {meta['num_measurements']}")
    print(f"RNEA-Vergleich:   {'✓ Ja' if has_rnea else '✗ Nein'}")
    print("="*70)


def analyze_statistics(data: Dict):
    """Analysiert statistische Eigenschaften der Messungen"""
    measurements = data['measurements']
    
    if not measurements:
        print("\n[WARN] Keine Messungen vorhanden")
        return
    
    # Extract torque data
    torques_nm = np.array([m['torque_nm'] for m in measurements])
    currents_ma = np.array([m['current_ma'] for m in measurements])
    current_stds = np.array([m['current_std_ma'] for m in measurements])
    
    has_rnea = 'rnea_torque_nm' in measurements[0]
    rnea_torques_nm = np.array([m['rnea_torque_nm'] for m in measurements]) if has_rnea else None
    
    num_joints = torques_nm.shape[1]
    joint_names = [f"Joint {i+1}" for i in range(num_joints)]
    
    print("\n" + "="*70)
    print("STATISTIK - HALTEMOMENTE [mNm]")
    print("="*70)
    
    # Per-joint statistics
    for i in range(num_joints):
        joint_torques = torques_nm[:, i] * 1000  # Convert to mNm
        joint_currents = currents_ma[:, i]
        
        mean_abs = np.mean(np.abs(joint_torques))
        max_abs = np.max(np.abs(joint_torques))
        std = np.std(joint_torques)
        
        print(f"\n{joint_names[i]}:")
        print(f"  Mean (abs):  {mean_abs:7.2f} mNm")
        print(f"  Max (abs):   {max_abs:7.2f} mNm")
        print(f"  Std Dev:     {std:7.2f} mNm")
        print(f"  Mean Current: {np.mean(joint_currents):6.1f} mA")
        
        if has_rnea:
            joint_rnea = rnea_torques_nm[:, i] * 1000
            rnea_mean_abs = np.mean(np.abs(joint_rnea))
            errors = np.abs(joint_torques - joint_rnea)
            rmse = np.sqrt(np.mean(errors**2))
            
            print(f"  RNEA (mean): {rnea_mean_abs:7.2f} mNm")
            print(f"  RMSE Error:  {rmse:7.2f} mNm")
            
            if mean_abs > 50:  # Only show relative error for significant torques
                rel_error = (rmse / mean_abs) * 100
                print(f"  Rel. Error:  {rel_error:7.1f} %")
    
    print("\n" + "="*70)


def print_detailed_table(data: Dict):
    """Druckt detaillierte Tabelle aller Messungen"""
    measurements = data['measurements']
    
    if not measurements:
        return
    
    num_joints = len(measurements[0]['torque_nm'])
    joint_names = [f"J{i+1}" for i in range(num_joints)]
    has_rnea = 'rnea_torque_nm' in measurements[0]
    
    print("\n" + "="*70)
    print("DETAILLIERTE MESSUNGEN")
    print("="*70)
    
    # Header
    print(f"\n{'Pose':<25} " + " ".join([f"{j:>9}" for j in joint_names]))
    print("-" * 70)
    
    # Torques
    print("\nHaltemomente (gemessen) [mNm]:")
    for m in measurements:
        torques_mnm = np.array(m['torque_mnm'])
        print(f"{m['name']:<25} " + " ".join([f"{t:>9.2f}" for t in torques_mnm]))
    
    if has_rnea:
        print("\nHaltemomente (RNEA/URDF) [mNm]:")
        for m in measurements:
            rnea_mnm = np.array(m['rnea_torque_mnm'])
            print(f"{m['name']:<25} " + " ".join([f"{t:>9.2f}" for t in rnea_mnm]))
        
        print("\nAbsoluter Fehler (|Δ|) [mNm]:")
        for m in measurements:
            error_mnm = np.array(m['rnea_error_mnm'])
            print(f"{m['name']:<25} " + " ".join([f"{e:>9.2f}" for e in error_mnm]))
    
    # Currents
    print("\nHalteströme [mA]:")
    for m in measurements:
        currents = np.array(m['current_ma'])
        print(f"{m['name']:<25} " + " ".join([f"{c:>9.1f}" for c in currents]))
    
    # Standard deviations
    print("\nStandardabweichungen [mA]:")
    for m in measurements:
        stds = np.array(m['current_std_ma'])
        print(f"{m['name']:<25} " + " ".join([f"{s:>9.1f}" for s in stds]))
    
    print("=" * 70)


def plot_torque_distribution(data: Dict, output_file: str = None):
    """Erstellt Balkendiagramm der Momente pro Pose"""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib nicht verfügbar")
        return
    
    measurements = data['measurements']
    if not measurements:
        return
    
    pose_names = [m['name'] for m in measurements]
    torques_nm = np.array([m['torque_nm'] for m in measurements])
    
    num_joints = torques_nm.shape[1]
    num_poses = len(pose_names)
    
    # Create figure with subplots for each joint
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 3*num_joints))
    if num_joints == 1:
        axes = [axes]
    
    fig.suptitle('Haltemomente pro Pose und Gelenk', fontsize=16, fontweight='bold')
    
    x = np.arange(num_poses)
    width = 0.7
    
    for i in range(num_joints):
        ax = axes[i]
        torques_mnm = torques_nm[:, i] * 1000  # Convert to mNm
        
        colors = ['green' if abs(t) < 50 else 'orange' if abs(t) < 100 else 'red' 
                  for t in torques_mnm]
        
        bars = ax.bar(x, torques_mnm, width, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Moment [mNm]', fontweight='bold')
        ax.set_title(f'Gelenk {i+1}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pose_names, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, torques_mnm)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[✓] Plot gespeichert: {output_file}")
    else:
        plt.show()


def plot_joint_comparison(data: Dict, output_file: str = None):
    """Erstellt Box-Plot Vergleich der Gelenke"""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib nicht verfügbar")
        return
    
    measurements = data['measurements']
    if not measurements:
        return
    
    torques_nm = np.array([m['torque_nm'] for m in measurements])
    num_joints = torques_nm.shape[1]
    
    # Convert to mNm and get absolute values
    torques_mnm_abs = np.abs(torques_nm * 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(torques_mnm_abs, labels=[f'J{i+1}' for i in range(num_joints)],
                    patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Gelenk', fontweight='bold', fontsize=12)
    ax.set_ylabel('Absolutes Haltemoment [mNm]', fontweight='bold', fontsize=12)
    ax.set_title('Verteilung der Haltemomente über alle Posen', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean values as text
    means = np.mean(torques_mnm_abs, axis=0)
    for i, mean in enumerate(means):
        ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean:.1f}',
               ha='center', va='top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[✓] Plot gespeichert: {output_file}")
    else:
        plt.show()


def plot_rnea_comparison(data: Dict, output_file: str = None):
    """Erstellt Vergleichs-Plot: Gemessen vs. RNEA"""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib nicht verfügbar")
        return
    
    measurements = data['measurements']
    if not measurements:
        return
    
    has_rnea = 'rnea_torque_nm' in measurements[0]
    if not has_rnea:
        print("[SKIP] Keine RNEA-Daten vorhanden")
        return
    
    torques_nm = np.array([m['torque_nm'] for m in measurements])
    rnea_torques_nm = np.array([m['rnea_torque_nm'] for m in measurements])
    num_joints = torques_nm.shape[1]
    
    # Create subplot for each joint
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle('Gemessen vs. RNEA (URDF-basiert)', fontsize=16, fontweight='bold')
    
    for i in range(num_joints):
        ax = axes[i]
        
        measured = torques_nm[:, i] * 1000  # mNm
        rnea = rnea_torques_nm[:, i] * 1000  # mNm
        
        # Scatter plot
        ax.scatter(rnea, measured, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect agreement line
        all_vals = np.concatenate([measured, rnea])
        lim_min, lim_max = all_vals.min(), all_vals.max()
        margin = (lim_max - lim_min) * 0.1
        ax.plot([lim_min-margin, lim_max+margin], [lim_min-margin, lim_max+margin],
               'r--', linewidth=2, label='Perfekt', alpha=0.7)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((measured - rnea)**2))
        mean_abs = np.mean(np.abs(measured))
        rel_error = (rmse / mean_abs * 100) if mean_abs > 10 else 0
        
        ax.set_xlabel('RNEA [mNm]', fontweight='bold')
        ax.set_ylabel('Gemessen [mNm]', fontweight='bold')
        ax.set_title(f'Gelenk {i+1}\nRMSE={rmse:.1f} mNm ({rel_error:.0f}%)',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[✓] RNEA-Vergleichs-Plot gespeichert: {output_file}")
    else:
        plt.show()


def plot_rnea_error_distribution(data: Dict, output_file: str = None):
    """Erstellt Error-Verteilung für RNEA-Vergleich"""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib nicht verfügbar")
        return
    
    measurements = data['measurements']
    if not measurements:
        return
    
    has_rnea = 'rnea_torque_nm' in measurements[0]
    if not has_rnea:
        print("[SKIP] Keine RNEA-Daten vorhanden")
        return
    
    pose_names = [m['name'] for m in measurements]
    errors_nm = np.array([m['rnea_error_nm'] for m in measurements])
    errors_mnm = errors_nm * 1000
    num_joints = errors_mnm.shape[1]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(pose_names))
    width = 0.12
    
    colors = plt.cm.Set2(np.linspace(0, 1, num_joints))
    
    for i in range(num_joints):
        offset = (i - num_joints/2 + 0.5) * width
        ax.bar(x + offset, errors_mnm[:, i], width,
              label=f'J{i+1}', color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Pose', fontweight='bold', fontsize=12)
    ax.set_ylabel('RNEA Fehler [mNm]', fontweight='bold', fontsize=12)
    ax.set_title('RNEA-Fehler |Gemessen - URDF| pro Pose', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pose_names, rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=num_joints)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[✓] RNEA-Error-Plot gespeichert: {output_file}")
    else:
        plt.show()


def compare_measurements(filepaths: List[str]):
    """Vergleicht mehrere Messungen"""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib nicht verfügbar - nur Text-Vergleich")
    
    print("\n" + "="*70)
    print("VERGLEICH MEHRERER MESSUNGEN")
    print("="*70)
    
    datasets = []
    for fp in filepaths:
        try:
            data = load_measurement(fp)
            datasets.append({'path': fp, 'data': data})
            print(f"\n✓ Geladen: {Path(fp).name}")
            print(f"  Zeitstempel: {data['metadata']['timestamp']}")
            print(f"  Messungen:   {data['metadata']['num_measurements']}")
        except Exception as e:
            print(f"\n✗ Fehler bei {fp}: {e}")
    
    if len(datasets) < 2:
        print("\n[WARN] Mindestens 2 Messungen für Vergleich erforderlich")
        return
    
    # Compare statistics
    print("\n" + "-"*70)
    print("VERGLEICH DER MITTLEREN ABSOLUTEN MOMENTE [mNm]")
    print("-"*70)
    
    num_joints = len(datasets[0]['data']['measurements'][0]['torque_nm'])
    
    # Header
    print(f"\n{'Datei':<30} " + " ".join([f"J{i+1:>8}" for i in range(num_joints)]))
    print("-" * 70)
    
    for ds in datasets:
        measurements = ds['data']['measurements']
        torques = np.array([m['torque_nm'] for m in measurements])
        means = np.mean(np.abs(torques), axis=0) * 1000  # mNm
        
        filename = Path(ds['path']).name[:28]
        print(f"{filename:<30} " + " ".join([f"{m:>9.2f}" for m in means]))
    
    print("="*70)


def export_csv(data: Dict, output_file: str):
    """Exportiert Messungen als CSV"""
    measurements = data['measurements']
    
    if not measurements:
        print("[WARN] Keine Messungen zum Exportieren")
        return
    
    num_joints = len(measurements[0]['torque_nm'])
    has_rnea = 'rnea_torque_nm' in measurements[0]
    
    # Header
    header_parts = ["pose"]
    for i in range(num_joints):
        header_parts.append(f"torque_j{i+1}_nm")
        header_parts.append(f"current_j{i+1}_ma")
        if has_rnea:
            header_parts.append(f"rnea_j{i+1}_nm")
            header_parts.append(f"error_j{i+1}_nm")
    header = ",".join(header_parts)
    
    # Data rows
    rows = []
    for m in measurements:
        row = [m['name']]
        for i in range(num_joints):
            row.append(f"{m['torque_nm'][i]:.6f}")
            row.append(f"{m['current_ma'][i]:.2f}")
            if has_rnea:
                row.append(f"{m['rnea_torque_nm'][i]:.6f}")
                row.append(f"{m['rnea_error_nm'][i]:.6f}")
        rows.append(",".join(row))
    
    # Write file
    with open(output_file, 'w') as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    
    print(f"\n[✓] CSV exportiert: {output_file}")
    if has_rnea:
        print(f"    (inkl. RNEA-Vergleichsdaten)")


def main():
    parser = argparse.ArgumentParser(
        description='Analysiert Haltemoment-Messungen'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON-Datei(en) mit Messungen'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Vergleiche mehrere Messungen'
    )
    parser.add_argument(
        '--plot-distribution',
        action='store_true',
        help='Erstelle Balkendiagramm pro Pose'
    )
    parser.add_argument(
        '--plot-comparison',
        action='store_true',
        help='Erstelle Box-Plot Vergleich'
    )
    parser.add_argument(
        '--plot-rnea',
        action='store_true',
        help='Erstelle RNEA-Vergleichs-Plot (Gemessen vs. URDF)'
    )
    parser.add_argument(
        '--plot-rnea-error',
        action='store_true',
        help='Erstelle RNEA-Fehler-Verteilung'
    )
    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Erstelle alle verfügbaren Plots'
    )
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Exportiere als CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Ausgabe-Datei für Plots'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards
    filepaths = []
    for pattern in args.files:
        filepaths.extend(Path().glob(pattern))
    filepaths = [str(p) for p in filepaths]
    
    if not filepaths:
        print("[ERROR] Keine Dateien gefunden")
        return
    
    if args.compare:
        compare_measurements(filepaths)
    else:
        # Single file analysis
        data = load_measurement(filepaths[0])
        
        print_measurement_info(data)
        analyze_statistics(data)
        print_detailed_table(data)
        
        if args.plot_distribution or args.plot_all:
            output = args.output or filepaths[0].replace('.json', '_distribution.png')
            plot_torque_distribution(data, output)
        
        if args.plot_comparison or args.plot_all:
            output = args.output or filepaths[0].replace('.json', '_comparison.png')
            plot_joint_comparison(data, output)
        
        if args.plot_rnea or args.plot_all:
            output = args.output or filepaths[0].replace('.json', '_rnea_comparison.png')
            plot_rnea_comparison(data, output)
        
        if args.plot_rnea_error or args.plot_all:
            output = args.output or filepaths[0].replace('.json', '_rnea_error.png')
            plot_rnea_error_distribution(data, output)
        
        if args.export_csv:
            export_csv(data, args.export_csv)


if __name__ == "__main__":
    main()
