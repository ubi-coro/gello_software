#!/usr/bin/env python3
"""
Log File Inspector

Quick utility to inspect GELLO log files and show basic statistics.

Usage:
    python inspect_log.py <logfile.csv>
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect GELLO log file")
    parser.add_argument("logfile", type=str, help="Path to CSV log file")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.logfile}...")
    df = pd.read_csv(args.logfile)
    
    # Convert timestamp to relative time
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Basic info
    print("\n" + "="*70)
    print("LOG FILE SUMMARY")
    print("="*70)
    print(f"File: {args.logfile}")
    print(f"Size: {Path(args.logfile).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Samples: {len(df)}")
    print(f"Duration: {df['time'].iloc[-1]:.2f} seconds")
    print(f"Sample rate: {len(df) / df['time'].iloc[-1]:.1f} Hz")
    
    # Columns
    print(f"\nColumns: {len(df.columns)}")
    
    # Control modes
    if 'control_mode' in df.columns:
        modes = df['control_mode'].value_counts()
        print(f"\nControl modes:")
        for mode, count in modes.items():
            duration = df[df['control_mode'] == mode]['time'].iloc[-1] - df[df['control_mode'] == mode]['time'].iloc[0]
            print(f"  {mode:20s}: {count:6d} samples ({duration:.1f}s)")
    
    # Joint statistics
    print("\n" + "="*70)
    print("JOINT POSITION STATISTICS (Leader)")
    print("="*70)
    print(f"{'Joint':<10s} {'Min [°]':>10s} {'Max [°]':>10s} {'Mean [°]':>10s} {'StdDev [°]':>12s}")
    print("-"*70)
    
    for i in range(6):
        col = f'q_leader_{i}'
        if col in df.columns:
            data_rad = df[col].values
            data_deg = np.rad2deg(data_rad)
            print(f"Joint {i:<4d} {np.min(data_deg):>10.2f} {np.max(data_deg):>10.2f} "
                  f"{np.mean(data_deg):>10.2f} {np.std(data_deg):>12.3f}")
    
    # Torque statistics
    print("\n" + "="*70)
    print("TORQUE STATISTICS")
    print("="*70)
    print(f"{'Component':<20s} {'Joint 0':>10s} {'Joint 1':>10s} {'Joint 2':>10s} "
          f"{'Joint 3':>10s} {'Joint 4':>10s} {'Joint 5':>10s}")
    print("-"*70)
    
    for component in ['gravity', 'friction', 'feedback', 'total']:
        row = [f"{component.capitalize():<20s}"]
        for i in range(6):
            col = f'tau_{component}_{i}'
            if col in df.columns:
                rms = np.sqrt(np.mean(df[col].values**2))
                row.append(f"{rms:>10.3f}")
            else:
                row.append(f"{'N/A':>10s}")
        print("".join(row))
    
    # TCP Force statistics
    if 'tcp_force_0' in df.columns:
        print("\n" + "="*70)
        print("TCP FORCE/TORQUE STATISTICS")
        print("="*70)
        labels = ["Fx [N]", "Fy [N]", "Fz [N]", "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"]
        print(f"{'Axis':<10s} {'Mean':>10s} {'Max':>10s} {'StdDev':>10s}")
        print("-"*70)
        
        for i in range(6):
            col = f'tcp_force_{i}'
            if col in df.columns:
                data = df[col].values
                print(f"{labels[i]:<10s} {np.mean(data):>10.3f} {np.max(np.abs(data)):>10.3f} "
                      f"{np.std(data):>10.3f}")
    
    # Data quality checks
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"⚠ Warning: {len(nan_cols)} columns contain NaN values:")
        for col in nan_cols[:10]:  # Show first 10
            nan_count = df[col].isna().sum()
            print(f"    {col}: {nan_count} NaN ({100*nan_count/len(df):.2f}%)")
        if len(nan_cols) > 10:
            print(f"    ... and {len(nan_cols)-10} more")
    else:
        print("✓ No NaN values found")
    
    # Check for constant columns (no variation)
    constant_cols = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if df[col].std() < 1e-9:
                constant_cols.append(col)
    
    if constant_cols:
        print(f"\n⚠ Warning: {len(constant_cols)} columns are constant (no variation):")
        for col in constant_cols[:10]:
            print(f"    {col}: {df[col].iloc[0]}")
        if len(constant_cols) > 10:
            print(f"    ... and {len(constant_cols)-10} more")
    
    # Check sample rate consistency
    time_diffs = np.diff(df['time'].values)
    expected_dt = 1.0 / 500.0  # 500 Hz
    outliers = np.abs(time_diffs - expected_dt) > 0.01  # >10ms deviation
    if np.any(outliers):
        print(f"\n⚠ Warning: {np.sum(outliers)} time steps deviate significantly from 500Hz")
        print(f"    Mean dt: {np.mean(time_diffs)*1000:.2f} ms (expected: {expected_dt*1000:.2f} ms)")
        print(f"    Max dt: {np.max(time_diffs)*1000:.2f} ms")
    else:
        print(f"\n✓ Sample rate is consistent (~{1.0/np.mean(time_diffs):.1f} Hz)")
    
    print("\n" + "="*70)
    print("Inspection complete!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
