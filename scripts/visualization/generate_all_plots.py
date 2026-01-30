#!/usr/bin/env python3
"""
Generate All Thesis Plots

Convenience script to generate all visualization plots at once.
Assumes you have recorded the necessary log files.

Usage:
    python generate_all_plots.py --data-dir logs/ --output-dir thesis_plots/

Directory structure expected:
    logs/
        friction_data.csv           # For friction identification
        dynamics_validation.csv     # For dynamics validation
        gravity_hold.csv            # For gravity compensation
        force_feedback.csv          # For force-torque correlation
        position_contact.csv        # For contact comparison
        impedance_contact.csv       # For contact comparison
        trials.csv                  # For statistics (or use --logs)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_plot(script, args, description):
    """Run a plotting script."""
    cmd = [sys.executable, script] + args
    print(f"\n{'='*70}")
    print(f"Generating: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"  Error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"✗ Failed: {description}")
        print(f"  Error: File not found - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate all thesis plots")
    parser.add_argument("--data-dir", type=str, default="logs", help="Directory containing log files")
    parser.add_argument("--output-dir", type=str, default="thesis_plots", help="Output directory for plots")
    parser.add_argument("--skip", type=str, nargs='+', default=[], 
                        help="Skip specific plots (friction, dynamics, filter, gravity, force, contact, stats)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    viz_dir = Path(__file__).parent
    
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Visualization scripts: {viz_dir.absolute()}")
    
    # Track success
    results = {}
    
    # 1. Friction Identification
    if 'friction' not in args.skip:
        friction_log = data_dir / "friction_data.csv"
        if friction_log.exists():
            results['friction'] = run_plot(
                str(viz_dir / "plot_friction_identification.py"),
                [str(friction_log), "--joint", "2", "--output", str(output_dir / "friction_identification.svg")],
                "Friction Identification (Section 4.2.1)"
            )
        else:
            print(f"\n⚠ Skipping friction plot: {friction_log} not found")
            results['friction'] = None
    
    # 2. Dynamics Validation
    if 'dynamics' not in args.skip:
        dynamics_log = data_dir / "dynamics_validation.csv"
        if dynamics_log.exists():
            results['dynamics'] = run_plot(
                str(viz_dir / "plot_dynamics_validation.py"),
                [str(dynamics_log), "--output", str(output_dir / "dynamics_validation.svg")],
                "Dynamics Model Validation (Section 4.2.1)"
            )
        else:
            print(f"\n⚠ Skipping dynamics plot: {dynamics_log} not found")
            results['dynamics'] = None
    
    # 3. Filter Comparison
    if 'filter' not in args.skip:
        filter_log = data_dir / "friction_data.csv"  # Can reuse friction data
        if filter_log.exists():
            results['filter'] = run_plot(
                str(viz_dir / "plot_filter_comparison.py"),
                [str(filter_log), "--joint", "4", "--output", str(output_dir / "filter_comparison.svg")],
                "Filter Comparison (Section 4.2.2)"
            )
        else:
            print(f"\n⚠ Skipping filter plot: {filter_log} not found")
            results['filter'] = None
    
    # 4. Gravity Compensation
    if 'gravity' not in args.skip:
        gravity_log = data_dir / "gravity_hold.csv"
        if gravity_log.exists():
            results['gravity'] = run_plot(
                str(viz_dir / "plot_gravity_compensation.py"),
                [str(gravity_log), "--joints", "1", "2", "3", "--output", str(output_dir / "gravity_compensation.svg")],
                "Gravity Compensation Validation (Section 4.3.1)"
            )
        else:
            print(f"\n⚠ Skipping gravity plot: {gravity_log} not found")
            results['gravity'] = None
    
    # 5. Force-Torque Correlation
    if 'force' not in args.skip:
        force_log = data_dir / "force_feedback.csv"
        if force_log.exists():
            results['force'] = run_plot(
                str(viz_dir / "plot_force_torque_correlation.py"),
                [str(force_log), "--joint", "2", "--tcp-axis", "2", 
                 "--output", str(output_dir / "force_torque_correlation.svg")],
                "Force-Torque Correlation (Section 4.3.2)"
            )
        else:
            print(f"\n⚠ Skipping force plot: {force_log} not found")
            results['force'] = None
    
    # 6. Contact Behavior
    if 'contact' not in args.skip:
        pos_log = data_dir / "position_contact.csv"
        imp_log = data_dir / "impedance_contact.csv"
        if pos_log.exists() and imp_log.exists():
            results['contact'] = run_plot(
                str(viz_dir / "plot_contact_behavior.py"),
                [str(pos_log), str(imp_log), "--output", str(output_dir / "contact_behavior.svg")],
                "Contact Behavior Comparison (Section 4.3.3)"
            )
        else:
            print(f"\n⚠ Skipping contact plot: Missing {pos_log} or {imp_log}")
            results['contact'] = None
    
    # 7. Teleoperation Statistics
    if 'stats' not in args.skip:
        trials_csv = data_dir / "trials.csv"
        if trials_csv.exists():
            results['stats'] = run_plot(
                str(viz_dir / "plot_teleoperation_statistics.py"),
                ["--trials", str(trials_csv), "--metric", "max_force", 
                 "--output", str(output_dir / "teleoperation_statistics.svg")],
                "Teleoperation Task Statistics (Section 4.3.4)"
            )
        else:
            print(f"\n⚠ Skipping stats plot: {trials_csv} not found")
            print("  Hint: Create trials.csv or use --logs with individual trial files")
            results['stats'] = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success_count = sum(1 for v in results.values() if v is True)
    failed_count = sum(1 for v in results.values() if v is False)
    skipped_count = sum(1 for v in results.values() if v is None)
    
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"⊘ Skipped (missing data): {skipped_count}")
    
    if success_count > 0:
        print(f"\nPlots saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        for svg_file in sorted(output_dir.glob("*.svg")):
            print(f"  - {svg_file.name}")
    
    print("\n" + "="*70)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
