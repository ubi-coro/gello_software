#!/usr/bin/env python3
"""
Teleoperation Task Statistics Plot (Section 4.3.4)

Statistical comparison of task performance with/without force feedback.
Creates boxplots showing peak forces or task completion times.

Usage:
    python plot_teleoperation_statistics.py --trials trials.csv --output teleoperation_stats.svg
    
    Or provide multiple log files:
    python plot_teleoperation_statistics.py --logs trial1.csv trial2.csv ... --output stats.svg

trials.csv format:
    condition,max_force,task_time
    without_feedback,15.3,8.2
    without_feedback,18.1,9.1
    ...
    with_feedback,8.2,7.5
    with_feedback,9.1,7.8
    ...
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Scientific plotting style
sns.set_context("paper", font_scale=1.4)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'svg',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def extract_max_force_from_log(logfile, tcp_axis=2):
    """Extract maximum TCP force from a log file."""
    try:
        df = pd.read_csv(logfile)
        force = df[f'tcp_force_{tcp_axis}'].values
        return np.max(np.abs(force))
    except Exception as e:
        print(f"Warning: Could not extract force from {logfile}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of teleoperation performance")
    parser.add_argument("--trials", type=str, help="CSV file with trial statistics")
    parser.add_argument("--logs", type=str, nargs='+', 
                        help="Multiple log files (alternating: without/with feedback)")
    parser.add_argument("--output", type=str, default="teleoperation_statistics.svg", 
                        help="Output filename")
    parser.add_argument("--metric", type=str, default="max_force", 
                        choices=["max_force", "task_time"],
                        help="Metric to plot (default: max_force)")
    parser.add_argument("--tcp-axis", type=int, default=2,
                        help="TCP axis for force extraction (0-5, default: 2=Fz)")
    args = parser.parse_args()

    # Load data
    if args.trials:
        # Load from summary CSV
        print(f"Loading trial data from {args.trials}...")
        df = pd.read_csv(args.trials)
    elif args.logs:
        # Extract from individual log files
        print(f"Extracting data from {len(args.logs)} log files...")
        
        # Assume alternating: without, with, without, with, ...
        data = []
        for i, logfile in enumerate(args.logs):
            condition = "Without Feedback" if i % 2 == 0 else "With Feedback"
            max_force = extract_max_force_from_log(logfile, args.tcp_axis)
            if max_force is not None:
                data.append({
                    'condition': condition,
                    'max_force': max_force,
                    'trial': i // 2 + 1
                })
        
        df = pd.DataFrame(data)
    else:
        print("Error: Must provide either --trials or --logs")
        return 1
    
    if len(df) == 0:
        print("Error: No data loaded")
        return 1
    
    print(f"Loaded {len(df)} trials")
    print(df.groupby('condition')[args.metric].describe())
    
    # Statistical test (Mann-Whitney U test for non-parametric comparison)
    without_feedback = df[df['condition'].str.contains('ithout', case=False)][args.metric].values
    with_feedback = df[df['condition'].str.contains('ith', case=False) & 
                       ~df['condition'].str.contains('ithout', case=False)][args.metric].values
    
    if len(without_feedback) > 0 and len(with_feedback) > 0:
        statistic, p_value = stats.mannwhitneyu(without_feedback, with_feedback, 
                                                 alternative='two-sided')
        
        print(f"\n=== Statistical Test (Mann-Whitney U) ===")
        print(f"Without Feedback: n={len(without_feedback)}, median={np.median(without_feedback):.2f}")
        print(f"With Feedback: n={len(with_feedback)}, median={np.median(with_feedback):.2f}")
        print(f"U-statistic: {statistic:.2f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No significant difference (p >= 0.05)")
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Color palette
    colors = ['#3498db', '#e74c3c']  # Blue without, red with
    
    # Create boxplot
    bp = ax.boxplot(
        [without_feedback, with_feedback],
        labels=['Without\nForce Feedback', 'With\nForce Feedback'],
        patch_artist=True,
        widths=0.6,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2.5, color='black'),
    )
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points (jittered)
    for i, data in enumerate([without_feedback, with_feedback], start=1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=80, c=colors[i-1], edgecolors='black', linewidths=0.8)
    
    # Labels and title
    if args.metric == "max_force":
        ylabel = "Maximum Contact Force [N]"
        title = "Teleoperation Task Performance:\nPeak Contact Forces"
    else:
        ylabel = "Task Completion Time [s]"
        title = "Teleoperation Task Performance:\nCompletion Time"
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical significance annotation
    if len(without_feedback) > 0 and len(with_feedback) > 0:
        y_max = max(np.max(without_feedback), np.max(with_feedback))
        y_range = y_max - min(np.min(without_feedback), np.min(with_feedback))
        
        # Draw significance bar
        bar_height = y_max + 0.05 * y_range
        ax.plot([1, 2], [bar_height, bar_height], 'k-', linewidth=1.5)
        ax.plot([1, 1], [bar_height - 0.01 * y_range, bar_height], 'k-', linewidth=1.5)
        ax.plot([2, 2], [bar_height - 0.01 * y_range, bar_height], 'k-', linewidth=1.5)
        
        # Add p-value text
        if p_value < 0.001:
            sig_text = "***"
        elif p_value < 0.01:
            sig_text = "**"
        elif p_value < 0.05:
            sig_text = "*"
        else:
            sig_text = "n.s."
        
        ax.text(1.5, bar_height + 0.02 * y_range, f'{sig_text}\n(p={p_value:.4f})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add summary statistics box
    stats_text = f'Without Feedback:\n  Median: {np.median(without_feedback):.2f}\n  IQR: {np.percentile(without_feedback, 75) - np.percentile(without_feedback, 25):.2f}\n\n'
    stats_text += f'With Feedback:\n  Median: {np.median(with_feedback):.2f}\n  IQR: {np.percentile(with_feedback, 75) - np.percentile(with_feedback, 25):.2f}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150)
    print(f"Saved preview to {png_path}")
    
    plt.show()
    return 0


if __name__ == "__main__":
    exit(main())
