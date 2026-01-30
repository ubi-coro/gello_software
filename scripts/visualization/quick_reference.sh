#!/bin/bash
# Quick Reference Script for GELLO Thesis Experiments
# This script provides easy-to-copy commands for each experimental scenario

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "===================================================================="
echo "GELLO Thesis Experiment Quick Reference"
echo "===================================================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_section() {
    echo ""
    echo -e "${GREEN}==== $1 ====${NC}"
    echo ""
}

show_command() {
    echo -e "${BLUE}$1:${NC}"
    echo "  $2"
    echo ""
}

show_section "4.2.1a: Friction Identification"
show_command "Record data" "python gello/factr/gravity_compensation.py -c configs/examples/gravity_comp_validation.yaml --log"
show_command "Rename log" "mv logs/gello_data_*.csv logs/friction_data.csv"
show_command "Generate plot" "python scripts/visualization/plot_friction_identification.py logs/friction_data.csv --joint 2 -o plots/friction.svg"

show_section "4.2.1b: Dynamics Model Validation"
show_command "Record data" "python gello/factr/gravity_compensation.py -c configs/examples/gravity_comp_validation.yaml --log"
show_command "Rename log" "mv logs/gello_data_*.csv logs/dynamics_validation.csv"
show_command "Generate plot" "python scripts/visualization/plot_dynamics_validation.py logs/dynamics_validation.csv -o plots/dynamics.svg"

show_section "4.2.2: Filter Comparison"
show_command "Generate plot" "python scripts/visualization/plot_filter_comparison.py logs/friction_data.csv --joint 4 -o plots/filter.svg"

show_section "4.3.1: Gravity Compensation Hold Test"
show_command "Record data" "python gello/factr/gravity_compensation.py -c configs/examples/gravity_comp_validation.yaml --log"
show_command "Rename log" "mv logs/gello_data_*.csv logs/gravity_hold.csv"
show_command "Generate plot" "python scripts/visualization/plot_gravity_compensation.py logs/gravity_hold.csv --joints 1 2 3 -o plots/gravity.svg"

show_section "4.3.2: Force-Torque Correlation"
show_command "Record data" "python gello/factr/gravity_compensation.py -c configs/examples/with_force_feedback.yaml --log"
show_command "Rename log" "mv logs/gello_data_*.csv logs/force_feedback.csv"
show_command "Generate plot" "python scripts/visualization/plot_force_torque_correlation.py logs/force_feedback.csv --joint 2 --tcp-axis 2 -o plots/force_corr.svg"

show_section "4.3.3: Contact Behavior Comparison"
show_command "Record Position" "python gello/factr/gravity_compensation.py -c configs/examples/position_mode_test.yaml --log"
show_command "Rename Position" "mv logs/gello_data_*.csv logs/position_contact.csv"
show_command "Record Impedance" "python gello/factr/gravity_compensation.py -c configs/examples/impedance_mode_test.yaml --log"
show_command "Rename Impedance" "mv logs/gello_data_*.csv logs/impedance_contact.csv"
show_command "Generate plot" "python scripts/visualization/plot_contact_behavior.py logs/position_contact.csv logs/impedance_contact.csv -o plots/contact.svg"

show_section "4.3.4: Teleoperation Statistics"
echo -e "${BLUE}Record 10 trials WITHOUT feedback:${NC}"
echo "  for i in {1..10}; do"
echo "    python gello/factr/gravity_compensation.py -c configs/examples/without_force_feedback.yaml --log"
echo "    mv logs/gello_data_*.csv logs/trial_no_fb_\${i}.csv"
echo "  done"
echo ""
echo -e "${BLUE}Record 10 trials WITH feedback:${NC}"
echo "  for i in {1..10}; do"
echo "    python gello/factr/gravity_compensation.py -c configs/examples/with_force_feedback.yaml --log"
echo "    mv logs/gello_data_*.csv logs/trial_with_fb_\${i}.csv"
echo "  done"
echo ""
show_command "Generate plot" "python scripts/visualization/plot_teleoperation_statistics.py --logs logs/trial_*.csv -o plots/stats.svg"

show_section "Generate All Plots at Once"
show_command "Master script" "python scripts/visualization/generate_all_plots.py --data-dir logs --output-dir plots"

show_section "Utility Commands"
show_command "Inspect log" "python scripts/visualization/inspect_log.py logs/gello_data_*.csv"
show_command "Check config" "python -c \"import yaml; yaml.safe_load(open('configs/my_config.yaml'))\""
show_command "Quick test" "timeout 10 python gello/factr/gravity_compensation.py -c configs/my_config.yaml --log"

echo ""
echo -e "${YELLOW}Tip: Copy-paste commands directly into your terminal!${NC}"
echo "===================================================================="
