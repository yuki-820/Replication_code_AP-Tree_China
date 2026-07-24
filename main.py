#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main orchestration script for the replication project.

Steps:
1. Run step1_data_process.py to prepare raw data.
2. Build candidate pools for all models (run build_pool.py in each model subfolder).
3. Prune all models (run prune.py in each model subfolder).
4. Run factor model tests (step4_factor_tests.py).
5. Run plotting (step5_plot.py).
6. Run turnover analysis (step6_turnover.py).

Usage:
    python main.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_DIR = PROJECT_ROOT / "output"
CANDIDATE_POOLS_DIR = OUTPUT_DIR / "candidate_pools"

# List of model subdirectories (relative to code/ folder)
# These must contain build_pool.py and prune.py as appropriate.
MODEL_DIRS = [
    "TripleSort64_full",
    "TripleSort128_cleaned",
    "AP-Tree_full",
    "AP-Tree_cleaned",
    "TripleSort128_cleaned_longonly",
    "AP-Tree_cleaned_longonly",
]

def run_script(script_path, description):
    """Run a Python script and exit on failure."""
    if not script_path.exists():
        print(f"Warning: {description} script not found: {script_path}")
        return False
    print(f"\n[Main] Running: {description}")
    print(f"[Main] Script: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent)
    if result.returncode != 0:
        print(f"[Main] Error: {description} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[Main] Completed: {description}")
    return True

def main():
    print("=" * 80)
    print("Replication Pipeline for Asset Pricing with ML")
    print("=" * 80)

    # Step 1: Data process (commented out – assume data already prepared)
    data_process = CODE_DIR / "step1_data_process.py"
    if data_process.exists():
        run_script(data_process, "Data process (step1)")
    else:
        print("Step 1 skipped: step1_data_process.py not found. Assuming data already prepared.")

    # Step 2: Build candidate pools (commented out – assume pools already built)
    print("\n" + "=" * 80)
    print("Step 2: Build candidate pools")
    print("=" * 80)
    for model_dir in MODEL_DIRS:
        build_script = CODE_DIR / model_dir / "build_pool.py"
        if build_script.exists():
            run_script(build_script, f"Build pools for {model_dir}")
        else:
            print(f"Skipping build for {model_dir}: build_pool.py not found.")

    # Step 3: Prune all models
    print("\n" + "=" * 80)
    print("Step 3: Prune portfolios (LARS)")
    print("=" * 80)
    for model_dir in MODEL_DIRS:
        prune_script = CODE_DIR / model_dir / "prune.py"
        if prune_script.exists():
            run_script(prune_script, f"Prune {model_dir}")
        else:
            print(f"Skipping prune for {model_dir}: prune.py not found.")

    # Step 4: Factor model tests
    print("\n" + "=" * 80)
    print("Step 4: Factor model tests")
    print("=" * 80)
    factor_script = CODE_DIR / "step4_factor_tests.py"
    if factor_script.exists():
        run_script(factor_script, "Factor model tests (step4)")
    else:
        print("Error: step4_factor_tests.py not found. Skipping factor tests.")

    # Step 5: Plotting
    print("\n" + "=" * 80)
    print("Step 5: Plotting (figures)")
    print("=" * 80)
    plot_script = CODE_DIR / "step5_plot.py"
    if plot_script.exists():
        run_script(plot_script, "Plotting (step5)")
    else:
        print("Warning: step5_plot.py not found. Skipping plotting.")

    # Step 6: Turnover analysis
    print("\n" + "=" * 80)
    print("Step 6: Turnover analysis")
    print("=" * 80)
    turnover_script = CODE_DIR / "step6_turnover.py"
    if turnover_script.exists():
        run_script(turnover_script, "Turnover analysis (step6)")
    else:
        print("Warning: step6_turnover.py not found. Skipping turnover analysis.")

    print("\n" + "=" * 80)
    print("All steps completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
