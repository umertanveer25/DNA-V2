#!/usr/bin/env python3
"""
================================================================================
DNA+ True Biological Genomic Sequences for Software Requirements Classification
Master Experimental Replication Suite
================================================================================
One-click CLI runner to reproduce all empirical benchmarks, statistical tests,
explainability attributions, diagnostic curves, and ablation studies reported
in Paper 2.
"""

import sys
import os
import argparse
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

BANNER = r"""
================================================================================
   ____  _   _    _         _     ____                          
  |  _ \| \ | |  / \   _   | |   |  _ \ __ _ _ __   ___ _ __ 2  
  | | | |  \| | / _ \ (_) _| |_  | |_) / _` | '_ \ / _ \ '__|   
  | |_| | |\  |/ ___ \ _ |_   _| |  __/ (_| | |_) |  __/ |      
  |____/|_| \_/_/   \_(_)  |_|   |_|   \__,_| .__/ \___|_|      
                                            |_|                 
  True Biological Genomic Sequences & Dynamic Syntactic-Semantic Mapping
================================================================================
"""

SCRIPTS = {
    'benchmark': {
        'desc': 'Tri-Dataset Master Benchmark (PROMISE, P-EXP, FNFC across 12 Classifiers)',
        'cmd': [sys.executable, '-u', 'run_master_benchmark.py']
    },
    'stats': {
        'desc': 'Formal Statistical Significance Tests (Wilcoxon Signed-Rank & Cohen\'s d)',
        'cmd': [sys.executable, '-u', 'run_stats.py']
    },
    'shap': {
        'desc': 'Global Feature Attribution via SHAP TreeExplainer',
        'cmd': [sys.executable, '-u', 'run_shap.py']
    },
    'lime': {
        'desc': 'Local Instance Interpretability via LIME Explanations',
        'cmd': [sys.executable, '-u', 'run_lime.py']
    },
    'ablation': {
        'desc': 'Component Ablation Study (Evaluating Contribution of Each Genomic Feature Group)',
        'cmd': [sys.executable, '-u', 'run_ablation.py']
    },
    'roc': {
        'desc': 'Multi-Algorithm Receiver Operating Characteristic (ROC) & AUC Curves',
        'cmd': [sys.executable, '-u', 'generate_multi_roc_curve.py']
    },
    'prc': {
        'desc': 'Multi-Algorithm Precision-Recall Curves (PRC) & Confusion Matrix Heatmap',
        'cmd': [sys.executable, '-u', 'generate_prc_and_diagnostics.py']
    },
    'optuna': {
        'desc': 'Bayesian Hyperparameter Optimization via Optuna Engine',
        'cmd': [sys.executable, '-u', 'run_optuna_fast.py']
    }
}

def run_step(step_key):
    step = SCRIPTS[step_key]
    print(f"\n{'#'*80}")
    print(f"[>] EXECUTING STEP: {step_key.upper()} - {step['desc']}")
    print(f"{'#'*80}\n", flush=True)
    res = subprocess.run(step['cmd'])
    if res.returncode != 0:
        print(f"\n[-] Step '{step_key}' exited with non-zero status code: {res.returncode}", flush=True)
        return False
    print(f"\n[+] Step '{step_key}' completed successfully!", flush=True)
    return True

def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="DNA+ Paper 2 Replication Suite: Execute any or all empirical experiments."
    )
    parser.add_argument('--all', action='store_true', help='Execute all experiments sequentially')
    parser.add_argument('--benchmark', action='store_true', help='Run Tri-Dataset Master Benchmark')
    parser.add_argument('--stats', action='store_true', help='Run Wilcoxon and Cohen\'s d statistical tests')
    parser.add_argument('--shap', action='store_true', help='Run global SHAP feature attribution')
    parser.add_argument('--lime', action='store_true', help='Run local LIME instance explanations')
    parser.add_argument('--ablation', action='store_true', help='Run component ablation study')
    parser.add_argument('--roc', action='store_true', help='Generate Multi-Algorithm ROC curves')
    parser.add_argument('--prc', action='store_true', help='Generate Precision-Recall curves & Confusion Matrix')
    parser.add_argument('--optuna', action='store_true', help='Run Bayesian hyperparameter optimization')

    args = parser.parse_args()

    # If no flags provided, display help and menu
    if not any(vars(args).values()):
        parser.print_help()
        print("\n[!] Quick run examples:")
        print("    python reproduce_all.py --benchmark")
        print("    python reproduce_all.py --roc")
        print("    python reproduce_all.py --prc")
        print("    python reproduce_all.py --all\n")
        return

    selected_steps = []
    if args.all:
        selected_steps = ['benchmark', 'stats', 'ablation', 'roc', 'prc', 'shap', 'lime', 'optuna']
    else:
        for k in SCRIPTS.keys():
            if getattr(args, k, False):
                selected_steps.append(k)

    total = len(selected_steps)
    passed = 0
    for idx, step_key in enumerate(selected_steps, 1):
        print(f"\n>>> Progress: Running [{idx}/{total}] : {step_key.upper()}...")
        ok = run_step(step_key)
        if ok:
            passed += 1

    print(f"\n{'='*80}")
    print(f"🏆 REPLICATION EXECUTION FINISHED: {passed}/{total} steps succeeded.")
    print(f"   Outputs and metrics available in 'results/' and publication plots in 'plots/'")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
