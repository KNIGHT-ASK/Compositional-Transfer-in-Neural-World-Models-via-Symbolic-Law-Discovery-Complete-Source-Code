#!/usr/bin/env python
"""
Kaggle P100 Runner — JMLR Revision Experiments
===============================================
Upload this script to Kaggle with both src/ files attached as datasets.

Runs in this order:
  1. exp_table2_complete.py  → fills all Table 2 metrics (W3 + W5)
  2. exp_gnn_baseline.py      → Interaction Network + GNS results (W1)

Outputs:
  - table2_complete_results.json
  - gnn_baseline_results.json
  Both are printed to the notebook output and also available for download.
"""

import subprocess, sys, json, os

def run_exp(script_name):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,  # stream output live
        check=True
    )
    return result.returncode

# ── Install deps if needed ──
try:
    import sklearn
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-q'], check=True)

# ── Make sure CUDA is visible ──
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Set paths (adjust if scripts are in /kaggle/input/) ──
SCRIPT_DIR = "/kaggle/working"   # or wherever you put the .py files

os.chdir(SCRIPT_DIR)

# ── Run experiments ──
run_exp("exp_table2_complete.py")
run_exp("exp_gnn_baseline.py")

# ── Print final summary ──
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)

for fname in ["table2_complete_results.json", "gnn_baseline_results.json"]:
    if os.path.exists(fname):
        with open(fname) as f:
            data = json.load(f)
        print(f"\n{'─'*40}")
        print(f"  {fname}")
        print(f"{'─'*40}")
        print(json.dumps(data, indent=2))

print("\n>>> Download the two JSON files, then give them to Antigravity to update paper.tex <<<")
