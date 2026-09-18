import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

print("=" * 80)
print("             DEEP DATASET PROFILING & LINGUISTIC UNDERSTANDING")
print("=" * 80)

# -------------------------------------------------------------
# 1. PROMISE NFR DATASET
# -------------------------------------------------------------
df_p = pd.read_csv(r"B:\Promise\Paper2\data\promise_nfr.csv", encoding="ISO-8859-1")
print("\n[DATASET 1: PROMISE NFR (Cleland-Huang et al.)]")
print(f"Total Rows: {len(df_p)}")
print(f"Projects count: {df_p['ProjectID'].nunique()} distinct software systems")
print(f"Projects list: {sorted(df_p['ProjectID'].unique().tolist())}")

print("\nClass Distribution:")
vc_p = df_p['class'].value_counts()
for c, cnt in vc_p.items():
    pct = cnt / len(df_p) * 100
    print(f"  {c:4s} : {cnt:4d}  ({pct:5.2f}%)")

print("\n--- Real Text Samples for Every Class in PROMISE ---")
for c in vc_p.index:
    samples = df_p[df_p['class'] == c]['RequirementText'].head(2).tolist()
    print(f"\n>> CLASS: {c} (Total: {vc_p[c]})")
    for i, s in enumerate(samples, 1):
        clean = s.strip().strip("'").strip('"')
        print(f"   [{i}] {clean}")

# -------------------------------------------------------------
# 2. FNFC DATASET
# -------------------------------------------------------------
df_f = pd.read_csv(r"B:\Promise\Paper2\data\fnfc.csv", encoding="ISO-8859-1")
print("\n" + "=" * 80)
print("[DATASET 2: FNFC (Functional and Non-Functional Classification)]")
print(f"Total Rows: {len(df_f)}")
print(f"Columns: {list(df_f.columns)}")

print("\nClass Distribution:")
vc_f = df_f['class'].value_counts()
for c, cnt in vc_f.items():
    pct = cnt / len(df_f) * 100
    print(f"  {c:4s} : {cnt:4d}  ({pct:5.2f}%)")

print("\n--- Real Text Samples for Every Class in FNFC ---")
for c in vc_f.index:
    samples = df_f[df_f['class'] == c]['text'].head(2).tolist()
    print(f"\n>> CLASS: {c} (Total: {vc_f[c]})")
    for i, s in enumerate(samples, 1):
        clean = s.strip().strip("'").strip('"')
        print(f"   [{i}] {clean}")

