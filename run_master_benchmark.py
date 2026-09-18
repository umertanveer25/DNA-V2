import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Paper 1 Reported Accuracies (from Paper Table 2 & Table 3)
PAPER_1_ACC = {
    'Random Forest': 78.67,
    'Gradient Boosting': 77.35,
    'SVM Linear': 76.23,
    'SVM RBF': 75.34,
    'Logistic Regression': 74.76,
    'AdaBoost': 73.78,
    'Naive Bayes': 72.23,
    'KNN (k=7)': 72.05,
    'KNN (k=5)': 71.53,
    'KNN (k=3)': 70.82,
    'Decision Tree': 69.88,
    'Multinomial NB': 66.42
}

def evaluate_dataset(csv_path, dataset_name, text_col='Requirement_Text', target_col='Dinucleotide_Target'):
    print(f"\n{'='*75}")
    print(f"[*] RUNNING MASTER BENCHMARK ON: {dataset_name.upper()}")
    print(f"{'='*75}", flush=True)
    
    df = pd.read_csv(csv_path)
    X_text = df[text_col].tolist()
    y = df[target_col].values
    
    print(f"[+] Total Requirements: {len(X_text)}")
    print(f"[+] Generating Biological Genomic Features + TF-IDF + SBERT...", flush=True)
    extractor = TrueGenomicFeatureExtractor(max_tfidf_features=50)
    X = extractor.fit_transform(X_text)
    print(f"[+] Feature Matrix Shape: {X.shape}", flush=True)
    
    X_non_neg = X.copy()
    X_non_neg -= X_non_neg.min()
    
    clfs = get_paper_classifiers()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    dataset_results = {}
    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        for name, clf in clfs.items():
            print(f"    Evaluating {name:20s}...", end=" ", flush=True)
            try:
                if name == "Multinomial NB":
                    clf.fit(X_non_neg[train_idx], y_tr)
                    preds = clf.predict(X_non_neg[test_idx])
                else:
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict(X_te)
                acc = round(accuracy_score(y_te, preds) * 100, 2)
                f1 = round(f1_score(y_te, preds, average='macro', zero_division=0) * 100, 2)
                dataset_results[name] = {'Accuracy': acc, 'Macro_F1': f1}
                print(f"Done! Accuracy: {acc:5.2f}% | Macro F1: {f1:5.2f}%", flush=True)
            except Exception as e:
                print(f"Skipped ({e})", flush=True)
                dataset_results[name] = {'Accuracy': 0.0, 'Macro_F1': 0.0}
        break
        
    return dataset_results

# 1. Evaluate PROMISE
promise_res = evaluate_dataset('knowledge_base/dynamic_promise_dinucleotides.csv', 'PROMISE')

# 2. Evaluate FNFC
fnfc_res = evaluate_dataset('knowledge_base/dynamic_fnfc_dinucleotides.csv', 'FNFC')

# 3. Create Master Comparison Table
rows = []
for algo in PAPER_1_ACC.keys():
    p1 = PAPER_1_ACC[algo]
    p2_prom = promise_res.get(algo, {}).get('Accuracy', 0.0)
    p2_prom_f1 = promise_res.get(algo, {}).get('Macro_F1', 0.0)
    p2_fnfc = fnfc_res.get(algo, {}).get('Accuracy', 0.0)
    p2_fnfc_f1 = fnfc_res.get(algo, {}).get('Macro_F1', 0.0)
    gain_prom = round(p2_prom - p1, 2)
    
    rows.append({
        'Algorithm': algo,
        'Paper 1 DNA Acc (%)': p1,
        'Paper 2 PROMISE Acc (%)': p2_prom,
        'PROMISE Gain (%)': f"+{gain_prom}%" if gain_prom > 0 else f"{gain_prom}%",
        'Paper 2 PROMISE F1 (%)': p2_prom_f1,
        'Paper 2 FNFC Acc (%)': p2_fnfc,
        'Paper 2 FNFC F1 (%)': p2_fnfc_f1
    })

res_df = pd.DataFrame(rows).sort_values(by='Paper 2 PROMISE Acc (%)', ascending=False)
print("\n" + "="*85)
print("🏆 MASTER BENCHMARK: PAPER 1 vs. PAPER 2 (DNA+ TRUE GENOMIC SEQUENCES)")
print("="*85)
print(res_df.to_string(index=False))

out_csv = "results/master_comparison_paper1_vs_paper2.csv"
res_df.to_csv(out_csv, index=False)
print(f"\n[+] Saved master comparison table to: {out_csv}")

# 4. Generate Master Publication Figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 7))

y_pos = np.arange(len(res_df))
h = 0.25

p1_vals = res_df['Paper 1 DNA Acc (%)'].values
p2_prom_vals = res_df['Paper 2 PROMISE Acc (%)'].values
p2_fnfc_vals = res_df['Paper 2 FNFC Acc (%)'].values

rects1 = ax.barh(y_pos - h, p1_vals, h, label='Paper 1 DNA Baseline', color='#7f8c8d', alpha=0.85)
rects2 = ax.barh(y_pos, p2_prom_vals, h, label='Paper 2 (Our Method - PROMISE)', color='#2980b9', alpha=0.95)
rects3 = ax.barh(y_pos + h, p2_fnfc_vals, h, label='Paper 2 (Our Method - FNFC Generalizability)', color='#27ae60', alpha=0.95)

ax.set_yticks(y_pos)
ax.set_yticklabels(res_df['Algorithm'], fontsize=11, fontweight='bold')
ax.set_xlabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Master Comparative Evaluation: Paper 1 vs. Paper 2 across PROMISE & FNFC', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 105)
ax.legend(loc='lower right', fontsize=11)

for r in rects2:
    w = r.get_width()
    if w > 0:
        ax.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(4, 0),
                    textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold', color='#1f4e78')

plt.tight_layout()
out_plot = "plots/master_publication_benchmark.png"
plt.savefig(out_plot, dpi=300)
plt.close()
print(f"[+] Saved publication figure to: {out_plot}")
