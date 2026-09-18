import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

print(f"\n{'='*75}")
print("[*] EVALUATING 12 CLASSIFIERS ON P-EXP (969 REQUIREMENTS)")
print(f"{'='*75}", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
X_text = df['Requirement_Text'].tolist()
y = df['Dinucleotide_Target'].values

print(f"[+] Generating True Genomic Features + TF-IDF + SBERT for {len(X_text)} requirements...", flush=True)
ext = TrueGenomicFeatureExtractor(max_tfidf_features=50)
X = ext.fit_transform(X_text)
print(f"[+] Feature matrix shape: {X.shape}", flush=True)

X_non_neg = X.copy()
X_non_neg -= X_non_neg.min()

clfs = get_paper_classifiers()
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []
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
            results.append({'Algorithm': name, 'Accuracy (%)': acc, 'Macro F1 (%)': f1})
            print(f"Accuracy: {acc:5.2f}% | Macro F1: {f1:5.2f}%", flush=True)
        except Exception as e:
            print(f"Error ({e})", flush=True)
    break

res_df = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False)
print("\n" + "="*60)
print("🏆 P-EXP BENCHMARK SUMMARY (969 REQUIREMENTS):")
print("="*60)
print(res_df.to_string(index=False))

res_df.to_csv('results/pexp_benchmark_results.csv', index=False)
print("\n[+] Saved results to: results/pexp_benchmark_results.csv")
