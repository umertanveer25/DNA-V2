import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

print("[*] Generating Detailed Per-Class Breakdown for P-EXP (969 samples)...", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
X = TrueGenomicFeatureExtractor(max_tfidf_features=50).fit_transform(df['Requirement_Text'].tolist())
y = df['Dinucleotide_Target'].values

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    # 1. Logistic Regression (balanced)
    clf_lr = get_paper_classifiers()['Logistic Regression']
    clf_lr.fit(X_tr, y_tr)
    preds_lr = clf_lr.predict(X_te)
    
    # 2. Decision Tree (balanced)
    clf_dt = get_paper_classifiers()['Decision Tree']
    clf_dt.fit(X_tr, y_tr)
    preds_dt = clf_dt.predict(X_te)
    
    print("\n" + "="*70)
    print("📋 PER-CLASS CLASSIFICATION REPORT: LOGISTIC REGRESSION")
    print("="*70)
    print(classification_report(y_te, preds_lr, zero_division=0))
    
    print("\n" + "="*70)
    print("📋 PER-CLASS CLASSIFICATION REPORT: DECISION TREE")
    print("="*70)
    print(classification_report(y_te, preds_dt, zero_division=0))
    break
