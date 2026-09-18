import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

print("[1] Loading data...")
df = pd.read_csv('knowledge_base/dynamic_promise_dinucleotides.csv')
X = TrueGenomicFeatureExtractor(max_tfidf_features=50).fit_transform(df['Requirement_Text'].tolist())
y = df['Dinucleotide_Target'].values

print("[2] Initializing classifiers...")
clfs = get_paper_classifiers()

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    # test Multinomial NB non-negative
    X_non_neg = X.copy()
    X_non_neg -= X_non_neg.min()
    
    for name, clf in clfs.items():
        print(f"    Testing {name}...", end=" ", flush=True)
        try:
            if name == "Multinomial NB":
                clf.fit(X_non_neg[train_idx], y_tr)
                acc = accuracy_score(y_te, clf.predict(X_non_neg[test_idx]))
            else:
                clf.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, clf.predict(X_te))
            print(f"OK! Accuracy = {acc*100:.2f}%", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
    break
print("[3] Test loop complete!")
