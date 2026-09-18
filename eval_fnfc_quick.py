import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

print("[1] Loading FNFC data...")
df = pd.read_csv('knowledge_base/dynamic_fnfc_dinucleotides.csv')
print(f"Total FNFC samples: {len(df)}")
X = TrueGenomicFeatureExtractor(max_tfidf_features=50).fit_transform(df['Requirement_Text'].tolist())
y = df['Dinucleotide_Target'].values

print("[2] Evaluating top models on FNFC...")
clfs = {
    'Random Forest': get_paper_classifiers()['Random Forest'],
    'Gradient Boosting': get_paper_classifiers()['Gradient Boosting'],
    'Logistic Regression': get_paper_classifiers()['Logistic Regression'],
    'Decision Tree': get_paper_classifiers()['Decision Tree'],
    'KNN (k=3)': get_paper_classifiers()['KNN (k=3)']
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    for name, clf in clfs.items():
        print(f"    Evaluating {name} on FNFC...", end=" ", flush=True)
        try:
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_te, preds) * 100
            f1 = f1_score(y_te, preds, average='macro', zero_division=0) * 100
            print(f"Accuracy = {acc:.2f}% | Macro F1 = {f1:.2f}%", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
    break
print("[3] FNFC Evaluation Complete!")
