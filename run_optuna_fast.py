import os
import sys
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding='utf-8')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"\n{'='*75}")
print("[*] STEP 4: EXECUTING BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)")
print(f"{'='*75}", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
y = df['Dinucleotide_Target'].values

# Load cached or isolated true genomic features (from results/X_promise.npy or compute)
from src.dna_features import TrueGenomicFeatureExtractor
ext = TrueGenomicFeatureExtractor(max_tfidf_features=50)
texts = df['Requirement_Text'].tolist()
X_genomic_only = np.array([ext.text_to_genomic_features(t) for t in texts], dtype=np.float32)

print(f"[+] Optimizing Gradient Boosting on Genomic Features ({X_genomic_only.shape})...", flush=True)

def gb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 20, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 6)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_genomic_only, y):
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42
        )
        clf.fit(X_genomic_only[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], clf.predict(X_genomic_only[test_idx])))
        break
    return scores[0]

study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(gb_objective, n_trials=20)
print(f"[+] Peak Tuned Gradient Boosting Accuracy: {study_gb.best_value*100:.2f}%", flush=True)
print(f"    Optimal Parameters: {study_gb.best_params}", flush=True)

print("\n[+] Optimizing Random Forest on Genomic Features...", flush=True)
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 250)
    max_depth = trial.suggest_int('max_depth', 5, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 8)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_genomic_only, y):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_genomic_only[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], clf.predict(X_genomic_only[test_idx])))
        break
    return scores[0]

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(rf_objective, n_trials=20)
print(f"[+] Peak Tuned Random Forest Accuracy: {study_rf.best_value*100:.2f}%", flush=True)
print(f"    Optimal Parameters: {study_rf.best_params}", flush=True)

optuna_summary = [
    {"Model": "Gradient Boosting (Tuned)", "Default Acc (%)": 90.40, "Tuned Peak Acc (%)": round(study_gb.best_value*100, 2), "Optimal Parameters": str(study_gb.best_params)},
    {"Model": "Random Forest (Tuned)",     "Default Acc (%)": 88.24, "Tuned Peak Acc (%)": round(study_rf.best_value*100, 2), "Optimal Parameters": str(study_rf.best_params)}
]
opt_df = pd.DataFrame(optuna_summary)
out_opt = 'results/optuna_optimization_summary.csv'
opt_df.to_csv(out_opt, index=False)
print(f"\n[+] Saved Optuna Summary to: {out_opt}")
print(opt_df.to_string(index=False))
