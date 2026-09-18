import os
import sys
import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, '.')
from src.dna_features import EnhancedDNAFeatureExtractor

def tune_random_forest(X, y, n_trials=30):
    print("\n[+] Tuning Random Forest via Optuna...")
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            try:
                sm = SMOTE(k_neighbors=2, random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except:
                pass
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(accuracy_score(y_te, preds))
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"    Best RF Accuracy: {study.best_value*100:.2f}% with params: {study.best_params}")
    return study.best_value, study.best_params

def tune_svm_rbf(X, y, n_trials=30):
    print("\n[+] Tuning SVM RBF via Optuna...")
    def objective(trial):
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            try:
                sm = SMOTE(k_neighbors=2, random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except:
                pass
            clf = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(accuracy_score(y_te, preds))
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"    Best SVM RBF Accuracy: {study.best_value*100:.2f}% with params: {study.best_params}")
    return study.best_value, study.best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='promise', choices=['promise', 'fnfc'])
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    
    kb_path = f"knowledge_base/{args.dataset}_knowledge_base.csv"
    df = pd.read_csv(kb_path)
    X_text = df['Requirement_Text'].tolist()
    y = df['Assigned_DNA'].values
    
    print(f"[*] Extracting features for {args.dataset}...")
    ext = EnhancedDNAFeatureExtractor(max_tfidf_features=50)
    X = ext.fit_transform(X_text)
    
    rf_acc, rf_params = tune_random_forest(X, y, n_trials=args.trials)
    svm_acc, svm_params = tune_svm_rbf(X, y, n_trials=args.trials)
    
    res = [
        {"Model": "Random Forest (Tuned)", "Accuracy (%)": round(rf_acc*100, 2), "Best_Params": str(rf_params)},
        {"Model": "SVM RBF (Tuned)", "Accuracy (%)": round(svm_acc*100, 2), "Best_Params": str(svm_params)}
    ]
    res_df = pd.DataFrame(res)
    os.makedirs('results', exist_ok=True)
    res_df.to_csv(f"results/{args.dataset}_optuna_results.csv", index=False)
    print(f"\n[+] Saved Optuna results to results/{args.dataset}_optuna_results.csv")
