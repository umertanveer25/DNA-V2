import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class PureGenomicEvaluator:
    def __init__(self, classifiers: dict, random_state=42):
        self.classifiers = classifiers
        self.random_state = random_state

    def run_benchmark(self, X: np.ndarray, y: np.ndarray, num_splits=3) -> pd.DataFrame:
        results_acc = {name: [] for name in self.classifiers.keys()}
        results_f1 = {name: [] for name in self.classifiers.keys()}
        
        X_non_negative = X.copy()
        if X_non_negative.min() < 0:
            X_non_negative -= X_non_negative.min()

        y_series = pd.Series(y)
        min_class_count = y_series.value_counts().min()
        actual_splits = max(2, min(5, min_class_count))
        
        print(f"[*] Executing Stratified {actual_splits}-Fold CV across {num_splits} iteration(s) (NO SMOTE - Pure Distribution)...", flush=True)

        for split_idx in range(num_splits):
            skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=self.random_state + split_idx)
            
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                for name, clf in self.classifiers.items():
                    if name == "Multinomial NB":
                        X_tr = X_non_negative[train_idx]
                        X_te = X_non_negative[test_idx]
                    else:
                        X_tr = X_train
                        X_te = X_test

                    try:
                        clf.fit(X_tr, y_train)
                        preds = clf.predict(X_te)
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                        results_acc[name].append(acc)
                        results_f1[name].append(f1)
                    except Exception:
                        pass

        summary = []
        for name in self.classifiers.keys():
            acc_arr = np.array(results_acc[name]) if len(results_acc[name]) > 0 else np.array([0.0])
            f1_arr = np.array(results_f1[name]) if len(results_f1[name]) > 0 else np.array([0.0])
            summary.append({
                "Algorithm": name,
                "Accuracy (%)": round(acc_arr.mean() * 100, 2),
                "Acc Std (%)": round(acc_arr.std() * 100, 2),
                "Macro F1 (%)": round(f1_arr.mean() * 100, 2),
                "F1 Std (%)": round(f1_arr.std() * 100, 2)
            })

        df_res = pd.DataFrame(summary).sort_values(by="Accuracy (%)", ascending=False)
        return df_res
