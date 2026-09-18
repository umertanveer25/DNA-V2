import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class RobustEvaluator:
    def __init__(self, classifiers: dict, random_state=42):
        self.classifiers = classifiers
        self.random_state = random_state

    def run_benchmark(self, X: np.ndarray, y: np.ndarray, num_splits=10, use_smote=True) -> pd.DataFrame:
        results_acc = {name: [] for name in self.classifiers.keys()}
        results_f1 = {name: [] for name in self.classifiers.keys()}
        
        X_non_negative = X.copy()
        if X_non_negative.min() < 0:
            X_non_negative -= X_non_negative.min()

        print(f"[*] Executing 10-Fold CV with {num_splits} iterations (SMOTE={'Enabled' if use_smote else 'Disabled'})...")
        
        # Check minimum class frequency for SMOTE k_neighbors
        y_series = pd.Series(y)
        min_class_count = y_series.value_counts().min()
        k_neighbors = max(1, min(5, min_class_count - 1)) if min_class_count > 1 else 1

        for split_idx in range(num_splits):
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state + split_idx)
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                # Apply SMOTE inside training fold only
                if use_smote and min_class_count > 1:
                    try:
                        smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                    except Exception:
                        X_train_res, y_train_res = X_train, y_train
                else:
                    X_train_res, y_train_res = X_train, y_train

                for name, clf in self.classifiers.items():
                    if name == "Multinomial NB":
                        X_tr = X_non_negative[train_idx]
                        X_te = X_non_negative[test_idx]
                    else:
                        X_tr = X_train_res
                        X_te = X_test

                    try:
                        clf.fit(X_tr, y_train_res if name != "Multinomial NB" else y_train)
                        preds = clf.predict(X_te)
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                        results_acc[name].append(acc)
                        results_f1[name].append(f1)
                    except Exception as e:
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
