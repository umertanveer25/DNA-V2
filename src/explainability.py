import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.dna_features import EnhancedDNAFeatureExtractor

def run_shap_analysis(kb_csv='knowledge_base/promise_knowledge_base.csv', dataset_name='promise'):
    print(f"\n[+] Running SHAP & Linguistic Explainability on {dataset_name.upper()}...")
    df = pd.read_csv(kb_csv)
    X_text = df['Requirement_Text'].tolist()
    y = df['Assigned_DNA'].values
    
    ext = EnhancedDNAFeatureExtractor(max_tfidf_features=50)
    X = ext.fit_transform(X_text)
    
    # Feature Names
    dna_names = [
        'DNA_Base_A', 'DNA_Base_T', 'DNA_Base_C', 'DNA_Base_G',
        'Pair_AT', 'Pair_AG', 'Pair_AC', 'Pair_TG', 'Pair_TC', 'Pair_GC',
        'Ratio_AT', 'Ratio_GC', 'Ratio_ATGC', 'DNA_Shannon_Entropy'
    ]
    tfidf_names = [f"TFIDF_{term}" for term in ext.tfidf_vectorizer.get_feature_names_out()]
    sbert_names = [f"SBERT_dim_{i}" for i in range(384)]
    feature_names = dna_names + tfidf_names + sbert_names
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    try:
        sm = SMOTE(k_neighbors=2, random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    except:
        X_tr_res, y_tr_res = X_tr, y_tr
        
    clf = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=42)
    clf.fit(X_tr_res, y_tr_res)
    
    print("[*] Calculating SHAP TreeExplainer values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_te)
    
    # Average across all classes to find overall feature importance
    if isinstance(shap_values, list):
        # multi-class list of arrays
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    importance_df.to_csv(f"results/{dataset_name}_shap_importance.csv", index=False)
    print(f"[+] Top 15 Most Influential Features across {dataset_name.upper()}:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot Bar Chart
    top15 = importance_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    colors = ['#2b5c8f' if 'DNA' in f or 'Pair' in f or 'Ratio' in f or 'Entropy' in f else '#e07a5f' for f in top15['Feature']]
    plt.barh(top15['Feature'], top15['Mean_Absolute_SHAP'], color=colors)
    plt.title(f'SHAP Global Feature Importance - {dataset_name.upper()}', fontsize=13, fontweight='bold')
    plt.xlabel('Mean |SHAP Value| (Impact on Model Decision)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"plots/{dataset_name}_shap_importance.png", dpi=300)
    plt.close()
    print(f"[+] Saved SHAP plot to plots/{dataset_name}_shap_importance.png")

if __name__ == '__main__':
    run_shap_analysis('knowledge_base/promise_knowledge_base.csv', 'promise')
