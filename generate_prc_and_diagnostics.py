import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.dna_features import TrueGenomicFeatureExtractor

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print(f"\n{'='*80}")
print("📈 GENERATING PRECISION-RECALL (PRC) & CONFUSION MATRIX DIAGNOSTICS")
print(f"{'='*80}\n", flush=True)

# 1. Load Data & Cache Features
cache_x = 'results/X_pexp.npy'
cache_y = 'results/y_pexp.npy'

if os.path.exists(cache_x) and os.path.exists(cache_y):
    print(f"[*] Loading cached features from {cache_x}...", flush=True)
    X = np.load(cache_x)
    y = np.load(cache_y)
else:
    data_path = 'knowledge_base/dynamic_pexp_dinucleotides.csv'
    print(f"[*] Loading raw dataset from: {data_path}", flush=True)
    df = pd.read_csv(data_path)
    
    class_counts = df['Dinucleotide_Target'].value_counts()
    valid_classes = class_counts[class_counts >= 4].index.tolist()
    df_filtered = df[df['Dinucleotide_Target'].isin(valid_classes)].copy().reset_index(drop=True)
    
    X_text = df_filtered['Requirement_Text'].tolist()
    y = df_filtered['Dinucleotide_Target'].values
    
    print("[*] Extracting True Genomic Features...", flush=True)
    extractor = TrueGenomicFeatureExtractor(max_tfidf_features=50)
    X = extractor.fit_transform(X_text)
    
    np.save(cache_x, X)
    np.save(cache_y, y)
    print(f"[+] Cached features to {cache_x} and {cache_y}", flush=True)

print(f"[+] Dataset shape: {X.shape}, Total samples: {len(y)}", flush=True)

# 2. Stratified Train-Test Split (75% Train, 25% Test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

classes = np.unique(y_train)
n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)

# 3. Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Linear SVM (Calibrated)': CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced', random_state=42, max_iter=2000))
}

colors = {
    'Random Forest': '#d95f02',
    'Gradient Boosting': '#1b9e77',
    'Logistic Regression': '#e7298a',
    'KNN (k=3)': '#7570b3',
    'Decision Tree': '#66a61e',
    'AdaBoost': '#a6761d',
    'Gaussian Naive Bayes': '#666666',
    'Linear SVM (Calibrated)': '#e6ab02'
}

prc_results = {}
recall_grid = np.linspace(0, 1, 200)

print("\n[*] Training classifiers & computing Precision-Recall curves...", flush=True)

for name, clf in models.items():
    print(f"    Evaluating {name:25s}...", end=" ", flush=True)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    
    # Per-class PR curves and Average Precision (AP)
    precision_dict = dict()
    recall_dict = dict()
    ap_dict = dict()
    
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        precision_dict[i] = p
        recall_dict[i] = r
        ap_dict[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
        
    # Macro Average Precision
    macro_ap = np.mean(list(ap_dict.values()))
    
    # Micro Average Precision
    micro_ap = average_precision_score(y_test_bin, y_score, average='micro')
    
    # Macro PR curve interpolation
    interp_precision = np.zeros_like(recall_grid)
    for i in range(n_classes):
        # r is descending in scikit-learn, flip for interpolation
        r_rev = recall_dict[i][::-1]
        p_rev = precision_dict[i][::-1]
        interp_precision += np.interp(recall_grid, r_rev, p_rev)
    interp_precision /= n_classes
    
    prc_results[name] = {
        'clf': clf,
        'y_pred': y_pred,
        'macro_ap': macro_ap,
        'micro_ap': micro_ap,
        'macro_precision': interp_precision,
        'macro_recall': recall_grid,
        'class_precision': precision_dict,
        'class_recall': recall_dict,
        'class_ap': ap_dict
    }
    print(f"Done! Macro AP (PR-AUC): {macro_ap:.4f} | Micro AP: {micro_ap:.4f}", flush=True)

# 4. Save PR-AUC Summary Table
prc_summary_df = pd.DataFrame([
    {
        'Algorithm': name,
        'Macro PR-AUC (AP)': round(data['macro_ap'], 4),
        'Micro PR-AUC (AP)': round(data['micro_ap'], 4)
    }
    for name, data in prc_results.items()
]).sort_values(by='Macro PR-AUC (AP)', ascending=False)

prc_csv = 'results/multi_algorithm_prc_summary.csv'
prc_summary_df.to_csv(prc_csv, index=False)
print(f"\n[+] Saved PR-AUC Summary Table to: {prc_csv}")
print(prc_summary_df.to_string(index=False))

# ==============================================================================
# 5. FIGURE 1: MULTI-ALGORITHM PRECISION-RECALL CURVE (STANDALONE)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

sorted_models = sorted(prc_results.items(), key=lambda x: x[1]['macro_ap'], reverse=True)

for name, data in sorted_models:
    ax.plot(
        data['macro_recall'],
        data['macro_precision'],
        color=colors.get(name, '#333333'),
        lw=2.5,
        label=f"{name:23s} (PR-AUC = {data['macro_ap']:.3f})"
    )

# Baseline chance level for PR curve (random baseline = 1 / num_classes)
chance_level = 1.0 / n_classes
ax.axhline(chance_level, color='#999999', lw=1.8, linestyle='--', label=f'Random Chance baseline (AP = {chance_level:.3f})')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall (True Positive Rate)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Multi-Algorithm Precision-Recall Comparison (DNA+ Genomic Sequences)\nMacro-Averaged Precision-Recall Characteristic', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='lower left', fontsize=11, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
out_single_prc = 'plots/multi_algorithm_prc_curve.png'
plt.savefig(out_single_prc, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Multi-Algorithm PRC plot to: {out_single_prc}")

# ==============================================================================
# 6. FIGURE 2: TWO-PANEL COMPREHENSIVE PRC FIGURE
#    Panel A: Multi-Algorithm Comparison
#    Panel B: Per-Class Dinucleotide Breakdown for Champion (Random Forest)
# ==============================================================================
champion_name = sorted_models[0][0]
champ_data = prc_results[champion_name]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# Panel A
for name, data in sorted_models:
    ax1.plot(
        data['macro_recall'],
        data['macro_precision'],
        color=colors.get(name, '#333333'),
        lw=2.3,
        label=f"{name} (PR-AUC = {data['macro_ap']:.3f})"
    )
ax1.axhline(chance_level, color='#999999', lw=1.5, linestyle='--', label=f'Random Chance (AP = {chance_level:.3f})')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('(A) Multi-Algorithm Macro Precision-Recall Comparison', fontsize=14, fontweight='bold', pad=12)
ax1.legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax1.tick_params(axis='both', labelsize=10)

# Panel B: Per-Class for Champion
class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
sorted_classes_by_ap = sorted(
    range(n_classes),
    key=lambda idx: champ_data['class_ap'][idx],
    reverse=True
)

for idx in sorted_classes_by_ap:
    c_name = classes[idx]
    c_ap = champ_data['class_ap'][idx]
    ax2.plot(
        champ_data['class_recall'][idx],
        champ_data['class_precision'][idx],
        lw=2.0,
        color=class_colors[idx],
        label=f"Class {c_name} (AP = {c_ap:.3f})"
    )

ax2.plot(champ_data['macro_recall'], champ_data['macro_precision'], color='#111111', lw=2.5, linestyle='-', label=f"Macro-Average (AP = {champ_data['macro_ap']:.3f})")
ax2.axhline(chance_level, color='#999999', lw=1.5, linestyle='--', label=f'Random Chance (AP = {chance_level:.3f})')
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title(f'(B) Per-Class Dinucleotide PR Curves for Champion: {champion_name}', fontsize=14, fontweight='bold', pad=12)
ax2.legend(loc='lower left', fontsize=9.5, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax2.tick_params(axis='both', labelsize=10)

plt.suptitle('Precision-Recall (PRC) Evaluation across Algorithms and Dinucleotide Semantic Targets', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dual_prc = 'plots/prc_comprehensive_two_panel.png'
plt.savefig(out_dual_prc, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Comprehensive Two-Panel PRC plot to: {out_dual_prc}")

# ==============================================================================
# 7. FIGURE 3: NORMALIZED CONFUSION MATRIX HEATMAP (CHAMPION MODEL)
# ==============================================================================
cm = confusion_matrix(y_test, champ_data['y_pred'], labels=classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    cbar=True,
    square=True,
    linewidths=1.0,
    linecolor='#eeeeee',
    annot_kws={"size": 11, "weight": "bold"},
    ax=ax
)

ax.set_xlabel('Predicted Dinucleotide Target', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Ground Truth Dinucleotide Target', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title(f'Normalized Confusion Matrix: {champion_name}\nDynamic Syntactic-Semantic Dinucleotide Mapping', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
out_cm = 'plots/champion_confusion_matrix_heatmap.png'
plt.savefig(out_cm, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Normalized Confusion Matrix Heatmap to: {out_cm}")

print("\n" + "="*80)
print("✅ ALL PRC & DIAGNOSTIC CURVES GENERATED SUCCESSFULLY!")
print("="*80)
