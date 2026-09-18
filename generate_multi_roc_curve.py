import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.dna_features import TrueGenomicFeatureExtractor

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print(f"\n{'='*80}")
print("📊 GENERATING MULTI-ALGORITHM ROC CURVES FOR PAPER 2 (DNA+ SEQUENCES)")
print(f"{'='*80}\n", flush=True)

# 1. Load Dataset (P-EXP: 969 Requirements with Dynamic Dinucleotides)
data_path = 'knowledge_base/dynamic_pexp_dinucleotides.csv'
print(f"[*] Loading dataset from: {data_path}", flush=True)
df = pd.read_csv(data_path)

# Filter out extremely rare classes (< 4 samples) so train/test split has valid stratification
class_counts = df['Dinucleotide_Target'].value_counts()
valid_classes = class_counts[class_counts >= 4].index.tolist()
print(f"[+] Total raw samples: {len(df)}")
print(f"[+] Active classes with sufficient support (>= 4 samples): {valid_classes}")

df_filtered = df[df['Dinucleotide_Target'].isin(valid_classes)].copy().reset_index(drop=True)
print(f"[+] Filtered dataset size: {len(df_filtered)} requirements across {len(valid_classes)} classes", flush=True)

X_text = df_filtered['Requirement_Text'].tolist()
y = df_filtered['Dinucleotide_Target'].values

# 2. Extract Features
print("[*] Extracting True Genomic Features (k-mer + physicochemical + TF-IDF + SBERT)...", flush=True)
extractor = TrueGenomicFeatureExtractor(max_tfidf_features=50)
X = extractor.fit_transform(X_text)
print(f"[+] Feature matrix shape: {X.shape}", flush=True)

# 3. Stratified Train-Test Split (75% Train, 25% Test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"[+] Train set: {len(y_train)} samples | Test set: {len(y_test)} samples", flush=True)

# Binarize labels for multiclass ROC
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)

# 4. Define Evaluated Classifiers
models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Linear SVM (Calibrated)': CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced', random_state=42, max_iter=2000)),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Color palette for publication plots
colors = {
    'Gradient Boosting': '#1b9e77',    # Teal Green
    'Random Forest': '#d95f02',        # Vermilion / Orange
    'KNN (k=3)': '#7570b3',            # Purple
    'Logistic Regression': '#e7298a',  # Magenta / Deep Rose
    'Linear SVM (Calibrated)': '#e6ab02', # Amber Gold
    'Decision Tree': '#66a61e',        # Olive Green
    'AdaBoost': '#a6761d',             # Brownish Gold
    'Gaussian Naive Bayes': '#666666'  # Dark Slate
}

roc_results = {}
all_fpr = np.linspace(0, 1, 200)

print("\n[*] Training algorithms and computing ROC metrics...", flush=True)

for name, clf in models.items():
    print(f"    Training {name:25s}...", end=" ", flush=True)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
    # Compute Macro-average ROC curve
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    macro_auc = auc(all_fpr, mean_tpr)
    
    # Compute Micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    
    roc_results[name] = {
        'clf': clf,
        'macro_fpr': all_fpr,
        'macro_tpr': mean_tpr,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'class_fpr': fpr_dict,
        'class_tpr': tpr_dict,
        'class_auc': roc_auc_dict,
        'classes': classes
    }
    print(f"Done! Macro AUC: {macro_auc:.4f} | Micro AUC: {micro_auc:.4f}", flush=True)

# Save ROC summary metrics to CSV
roc_summary_df = pd.DataFrame([
    {
        'Algorithm': name,
        'Macro-Average ROC AUC': round(data['macro_auc'], 4),
        'Micro-Average ROC AUC': round(data['micro_auc'], 4)
    }
    for name, data in roc_results.items()
]).sort_values(by='Macro-Average ROC AUC', ascending=False)

roc_summary_csv = 'results/multi_algorithm_roc_summary.csv'
roc_summary_df.to_csv(roc_summary_csv, index=False)
print(f"\n[+] Saved ROC Summary Metrics to: {roc_summary_csv}")
print(roc_summary_df.to_string(index=False))

# ==============================================================================
# 5. FIGURE 1: MULTI-ALGORITHM ROC CURVE (STANDALONE PUBLICATION FIGURE)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Sort models by Macro AUC descending for clean legend order
sorted_models = sorted(roc_results.items(), key=lambda x: x[1]['macro_auc'], reverse=True)

for name, data in sorted_models:
    ax.plot(
        data['macro_fpr'],
        data['macro_tpr'],
        color=colors.get(name, '#333333'),
        lw=2.5,
        label=f"{name:23s} (AUC = {data['macro_auc']:.3f})"
    )

# Plot Random Baseline
ax.plot([0, 1], [0, 1], color='#999999', lw=1.8, linestyle='--', label='Random Chance baseline (AUC = 0.500)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Multi-Algorithm ROC Comparison (DNA+ Genomic Sequences)\nMacro-Averaged Receiver Operating Characteristic', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax.tick_params(axis='both', which='major', labelsize=11)

# Inset zoom-in for top-left corner [0.0, 0.25] FPR, [0.75, 1.01] TPR
ax_inset = ax.inset_axes([0.48, 0.18, 0.42, 0.38])
for name, data in sorted_models:
    ax_inset.plot(data['macro_fpr'], data['macro_tpr'], color=colors.get(name, '#333333'), lw=2.2)
ax_inset.plot([0, 1], [0, 1], color='#999999', lw=1.5, linestyle='--')
ax_inset.set_xlim([0.0, 0.25])
ax_inset.set_ylim([0.75, 1.01])
ax_inset.set_title('Zoom: Low FPR Regime', fontsize=9, fontweight='bold')
ax_inset.tick_params(axis='both', which='major', labelsize=8)
ax_inset.grid(True, linestyle=':', alpha=0.6)
ax.indicate_inset_zoom(ax_inset, edgecolor='#555555', alpha=0.5)

plt.tight_layout()
out_single_roc = 'plots/multi_algorithm_roc_curve.png'
plt.savefig(out_single_roc, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[+] Saved Multi-Algorithm ROC plot to: {out_single_roc}")

# ==============================================================================
# 6. FIGURE 2: TWO-PANEL PUBLICATION MASTER ROC FIGURE
#    Panel A: Multi-Algorithm Comparison
#    Panel B: Per-Class Dinucleotide Breakdown for Champion Model
# ==============================================================================
champion_name = sorted_models[0][0]
champ_data = roc_results[champion_name]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# --- Panel A: Multi-Algorithm ---
for name, data in sorted_models:
    ax1.plot(
        data['macro_fpr'],
        data['macro_tpr'],
        color=colors.get(name, '#333333'),
        lw=2.3,
        label=f"{name} (AUC = {data['macro_auc']:.3f})"
    )
ax1.plot([0, 1], [0, 1], color='#999999', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax1.set_title('(A) Multi-Algorithm Macro ROC Comparison', fontsize=14, fontweight='bold', pad=12)
ax1.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax1.tick_params(axis='both', labelsize=10)

# --- Panel B: Per-Class One-vs-Rest for Champion Model ---
class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
sorted_classes_by_auc = sorted(
    range(n_classes),
    key=lambda idx: champ_data['class_auc'][idx],
    reverse=True
)

for idx in sorted_classes_by_auc:
    c_name = champ_data['classes'][idx]
    c_auc = champ_data['class_auc'][idx]
    ax2.plot(
        champ_data['class_fpr'][idx],
        champ_data['class_tpr'][idx],
        lw=2.0,
        color=class_colors[idx],
        label=f"Class {c_name} (AUC = {c_auc:.3f})"
    )

# Add Champion Micro and Macro lines to Panel B
ax2.plot(champ_data['macro_fpr'], champ_data['macro_tpr'], color='#111111', lw=2.5, linestyle='-', label=f"Macro-Average (AUC = {champ_data['macro_auc']:.3f})")
ax2.plot([0, 1], [0, 1], color='#999999', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')

ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax2.set_title(f'(B) Per-Class Dinucleotide ROC for Champion: {champion_name}', fontsize=14, fontweight='bold', pad=12)
ax2.legend(loc='lower right', fontsize=9.5, frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc')
ax2.tick_params(axis='both', labelsize=10)

plt.suptitle('Receiver Operating Characteristic (ROC) Evaluation across Algorithms and Dinucleotide Targets', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dual_roc = 'plots/roc_comprehensive_two_panel.png'
plt.savefig(out_dual_roc, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved Comprehensive Two-Panel ROC plot to: {out_dual_roc}")

print("\n" + "="*80)
print("✅ ALL ROC CURVES GENERATED SUCCESSFULLY!")
print("="*80)
