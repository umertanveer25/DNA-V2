import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dna_features import TrueGenomicFeatureExtractor

print(f"\n{'='*75}")
print("[*] STEP 1: EXECUTING SHAP TREE-EXPLAINER ANALYSIS")
print(f"{'='*75}", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
X_text = df['Requirement_Text'].tolist()
y = df['Dinucleotide_Target'].values

ext = TrueGenomicFeatureExtractor(max_tfidf_features=50)
X = ext.fit_transform(X_text)

# Feature Names construction
monomers = ['A', 'T', 'C', 'G']
dimers = [b1 + b2 for b1 in monomers for b2 in monomers]
genomic_names = [f"k1_{m}" for m in monomers] + [f"k2_{d}" for d in dimers] + ['GC_Content', 'AT_GC_Ratio', 'Seq_Length_Log']
tfidf_names = [f"TFIDF_{t}" for t in ext.tfidf_vectorizer.get_feature_names_out()]
sbert_names = [f"SBERT_dim_{i}" for i in range(384)]
feature_names = genomic_names + tfidf_names + sbert_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_tr, y_tr)

print("[+] Model fitted, calculating SHAP TreeExplainer values...", flush=True)
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_te)

# Average across classes
if isinstance(shap_vals, list):
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
elif len(shap_vals.shape) == 3:
    mean_shap = np.abs(shap_vals).mean(axis=(0, 2))
else:
    mean_shap = np.abs(shap_vals).mean(axis=0)

shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP_Impact': mean_shap
}).sort_values(by='Mean_SHAP_Impact', ascending=False)

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

out_shap_csv = 'results/shap_feature_importance.csv'
shap_df.to_csv(out_shap_csv, index=False)
print(f"[+] Saved SHAP values to: {out_shap_csv}")

print("\n🏆 Top 15 Most Influential Features Driving Predictions:")
print(shap_df.head(15).to_string(index=False))

# Plot Top 15 Features
top15 = shap_df.head(15).iloc[::-1]
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 7))

colors = ['#e76f51' if any(f.startswith(p) for p in ['k1_', 'k2_', 'GC_', 'AT_', 'Seq_']) else '#2a9d8f' for f in top15['Feature']]
ax.barh(top15['Feature'], top15['Mean_SHAP_Impact'], color=colors)
ax.set_xlabel('Mean |SHAP Value| (Impact on Model Classification)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Global Interpretability: Top 15 Discriminative Features (Orange = Genomic)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
out_shap_png = 'plots/shap_feature_importance.png'
plt.savefig(out_shap_png, dpi=300)
plt.close()
print(f"[+] Saved SHAP plot to: {out_shap_png}")
