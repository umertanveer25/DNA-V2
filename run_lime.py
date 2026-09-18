import os
import sys
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dna_features import TrueGenomicFeatureExtractor

print(f"\n{'='*75}")
print("[*] GENERATING LIME LOCAL EXPLANATIONS FOR INDIVIDUAL REQUIREMENTS")
print(f"{'='*75}", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
texts = df['Requirement_Text'].tolist()
y = df['Dinucleotide_Target'].values

ext = TrueGenomicFeatureExtractor(max_tfidf_features=50)
X = ext.fit_transform(texts)

# Feature Names construction
monomers = ['A', 'T', 'C', 'G']
dimers = [b1 + b2 for b1 in monomers for b2 in monomers]
genomic_names = [f"k1_{m}" for m in monomers] + [f"k2_{d}" for d in dimers] + ['GC_Content', 'AT_GC_Ratio', 'Seq_Length_Log']
tfidf_names = [f"TFIDF_{t}" for t in ext.tfidf_vectorizer.get_feature_names_out()]
sbert_names = [f"SBERT_dim_{i}" for i in range(384)]
feature_names = genomic_names + tfidf_names + sbert_names

unique_classes = sorted(list(set(y)))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
y_idx = np.array([class_to_idx[c] for c in y])

X_tr, X_te, y_tr, y_te, texts_tr, texts_te = train_test_split(
    X, y_idx, texts, test_size=0.25, random_state=42, stratify=y_idx
)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_tr, y_tr)

# Initialize LIME Tabular Explainer
explainer = LimeTabularExplainer(
    training_data=X_tr,
    feature_names=feature_names,
    class_names=unique_classes,
    mode='classification'
)

# Explain 2 interesting requirements:
# 1. Performance / Operational (AG)
# 2. Security / Policy (AC)
test_indices = [0, 5]

os.makedirs('plots/lime_reports', exist_ok=True)
os.makedirs('results', exist_ok=True)

for sample_idx in test_indices:
    instance = X_te[sample_idx]
    true_label = unique_classes[y_te[sample_idx]]
    pred_label = unique_classes[clf.predict([instance])[0]]
    req_text = texts_te[sample_idx]
    
    print(f"\n[+] Explaining Instance #{sample_idx}:")
    print(f"    Requirement: \"{req_text[:85]}...\"")
    print(f"    True Label: {true_label} | Model Prediction: {pred_label}")
    
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=clf.predict_proba,
        num_features=8,
        top_labels=1
    )
    
    # Save HTML report
    out_html = f"plots/lime_reports/lime_explanation_sample_{sample_idx}_{pred_label}.html"
    exp.save_to_file(out_html)
    print(f"    Saved interactive LIME HTML to: {out_html}")
    
    # Save static figure for publication
    fig = exp.as_pyplot_figure(label=clf.predict([instance])[0])
    plt.title(f"LIME Local Explanation for Pred: {pred_label} (True: {true_label})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_png = f"plots/lime_reports/lime_sample_{sample_idx}_{pred_label}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"    Saved LIME figure to: {out_png}")

print("\n[+] LIME Explanations Generated Successfully!")
