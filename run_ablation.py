import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dna_features import TrueGenomicFeatureExtractor, MONOMERS, DIMERS
from src.preprocessor import TextPreprocessor

print(f"\n{'='*75}")
print("[*] STEP 2: EXECUTING RIGOROUS COMPONENT ABLATION STUDY")
print(f"{'='*75}", flush=True)

df = pd.read_csv('knowledge_base/dynamic_pexp_dinucleotides.csv')
texts = df['Requirement_Text'].tolist()
cleaned = [TextPreprocessor.clean_text(t) for t in texts]
y = df['Dinucleotide_Target'].values

# Feature Extractions
print("[1/4] Extracting TF-IDF & SBERT embeddings...", flush=True)
tfidf = TfidfVectorizer(max_features=50, stop_words='english', sublinear_tf=True).fit_transform(cleaned).toarray()
sbert = SentenceTransformer('all-MiniLM-L6-v2').encode(cleaned, show_progress_bar=False)

# Pure NLP Baseline (No DNA)
X_no_dna = np.hstack((tfidf, sbert * 1.5))

# Paper 1 Baseline DNA (Old 4-count monomer only)
print("[2/4] Extracting Paper 1 4-count DNA features...", flush=True)
ext_full = TrueGenomicFeatureExtractor(max_tfidf_features=50)
genomic_all = np.array([ext_full.text_to_genomic_features(t) for t in texts], dtype=np.float32)
paper1_dna_counts = genomic_all[:, :4] # Only the 4 monomers
X_paper1_dna = np.hstack((paper1_dna_counts * 2.0, tfidf, sbert * 1.5))

# Our True Genomic Sequences Alone (No NLP / SBERT)
print("[3/4] Isolating True Genomic Sequences alone (k-mers + ratios)...", flush=True)
X_genomic_only = genomic_all

# Our Full Fused Representation
print("[4/4] Constructing Full Fused Model...", flush=True)
X_full = np.hstack((genomic_all * 2.0, tfidf, sbert * 1.5))

configurations = {
    "1. Baseline NLP (TF-IDF + SBERT only - No DNA)": X_no_dna,
    "2. Paper 1 DNA Representation (4-count vector + NLP)": X_paper1_dna,
    "3. Our True Genomic Sequences Alone (k-mers + Ratios)": X_genomic_only,
    "4. Our Full Fused Model (True Genomic + NLP)": X_full
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ablation_results = []
for name, X_matrix in configurations.items():
    print(f"    Evaluating Configuration: {name[:45]}...", end=" ", flush=True)
    rf_accs, gb_accs = [], []
    for train_idx, test_idx in skf.split(X_matrix, y):
        # RF
        clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf_rf.fit(X_matrix[train_idx], y[train_idx])
        rf_accs.append(accuracy_score(y[test_idx], clf_rf.predict(X_matrix[test_idx])))
        # GB
        clf_gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        clf_gb.fit(X_matrix[train_idx], y[train_idx])
        gb_accs.append(accuracy_score(y[test_idx], clf_gb.predict(X_matrix[test_idx])))
        break
        
    mean_rf = round(np.mean(rf_accs) * 100, 2)
    mean_gb = round(np.mean(gb_accs) * 100, 2)
    print(f"RF Acc: {mean_rf}% | GB Acc: {mean_gb}%", flush=True)
    ablation_results.append({
        'Feature Configuration': name,
        'Feature Count': X_matrix.shape[1],
        'Random Forest Acc (%)': mean_rf,
        'Gradient Boosting Acc (%)': mean_gb
    })

ablation_df = pd.DataFrame(ablation_results)
print("\n" + "="*80)
print("🔬 ABLATION STUDY RESULTS (P-EXP DATASET):")
print("="*80)
print(ablation_df.to_string(index=False))

out_ablation = 'results/ablation_study_results.csv'
ablation_df.to_csv(out_ablation, index=False)
print(f"[+] Saved ablation study to: {out_ablation}")
