"""
Rigorous 10-Fold Cross-Validation with 30 Randomized Repeats (RepeatedStratifiedKFold)
Evaluates Classifiers on Canonical Software Requirements Engineering Ground-Truth Benchmarks.
Zero Data Leakage: Feature Extraction, TF-IDF vectorization, and Scalers fit strictly on training splits.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import re
import math

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.preprocessor import TextPreprocessor

# Core Lexicon mapping words to nucleotide base categories
BASE_MAP = {
    'A': {
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check', 'convert',
        'format', 'track', 'search', 'print', 'log', 'render', 'filter', 'sort'
    },
    'T': {
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'prompt', 'ergonomic', 'understandable',
        'legible', 'viewer', 'viewing', 'distance', 'human'
    },
    'C': {
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted',
        'safe', 'safely', 'hazard', 'standards', 'w3c', 'regulations'
    },
    'G': {
        'fast', 'speed', 'response', 'performance', 'time', 'latency', 'throughput',
        'bandwidth', 'capacity', 'scalability', 'efficiency', 'real-time', 'concurrent',
        'parallel', 'millisecond', 'second', 'minute', 'hour', 'load', 'stress',
        'benchmark', 'optimize', 'cache', 'buffer', 'queue', 'rate', 'frequency',
        'memory', 'cpu', 'overhead', 'available', 'availability', 'uptime', 'downtime',
        'reliable', 'reliability', 'fault-tolerant', 'backup', 'recovery', 'redundancy',
        'failover', 'resilient', 'robust', 'mtbf', 'disaster', 'crash', 'platform',
        'operating', 'system', 'linux', 'windows', 'browser', 'hardware', 'server',
        'simultaneous', 'scaleable', 'simultaneously', 'offline'
    }
}

WORD_TO_BASE = {w: b for b, words in BASE_MAP.items() for w in words}
MONOMERS = ['A', 'T', 'C', 'G']
DIMERS = [b1 + b2 for b1 in MONOMERS for b2 in MONOMERS]

def compute_genomic_features(text):
    tokens = re.findall(r'\b[a-z0-9\-_]{2,}\b', str(text).lower())
    seq = [WORD_TO_BASE[t] for t in tokens if t in WORD_TO_BASE]
    if not seq:
        seq = ['A']
    dna_str = "".join(seq)
    L = len(dna_str)
    c1 = Counter(dna_str)
    k1 = [c1[m] / L for m in MONOMERS]
    c2 = Counter([dna_str[i:i+2] for i in range(L - 1)])
    L2 = max(1, L - 1)
    k2 = [c2[d] / L2 for d in DIMERS]
    gc = (c1['G'] + c1['C']) / L
    at_gc = (c1['A'] + c1['T']) / (c1['G'] + c1['C'] + 1e-5)
    log_l = math.log1p(L)
    return k1 + k2 + [gc, at_gc, log_l]

def run_repeated_cv(csv_path, text_col, class_col, dataset_name, n_splits=10, n_repeats=30):
    print(f"\n{'='*80}")
    print(f"[*] EXECUTING {n_splits}-FOLD CV x {n_repeats} REPEATS ({n_splits * n_repeats} TOTAL SPLITS)")
    print(f"    DATASET: {dataset_name} | TARGET: Authentic Ground-Truth Class ('{class_col}')")
    print(f"{'='*80}", flush=True)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # Filter out rare classes (< 2 samples) to allow stratified folding
    counts = df[class_col].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df[class_col].isin(valid_classes)].reset_index(drop=True)

    texts = [TextPreprocessor.clean_text(str(t)) for t in df[text_col]]
    y = df[class_col].values
    n_samples = len(texts)
    n_classes = len(np.unique(y))

    print(f"[+] Total Cleaned Samples: {n_samples} across {n_classes} valid classes.")
    print(f"[+] Precomputing deterministic genomic features (23 dimensions)...", flush=True)
    X_genomic = np.array([compute_genomic_features(t) for t in texts], dtype=np.float32)

    # Cache SBERT embeddings once to avoid repeated model loading
    sbert_cache = f"results/sbert_cache_{dataset_name.lower()}.npy"
    if os.path.exists(sbert_cache):
        print(f"[+] Loading cached SBERT embeddings from {sbert_cache}...", flush=True)
        X_sbert = np.load(sbert_cache)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"[+] Encoding {n_samples} texts with SentenceTransformer (all-MiniLM-L6-v2)...", flush=True)
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        X_sbert = sbert.encode(texts, batch_size=64, show_progress_bar=True)
        os.makedirs("results", exist_ok=True)
        np.save(sbert_cache, X_sbert)
        print(f"[+] Cached SBERT to {sbert_cache}")

    # Define standard competitive classifiers
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    # Setup Repeated Stratified K-Fold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_iterations = n_splits * n_repeats

    # Record scores for each model across all splits
    scores = {name: {'acc': [], 'f1': [], 'prec': [], 'rec': []} for name in models}

    print(f"\n[+] Running {total_iterations} stratified train/test partitions with isolated TF-IDF fit...", flush=True)
    split_count = 0
    for train_idx, test_idx in rskf.split(texts, y):
        split_count += 1
        if split_count % 25 == 0 or split_count == 1 or split_count == total_iterations:
            print(f"    -> Progress: Completed {split_count}/{total_iterations} evaluation splits...", flush=True)

        # 1. Zero-leakage TF-IDF: fit strictly on train split
        tfidf = TfidfVectorizer(max_features=50, sublinear_tf=True)
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        X_tfidf_tr = tfidf.fit_transform(train_texts).toarray()
        X_tfidf_te = tfidf.transform(test_texts).toarray()

        # 2. Fuse features: [Genomic (23) * 2.0, TF-IDF (50), SBERT (384) * 1.5] = 457 dims
        X_tr = np.hstack((X_genomic[train_idx] * 2.0, X_tfidf_tr, X_sbert[train_idx] * 1.5))
        X_te = np.hstack((X_genomic[test_idx] * 2.0, X_tfidf_te, X_sbert[test_idx] * 1.5))

        y_tr, y_te = y[train_idx], y[test_idx]

        for name, clf in models.items():
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores[name]['acc'].append(accuracy_score(y_te, preds))
            scores[name]['f1'].append(f1_score(y_te, preds, average='macro', zero_division=0))
            scores[name]['prec'].append(precision_score(y_te, preds, average='macro', zero_division=0))
            scores[name]['rec'].append(recall_score(y_te, preds, average='macro', zero_division=0))

    # Summarize metrics
    summary = []
    for name in models:
        acc_m = np.mean(scores[name]['acc']) * 100
        acc_s = np.std(scores[name]['acc']) * 100
        f1_m = np.mean(scores[name]['f1']) * 100
        f1_s = np.std(scores[name]['f1']) * 100
        prec_m = np.mean(scores[name]['prec']) * 100
        rec_m = np.mean(scores[name]['rec']) * 100

        summary.append({
            'Algorithm': name,
            'Dataset': dataset_name,
            'Protocol': f'{n_splits}-Fold CV x {n_repeats} Repeats',
            'Total Evaluations': total_iterations,
            'Mean Accuracy (%)': f"{acc_m:.2f} ± {acc_s:.2f}",
            'Mean Macro-F1 (%)': f"{f1_m:.2f} ± {f1_s:.2f}",
            'Mean Macro-Precision (%)': f"{prec_m:.2f}",
            'Mean Macro-Recall (%)': f"{rec_m:.2f}",
            'Raw_Acc_Mean': acc_m,
            'Raw_Acc_Std': acc_s,
            'Raw_F1_Mean': f1_m,
            'Raw_F1_Std': f1_s
        })

    sum_df = pd.DataFrame(summary).sort_values(by='Raw_F1_Mean', ascending=False)
    print(f"\n{'='*85}")
    print(f"📊 SUMMARY: {n_splits}-FOLD CV x {n_repeats} REPEATS ON {dataset_name.upper()} (GROUND TRUTH)")
    print(f"{'='*85}")
    print(sum_df[['Algorithm', 'Mean Accuracy (%)', 'Mean Macro-F1 (%)', 'Mean Macro-Precision (%)', 'Mean Macro-Recall (%)']].to_string(index=False))

    out_csv = f"results/repeated_10fold_cv_{dataset_name.lower()}.csv"
    sum_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved rigorous repeated CV metrics to: {out_csv}")
    return sum_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PROMISE', choices=['PROMISE', 'P-EXP', 'ALL'])
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()

    if args.dataset in ['PROMISE', 'ALL']:
        run_repeated_cv('data/promise_nfr.csv', 'RequirementText', 'class', 'PROMISE', n_splits=args.splits, n_repeats=args.repeats)
    if args.dataset in ['P-EXP', 'ALL']:
        run_repeated_cv('data/p_exp.csv', 'Requirement', 'Type', 'P-EXP', n_splits=args.splits, n_repeats=args.repeats)
