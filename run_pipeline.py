import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from src.dna_features import EnhancedDNAFeatureExtractor
from src.models import get_paper_classifiers
from src.evaluation import RobustEvaluator

def run_experiment_on_dataset(kb_csv_path, dataset_name, num_splits=5):
    print(f"\n{'='*70}")
    print(f"🧬 BENCHMARKING ON {dataset_name.upper()} DATASET")
    print(f"{'='*70}")
    
    df = pd.read_csv(kb_csv_path)
    X_text = df['Requirement_Text'].tolist()
    y = df['Assigned_DNA'].values
    
    print(f"[*] Dataset samples: {len(X_text)}")
    print("[*] Target DNA distribution:")
    print(pd.Series(y).value_counts())
    
    # Feature Extraction
    print("\n[*] Extracting Multimodal DNA Features (Monomers, Dimers, Ratios, TF-IDF, SBERT)...")
    extractor = EnhancedDNAFeatureExtractor(max_tfidf_features=50)
    X_fused = extractor.fit_transform(X_text)
    print(f"[+] Total Feature Vector Dimension: {X_fused.shape[1]}")
    
    # Classifiers
    classifiers = get_paper_classifiers()
    evaluator = RobustEvaluator(classifiers, random_state=42)
    
    # 1. Run without SMOTE
    print("\n--- PHASE 1: Baseline Evaluation (Without SMOTE) ---")
    res_nosmote = evaluator.run_benchmark(X_fused, y, num_splits=num_splits, use_smote=False)
    
    # 2. Run with SMOTE
    print("\n--- PHASE 2: Enhanced DNA+ Evaluation (With SMOTE) ---")
    res_smote = evaluator.run_benchmark(X_fused, y, num_splits=num_splits, use_smote=True)
    
    # Merge and compare
    merged = pd.merge(
        res_nosmote[['Algorithm', 'Accuracy (%)', 'Macro F1 (%)']],
        res_smote[['Algorithm', 'Accuracy (%)', 'Macro F1 (%)']],
        on='Algorithm',
        suffixes=(' (Original)', ' (DNA+ & SMOTE)')
    )
    merged['Acc Gain (%)'] = (merged['Accuracy (%) (DNA+ & SMOTE)'] - merged['Accuracy (%) (Original)']).round(2)
    merged['F1 Gain (%)'] = (merged['Macro F1 (%) (DNA+ & SMOTE)'] - merged['Macro F1 (%) (Original)']).round(2)
    
    print(f"\n🏆 PERFORMANCE COMPARISON ON {dataset_name.upper()}:")
    print(merged.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    out_path = f"results/{dataset_name.lower()}_benchmark_comparison.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n[+] Saved results to {out_path}")
    
    return merged, X_fused, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['promise', 'fnfc', 'all'])
    parser.add_argument('--splits', type=int, default=3, help="Number of outer CV iterations")
    args = parser.parse_args()
    
    if args.dataset in ['promise', 'all']:
        run_experiment_on_dataset('knowledge_base/promise_knowledge_base.csv', 'promise', num_splits=args.splits)
        
    if args.dataset in ['fnfc', 'all']:
        run_experiment_on_dataset('knowledge_base/fnfc_knowledge_base.csv', 'fnfc', num_splits=args.splits)
