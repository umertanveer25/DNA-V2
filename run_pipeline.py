import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.dna_features import TrueGenomicFeatureExtractor
from src.models import get_paper_classifiers
from src.evaluation import PureGenomicEvaluator

def run_experiment_on_dataset(seq_csv_path, dataset_name, target_col='Dinucleotide_Target', num_splits=1):
    print(f"\n{'='*75}")
    print(f"[*] BENCHMARKING TRUE GENOMIC DNA SEQUENCES ON {dataset_name.upper()} (NO SMOTE)")
    print(f"{'='*75}")
    
    df = pd.read_csv(seq_csv_path)
    X_text = df['Requirement_Text'].tolist()
    y = df[target_col].values
    
    print(f"[+] Total samples: {len(X_text)}")
    print(f"[+] Target classes ({len(np.unique(y))}): {np.unique(y).tolist()}")
    
    # Feature Extraction
    print("\n[*] Extracting True Biological Genomic Features (1-mers, 2-mers, GC%, TF-IDF, SBERT)...")
    extractor = TrueGenomicFeatureExtractor(max_tfidf_features=50)
    X_fused = extractor.fit_transform(X_text)
    print(f"[+] Generated Feature Matrix: {X_fused.shape}")
    
    classifiers = get_paper_classifiers()
    evaluator = PureGenomicEvaluator(classifiers, random_state=42)
    
    # Benchmark
    res_df = evaluator.run_benchmark(X_fused, y, num_splits=num_splits)
    
    print(f"\n[+] BENCHMARK SUMMARY FOR {dataset_name.upper()} (NO SMOTE):")
    print(res_df.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    out_file = f"results/{dataset_name.lower()}_genomic_pure_benchmark.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\n[+] Successfully saved benchmark results to: {out_file}")
    return res_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='promise', choices=['promise', 'fnfc', 'all'])
    parser.add_argument('--splits', type=int, default=1)
    args = parser.parse_args()
    
    if args.dataset in ['promise', 'all']:
        run_experiment_on_dataset('knowledge_base/dynamic_promise_dinucleotides.csv', 'promise', num_splits=args.splits)
        
    if args.dataset in ['fnfc', 'all']:
        run_experiment_on_dataset('knowledge_base/dynamic_fnfc_dinucleotides.csv', 'fnfc', num_splits=args.splits)
