import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import os
import numpy as np
import pandas as pd
from src.dna_features import TrueGenomicFeatureExtractor

os.makedirs('results', exist_ok=True)
ext = TrueGenomicFeatureExtractor(max_tfidf_features=50)

# 1. PROMISE
print("[*] Extracting and Caching PROMISE features...", flush=True)
df_p = pd.read_csv('knowledge_base/dynamic_promise_dinucleotides.csv')
X_p = ext.fit_transform(df_p['Requirement_Text'].tolist())
y_p = df_p['Dinucleotide_Target'].values
np.save('results/X_promise.npy', X_p)
np.save('results/y_promise.npy', y_p)
print(f"[+] PROMISE Features cached successfully: {X_p.shape}", flush=True)

# 2. FNFC
print("[*] Extracting and Caching FNFC features...", flush=True)
df_f = pd.read_csv('knowledge_base/dynamic_fnfc_dinucleotides.csv')
X_f = ext.fit_transform(df_f['Requirement_Text'].tolist())
y_f = df_f['Dinucleotide_Target'].values
np.save('results/X_fnfc.npy', X_f)
np.save('results/y_fnfc.npy', y_f)
print(f"[+] FNFC Features cached successfully: {X_f.shape}", flush=True)
