import pandas as pd

for path in ['data/p_exp.csv', 'data/promise_nfr.csv', 'data/fnfc.csv', 'knowledge_base/dynamic_pexp_dinucleotides.csv']:
    try:
        df = pd.read_csv(path, encoding='latin1')
        print(f"=== {path} (shape={df.shape}) ===")
        print("Columns:", df.columns.tolist())
        for c in df.columns:
            u = df[c].dropna().unique()
            if len(u) < 25:
                print(f"  Col '{c}' ({len(u)} unique values):")
                print(df[c].value_counts().to_dict())
        print()
    except Exception as e:
        print(f"Error reading {path}: {e}")
