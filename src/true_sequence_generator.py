import os
import sys
import re
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Token-to-Base Mapping Vocabulary
BASE_MAP = {
    'A': { # Imperative Action, Obligation, Functional Capability
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check', 'convert',
        'format', 'track', 'search', 'print', 'log', 'render', 'filter', 'sort'
    },
    'T': { # Usability, UI, Interaction, Visual Aesthetics, Accessibility
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'prompt', 'ergonomic', 'understandable',
        'legible', 'viewer', 'viewing', 'distance', 'human'
    },
    'C': { # Security, Protection, Confidentiality, Compliance, Legal
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted',
        'safe', 'safely', 'hazard', 'standards', 'w3c', 'regulations'
    },
    'G': { # Performance, Temporal Constraints, Capacity, Availability, Reliability, Platform
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

# Invert dictionary for fast token lookup
WORD_TO_BASE = {}
for base, words in BASE_MAP.items():
    for w in words:
        WORD_TO_BASE[w] = base

def sentence_to_dna_sequence(text, keep_spacers=False):
    tokens = re.findall(r'\b[a-z0-9\-_]{2,}\b', str(text).lower())
    seq_bases = []
    
    for t in tokens:
        if t in WORD_TO_BASE:
            seq_bases.append(WORD_TO_BASE[t])
        elif re.match(r'^\d+$', t):
            # Quantitative number constraint -> maps to G (metric)
            seq_bases.append('G')
        elif keep_spacers:
            seq_bases.append('N')
            
    # If no keywords matched, default to minimal functional motif 'A'
    if not seq_bases:
        seq_bases = ['A']
        
    return "".join(seq_bases)

def extract_kmer_features(dna_sequence):
    # 1-mers (A, T, C, G)
    monomers = ['A', 'T', 'C', 'G']
    c_mono = Counter(dna_sequence)
    total_len = len(dna_sequence) if len(dna_sequence) > 0 else 1
    
    mono_freqs = {f"k1_{m}": c_mono[m] / total_len for m in monomers}
    
    # 2-mers (all 16 dinucleotides)
    dimers = [b1 + b2 for b1 in monomers for b2 in monomers]
    dimer_counts = Counter([dna_sequence[i:i+2] for i in range(len(dna_sequence)-1)])
    total_dimers = max(1, len(dna_sequence) - 1)
    dimer_freqs = {f"k2_{d}": dimer_counts[d] / total_dimers for d in dimers}
    
    # GC Content
    gc_content = (c_mono['G'] + c_mono['C']) / total_len
    
    # Ratio AT / GC
    at_gc_ratio = (c_mono['A'] + c_mono['T']) / (c_mono['G'] + c_mono['C'] + 1e-5)
    
    features = {}
    features.update(mono_freqs)
    features.update(dimer_freqs)
    features['gc_content'] = gc_content
    features['at_gc_ratio'] = at_gc_ratio
    features['seq_length'] = float(len(dna_sequence))
    
    return features

def process_true_dna_sequences(input_csv, text_col, class_col, dataset_name, out_csv):
    print(f"\n[+] Synthesizing True Biological DNA Sequences for {dataset_name} ({input_csv})...")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')
        
    records = []
    for idx, row in df.iterrows():
        text = str(row[text_col])
        cls = str(row[class_col]).strip()
        dna_seq = sentence_to_dna_sequence(text, keep_spacers=False)
        kmers = extract_kmer_features(dna_seq)
        
        row_dict = {
            'Req_ID': f"{dataset_name}_{idx+1}",
            'Dataset': dataset_name,
            'Original_Class': cls,
            'Requirement_Text': text,
            'DNA_Sequence': dna_seq,
            'Sequence_Length': len(dna_seq),
            'GC_Content': round(kmers['gc_content'], 4),
            'AT_GC_Ratio': round(kmers['at_gc_ratio'], 4)
        }
        # Add 16 2-mer counts
        for k, v in kmers.items():
            if k.startswith('k2_') or k.startswith('k1_'):
                row_dict[k] = round(v, 4)
                
        records.append(row_dict)
        
    seq_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    seq_df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"    Saved {len(seq_df)} synthetic DNA sequences to: {out_csv}")
    print(f"    Average DNA Strand Length: {seq_df['Sequence_Length'].mean():.2f} base pairs")
    return seq_df

if __name__ == '__main__':
    # 1. PROMISE
    p_in = "B:/Promise/Paper2/data/promise_nfr.csv"
    p_out = "B:/Promise/Paper2/knowledge_base/promise_genomic_sequences.csv"
    process_true_dna_sequences(p_in, 'RequirementText', 'class', 'PROMISE', p_out)
    
    # 2. FNFC
    f_in = "B:/Promise/Paper2/data/fnfc.csv"
    f_out = "B:/Promise/Paper2/knowledge_base/fnfc_genomic_sequences.csv"
    process_true_dna_sequences(f_in, 'text', 'class', 'FNFC', f_out)
