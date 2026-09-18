import os
import sys
import re
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Comprehensive Semantic Lexicons for Syntactic Roles
LEXICON = {
    'A': { # Functional Action & Obligation
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check', 'convert',
        'format', 'track', 'search', 'print', 'log', 'render', 'filter', 'sort'
    },
    'T': { # Usability, Ergonomics, Human-Interaction & Visual Aesthetics
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'prompt', 'ergonomic', 'understandable',
        'legible', 'viewer', 'viewing', 'distance', 'display size', 'human'
    },
    'C': { # Security, Integrity, Privacy, Legal & Policy Compliance
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted',
        'safe', 'safely', 'hazard', 'damage', 'precaution', 'standards', 'w3c', 'regulations'
    },
    'G': { # Performance, Temporal, Scalability, Availability, Reliability & Platform
        'fast', 'speed', 'response', 'performance', 'time', 'latency', 'throughput',
        'bandwidth', 'capacity', 'scalability', 'efficiency', 'real-time', 'concurrent',
        'parallel', 'millisecond', 'second', 'minute', 'hour', 'load', 'stress',
        'benchmark', 'optimize', 'cache', 'buffer', 'queue', 'rate', 'frequency',
        'memory', 'cpu', 'overhead', 'available', 'availability', 'uptime', 'downtime',
        'reliable', 'reliability', 'fault-tolerant', 'backup', 'recovery', 'redundancy',
        'failover', 'resilient', 'robust', 'mtbf', 'disaster', 'crash', 'platform',
        'operating', 'system', 'linux', 'windows', 'browser', 'hardware', 'server',
        'simultaneous', 'scaleable', 'simultaneously', 'offline', 'bandwidth'
    }
}

MODAL_PATTERN = r'\b(shall|must|will|should|can|may|is required to|expected to)\b'
TEMPORAL_NUM_PATTERN = r'\b(\d+\s*(?:ms|seconds?|minutes?|hours?|users?|%|percent))\b'

def parse_syntax_and_semantics(text):
    text_clean = str(text).strip().strip("'").strip('"')
    tokens = re.findall(r'\b[a-z0-9\-_]{2,}\b', text_clean.lower())
    token_set = set(tokens)
    
    # 1. Syntactic Modal Detection
    modal_match = re.search(MODAL_PATTERN, text_clean, re.IGNORECASE)
    modal_verb = modal_match.group(0).lower() if modal_match else 'none'
    
    # 2. Extract Constraint Clause Signals
    has_temporal_num = bool(re.search(TEMPORAL_NUM_PATTERN, text_clean, re.IGNORECASE))
    
    # 3. Calculate lexical densities across A, T, C, G
    hits = {}
    matched_words = {}
    for base, vocab in LEXICON.items():
        m = token_set.intersection(vocab)
        hits[base] = len(m)
        matched_words[base] = list(m)
        
    # Weighting: Action verbs and imperative modals reinforce A
    if modal_verb in ['shall', 'must', 'will']:
        hits['A'] += 1
        
    # Temporal numbers strongly reinforce G (Performance/Uptime/Load)
    if has_temporal_num:
        hits['G'] += 2
        
    # 4. Syntactic-Semantic Disambiguation Rules:
    # If "user" appears strictly as an agent in a security context (e.g. "user authentication", "authorized users")
    if 'user' in matched_words['T'] or 'users' in matched_words['T']:
        if hits['C'] > 0 and any(w in token_set for w in ['access', 'authorized', 'unauthorized', 'login', 'password', 'permission']):
            hits['T'] = max(0, hits['T'] - 1)
            hits['C'] += 1
            
    # 5. Determine Primary Base X and Secondary Base Y
    # Rank bases by contextual evidence
    sorted_bases = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    top_base, top_score = sorted_bases[0]
    second_base, second_score = sorted_bases[1]
    
    # If no secondary evidence exists or top base overwhelmingly dominates
    if second_score == 0:
        # Canonical monomer / homozygous dinucleotide
        b1 = top_base if top_score > 0 else 'A'
        b2 = b1
        reason = f"Pure {b1} profile dominated by tokens: {', '.join(matched_words[b1][:3]) if matched_words[b1] else modal_verb}"
    else:
        # Heterozygous dinucleotide pair
        # Arrange in standardized alphabetical order (e.g., AT, AG, AC, GC, TC, TG)
        pair = sorted([top_base, second_base])
        b1, b2 = pair[0], pair[1]
        reason = (f"Compound requirement: Primary {top_base} ({', '.join(matched_words[top_base][:2])}) "
                  f"constrained by {second_base} ({', '.join(matched_words[second_base][:2])})")
        
    dinucleotide_code = f"{b1}{b2}"
    
    return {
        'Modal': modal_verb,
        'Base_1': b1,
        'Base_2': b2,
        'Dinucleotide': dinucleotide_code,
        'Score_A': hits['A'],
        'Score_T': hits['T'],
        'Score_C': hits['C'],
        'Score_G': hits['G'],
        'Reasoning': reason
    }

def process_dynamic_mapping(input_csv, text_col, class_col, dataset_name, out_csv):
    print(f"\n[+] Executing Dynamic Syntactic & Semantic Dinucleotide Mapping for {dataset_name}...")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')
        
    results = []
    for idx, row in df.iterrows():
        req_text = str(row[text_col])
        orig_class = str(row[class_col]).strip()
        parsed = parse_syntax_and_semantics(req_text)
        
        results.append({
            'Req_ID': f"{dataset_name}_{idx+1}",
            'Dataset': dataset_name,
            'Original_Class': orig_class,
            'Requirement_Text': req_text,
            'Modal': parsed['Modal'],
            'Primary_Base': parsed['Base_1'],
            'Secondary_Base': parsed['Base_2'],
            'Dinucleotide_Target': parsed['Dinucleotide'],
            'Score_A': parsed['Score_A'],
            'Score_T': parsed['Score_T'],
            'Score_C': parsed['Score_C'],
            'Score_G': parsed['Score_G'],
            'Syntactic_Semantic_Reasoning': parsed['Reasoning']
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"    Saved dynamic mapping audit to: {out_csv}")
    print("    Dynamic Dinucleotide Distribution:")
    print(res_df['Dinucleotide_Target'].value_counts())
    return res_df

if __name__ == '__main__':
    # 1. PROMISE
    p_in = "B:/Promise/Paper2/data/promise_nfr.csv"
    p_out = "B:/Promise/Paper2/knowledge_base/dynamic_promise_dinucleotides.csv"
    process_dynamic_mapping(p_in, 'RequirementText', 'class', 'PROMISE', p_out)
    
    # 2. FNFC
    f_in = "B:/Promise/Paper2/data/fnfc.csv"
    f_out = "B:/Promise/Paper2/knowledge_base/dynamic_fnfc_dinucleotides.csv"
    process_dynamic_mapping(f_in, 'text', 'class', 'FNFC', f_out)
