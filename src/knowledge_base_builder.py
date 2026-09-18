import os
import re
import math
import pandas as pd
import numpy as np

VOCABULARY = {
    'A': {  # Obligation, Action, Functional Provision
        'shall', 'must', 'will', 'provide', 'enable', 'support', 'allow', 'implement',
        'execute', 'process', 'generate', 'display', 'store', 'retrieve', 'transmit',
        'receive', 'validate', 'verify', 'calculate', 'compute', 'update', 'delete',
        'create', 'modify', 'send', 'record', 'maintain', 'manage', 'handle', 'schedule',
        'allocate', 'deliver', 'notify', 'trigger', 'initiate', 'require', 'specify',
        'output', 'input', 'produce', 'export', 'import', 'accept', 'check'
    },
    'T': {  # Usability, Look & Feel, Human-Computer Interaction, Interface Ease
        'user', 'interface', 'usable', 'easy', 'simple', 'intuitive', 'friendly',
        'accessible', 'screen', 'button', 'menu', 'dialog', 'form', 'report',
        'dashboard', 'visualization', 'navigation', 'help', 'documentation', 'training',
        'look', 'feel', 'color', 'font', 'theme', 'style', 'layout', 'readable',
        'visual', 'guideline', 'clear', 'icon', 'graphic', 'presentation', 'aesthetic',
        'appearance', 'view', 'wizard', 'feedback', 'message', 'prompt', 'error-message'
    },
    'C': {  # Security, Protection, Confidentiality, Legal & Policy Compliance
        'secure', 'security', 'authentication', 'authorization', 'encryption', 'privacy',
        'confidential', 'confidentiality', 'integrity', 'audit', 'compliance', 'credential',
        'certificate', 'firewall', 'token', 'session', 'vulnerability', 'threat', 'attack',
        'access', 'permission', 'role', 'protect', 'protection', 'policy', 'legal',
        'regulation', 'statutory', 'gdpr', 'hipaa', 'license', 'copyright', 'breach',
        'tamper', 'safeguard', 'password', 'login', 'logout', 'sign-in', 'restricted'
    },
    'G': {  # Performance, Temporal constraints, Availability, Reliability, Scalability, Architecture
        'fast', 'speed', 'response', 'performance', 'time', 'latency', 'throughput',
        'bandwidth', 'capacity', 'scalability', 'efficiency', 'real-time', 'concurrent',
        'parallel', 'millisecond', 'second', 'minute', 'hour', 'load', 'stress',
        'benchmark', 'optimize', 'cache', 'buffer', 'queue', 'rate', 'frequency',
        'memory', 'cpu', 'overhead', 'available', 'availability', 'uptime', 'downtime',
        'reliable', 'reliability', 'fault-tolerant', 'backup', 'recovery', 'redundancy',
        'failover', 'resilient', 'robust', 'mtbf', 'disaster', 'crash', 'platform',
        'operating', 'system', 'linux', 'windows', 'browser', 'hardware', 'server'
    }
}

CLASS_PRIMARY_HINTS = {
    'F': 'A',
    'US': 'T',
    'SE': 'C',
    'PE': 'G',
    'SC': 'G',
    'LF': 'AT',
    'MN': 'AT',
    'M': 'AT',
    'O': 'AG',
    'PO': 'AG',
    'P': 'AG',
    'L': 'AC',
    'LL': 'AC',
    'A': 'GC',
    'AU': 'GC',
    'FT': 'GC',
    'R': 'GC'
}

def clean_and_tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9\-_]{2,}\b', text)
    return tokens

def analyze_requirement_linguistics(req_text, original_class):
    tokens = clean_and_tokenize(req_text)
    token_set = set(tokens)
    
    evidence = {}
    matched_words = {}
    for base, vocab in VOCABULARY.items():
        matched = token_set.intersection(vocab)
        evidence[base] = len(matched)
        matched_words[base] = sorted(list(matched))
    
    total_matches = sum(evidence.values())
    
    scores = {}
    for base in ['A', 'T', 'C', 'G']:
        scores[base] = evidence[base] / total_matches if total_matches > 0 else 0.0

    orig_clean = str(original_class).strip().upper()
    hint = CLASS_PRIMARY_HINTS.get(orig_clean, None)
    
    rationale = []
    if hint and len(hint) == 2:
        b1, b2 = hint[0], hint[1]
        assigned_dna = hint
        rationale.append(f"Domain Quality Class '{orig_clean}' defines compound requirement.")
        if evidence[b1] > 0:
            rationale.append(f"Found {b1}-evidence ({', '.join(matched_words[b1][:3])}).")
        if evidence[b2] > 0:
            rationale.append(f"Found {b2}-evidence ({', '.join(matched_words[b2][:3])}).")
        if evidence[b1] == 0 and evidence[b2] == 0:
            rationale.append(f"Mapped to {hint} based on ISO-25010 structural specification for class {orig_clean}.")
    elif hint and len(hint) == 1:
        assigned_dna = hint
        rationale.append(f"Domain Quality Class '{orig_clean}' defines orthogonal monomer {hint}.")
        if evidence[hint] > 0:
            rationale.append(f"Direct {hint}-lexical activation ({', '.join(matched_words[hint][:3])}).")
        else:
            rationale.append(f"Adhering to primary functional role {hint}.")
    else:
        # Fallback for unknown classes
        sorted_bases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_base, top_score = sorted_bases[0]
        second_base, second_score = sorted_bases[1]
        if top_score > 0.6:
            assigned_dna = top_base
            rationale.append(f"Dominant lexical density on {top_base} ({top_score:.2f}).")
        elif top_score > 0 and second_score > 0:
            pair = "".join(sorted([top_base, second_base]))
            assigned_dna = pair
            rationale.append(f"Hybrid co-occurrence between {top_base} and {second_base}.")
        else:
            assigned_dna = 'A'
            rationale.append("Defaulted to Adenine (A) as generalized imperative requirement.")
            
    justification_str = " ".join(rationale)
    
    return {
        'assigned_dna': assigned_dna,
        'score_A': round(scores['A'], 3),
        'score_T': round(scores['T'], 3),
        'score_C': round(scores['C'], 3),
        'score_G': round(scores['G'], 3),
        'matched_A': ", ".join(matched_words['A']),
        'matched_T': ", ".join(matched_words['T']),
        'matched_C': ", ".join(matched_words['C']),
        'matched_G': ", ".join(matched_words['G']),
        'justification': justification_str
    }

def process_dataset(input_file, text_col, class_col, dataset_name, out_csv):
    print(f"\n[+] Building Linguistic Knowledge Base for {dataset_name} ({input_file})...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        
    print(f"    Loaded {len(df)} requirements.")
    records = []
    for idx, row in df.iterrows():
        req_id = f"{dataset_name}_{idx+1}"
        text = str(row[text_col])
        cls = str(row[class_col]).strip()
        analysis = analyze_requirement_linguistics(text, cls)
        records.append({
            'Req_ID': req_id,
            'Dataset': dataset_name,
            'Original_Class': cls,
            'Assigned_DNA': analysis['assigned_dna'],
            'Requirement_Text': text,
            'Score_A': analysis['score_A'],
            'Score_T': analysis['score_T'],
            'Score_C': analysis['score_C'],
            'Score_G': analysis['score_G'],
            'Matched_A': analysis['matched_A'],
            'Matched_T': analysis['matched_T'],
            'Matched_C': analysis['matched_C'],
            'Matched_G': analysis['matched_G'],
            'Linguistic_Justification': analysis['justification']
        })
        
    kb_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    kb_df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"    Saved Knowledge Base to {out_csv}")
    print("    DNA Target Distribution:")
    print(kb_df['Assigned_DNA'].value_counts())
    return kb_df

if __name__ == '__main__':
    promise_in = "B:/Promise/Paper2/data/promise_nfr.csv"
    promise_out = "B:/Promise/Paper2/knowledge_base/promise_knowledge_base.csv"
    process_dataset(promise_in, 'RequirementText', 'class', 'PROMISE', promise_out)
    
    fnfc_in = "B:/Promise/Paper2/data/fnfc.csv"
    fnfc_out = "B:/Promise/Paper2/knowledge_base/fnfc_knowledge_base.csv"
    process_dataset(fnfc_in, 'text', 'class', 'FNFC', fnfc_out)
